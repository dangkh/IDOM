"""IDOM-SC (exp_spec_revise.md): cached history encoder + light per-step head, optional score coupling.

Per modality m (id, text), with item table E^m (V x H):
  encoder   : Transformer_rep over the history, run ONCE per batch (no x_t injection)
              S^m = LN(dropout(hidden)) in R^{L x H},  h_last^m = S^m[:, -1]
  step head : x1_hat = h_last + Head(query = MLP(x_t) + modality emb [+ message], cross-attn to S, AdaLN(t))
              last layer zero-initialised -> at init x1_hat = h_last (= baseline B4, the direct predictor)
  sampler   : x_{k+1} = x_k + (x1_hat_k - x_k) / (K (1 - t_k)),  t_k = k/K  ->  x_K = x1_hat_{K-1}
  coupling  : message to receiver m from source n at step k (applied at step k+1):
              r = alpha(q, summaries, t) * (relu(p_n - p_m) @ E^m)   (NOT normalised by sum of weights)
              msg_m = W_msg r + MLP([alpha*TV, H(p_n), H(p_m), margin(p_n)])   (both zero-initialised)
Losses:
  teacher FM branch (shares the encoder): x_t = t e_y + (1-t) z0 -> head(msg=0) -> MSE + alpha*CE (+ beta*decoder)
  rollout branch (K_train > 0): free rollout from z0 exactly as inference (target never enters) ->
              lam_rank * CE(sum_m s_m) + lam_branch * sum_m CE(s_m)
"""
import math
import types
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from fmrec import Transformer_rep
from model import LayerNorm, multi_hot, balanced_mse_loss


def timestep_embedding(t, dim, max_period=10000):
    """Sinusoidal embedding of t in [0, 1] (scaled by 1000, like diffusion step indices)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = (t.float() * 1000.0)[:, None] * freqs[None]
    emb = torch.cat([torch.cos(a), torch.sin(a)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def _zero(module):
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class HeadBlock(nn.Module):
    """Query token cross-attends to the cached history; AdaLN (shift/scale from t) before attn and FFN."""

    def __init__(self, H, heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(H, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(H, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(H, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(H, 4 * H), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * H, H))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(H, 4 * H))
        _zero(self.ada[1])  # identity modulation at init

    def forward(self, tok, S, key_pad, c):
        sh1, sc1, sh2, sc2 = self.ada(c).unsqueeze(1).chunk(4, dim=-1)
        h = self.ln1(tok) * (1 + sc1) + sh1
        a, _ = self.attn(h, S, S, key_padding_mask=key_pad, need_weights=False)
        tok = tok + a
        h = self.ln2(tok) * (1 + sc2) + sh2
        return tok + self.ffn(h)


class StepHead(nn.Module):
    def __init__(self, H, heads, dropout, n_layers):
        super().__init__()
        self.H = H
        self.x_in = nn.Sequential(nn.Linear(H, H), nn.GELU(), nn.Linear(H, H))
        self.t_embed = nn.Sequential(nn.Linear(H, 4 * H), nn.SiLU(), nn.Linear(4 * H, H))
        self.blocks = nn.ModuleList([HeadBlock(H, heads, dropout) for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(H)
        self.out = nn.Linear(H, H)
        _zero(self.out)  # x1_hat = h_last at init

    def forward(self, S, key_pad, x, t, extra):
        c = self.t_embed(timestep_embedding(t, self.H))
        tok = (self.x_in(x) + extra).unsqueeze(1)                 # [N, 1, H]
        for blk in self.blocks:
            tok = blk(tok, S, key_pad, c)
        return self.out(self.norm_out(tok[:, 0]))                 # [N, H]


class ScoreCoupler(nn.Module):
    """Cross-modal messages from step-k scores, expressed with the RECEIVER's item table."""

    def __init__(self, H, modalities, tau, q_const):
        super().__init__()
        self.mods = list(modalities)
        self.tau = tau
        self.q_const = q_const
        M = len(self.mods)
        self.pair_emb = nn.Embedding(M * M, 8)
        self.q_net = nn.Sequential(nn.Linear(H, H), nn.GELU(), nn.Linear(H, H))
        self.q_norm = nn.LayerNorm(H)
        self.gate = nn.Sequential(nn.Linear(H + 3 + 3 + 1 + 8, H), nn.GELU(), nn.Linear(H, 1))
        self.msg_proj = nn.Linear(H, H)
        self.scal_proj = nn.Sequential(nn.Linear(4, H), nn.GELU(), nn.Linear(H, H))
        _zero(self.msg_proj)
        _zero(self.scal_proj[2])

    def intent(self, h):
        q = self.q_norm(self.q_net(h))
        return torch.zeros_like(q) if self.q_const else q

    @staticmethod
    def summary(p):
        V = p.size(-1)
        ent = -(p * torch.log(p.clamp_min(1e-12))).sum(-1) / math.log(V)
        top2 = p.topk(2, dim=-1).values
        return torch.stack([ent, top2[:, 0], top2[:, 0] - top2[:, 1]], dim=-1)   # [B, 3]

    def forward(self, scores, tables, q, t):
        """scores/tables: dict modality -> [B, V] / [V, H]; q: [B, H]; t: float. Returns dict modality -> [B, H]."""
        probs = {m: F.softmax(scores[m] / self.tau, dim=-1) for m in self.mods}
        summ = {m: self.summary(probs[m]) for m in self.mods}
        B, M = q.size(0), len(self.mods)
        tt = torch.full((B, 1), float(t), device=q.device)
        out = {}
        for i, m in enumerate(self.mods):                         # receiver
            r, scal = 0.0, 0.0
            for j, n in enumerate(self.mods):                     # source
                if n == m:
                    continue
                diff = (probs[n] - probs[m]).clamp_min(0)         # [B, V]
                tv = diff.sum(-1, keepdim=True)                   # total variation in [0, 1]
                pe = self.pair_emb.weight[j * M + i].unsqueeze(0).expand(B, -1)
                alpha = torch.sigmoid(self.gate(torch.cat([q, summ[n], summ[m], tt, pe], dim=-1)))   # [B, 1]
                r = r + alpha * (diff @ tables[m])                # sum_i alpha * diff_i * e_i^m
                scal = scal + torch.cat([alpha * tv, summ[n][:, :1], summ[m][:, :1], summ[n][:, 2:3]], dim=-1)
            out[m] = self.msg_proj(r) + self.scal_proj(scal)
        return out


class EHModel(nn.Module):
    MODALITY_IDS = {'id': 0, 'text': 1}

    def __init__(self, args, text_feat=None):
        super().__init__()
        H = args.hidden_size
        self.H, self.item_num = H, args.item_num
        self.modalities = list(args.modalities)
        for m in self.modalities:
            assert m in self.MODALITY_IDS, f'unknown modality {m}'
        self.embed_dropout = nn.Dropout(args.emb_dropout)
        self.modality_embeddings = nn.Embedding(len(self.MODALITY_IDS), H)
        nn.init.normal_(self.modality_embeddings.weight, std=args.modality_emb_std)
        self.layer_norms = nn.ModuleDict({m: LayerNorm(H, eps=1e-12) for m in self.modalities})
        if 'id' in self.modalities:
            self.item_embeddings = nn.Embedding(self.item_num, H)
        if 'text' in self.modalities:
            assert text_feat is not None and text_feat.shape[0] == self.item_num, 'bad / missing --text_emb'
            self.register_buffer('text_feat', text_feat.float())
            self.text_proj = nn.Sequential(nn.Linear(text_feat.shape[1], 2 * H), nn.GELU(),
                                           nn.Dropout(args.dropout), nn.Linear(2 * H, H))

        # history encoder (same module + names as FMRec's FM_xstart.att, so IDOM checkpoints load into it)
        self.encoder = Transformer_rep(args)
        self.enc_dropout = nn.Dropout(args.dropout)
        self.norm_out = LayerNorm(H)
        dims = [H, 512, 2048, self.item_num]                     # FMRec's auxiliary decoder
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.Tanh()]
        self.decoder = nn.Sequential(*layers[:-1])
        for mod in self.decoder:
            if isinstance(mod, nn.Linear):
                nn.init.xavier_normal_(mod.weight)
                nn.init.zeros_(mod.bias)

        self.head_off = args.head_off
        if not self.head_off:
            self.head = StepHead(H, args.head_heads, args.dropout, args.head_layers)
        self.coupling = args.coupling
        if self.coupling:
            assert len(self.modalities) >= 2, 'coupling needs at least two modalities'
            self.coupler = ScoreCoupler(H, self.modalities, args.tau, args.q_const)

        self.fm = types.SimpleNamespace(sample_N=args.sample_N)   # tools set model.fm.sample_N = K_test
        self.K_train, self.x_frozen = args.K_train, args.x_frozen
        self.msg_off_eval, self.grad_step0 = args.msg_off_eval, args.grad_step0
        self.lam_flow, self.lam_rank, self.lam_branch = args.lam_flow, args.lam_rank, args.lam_branch
        self.Loss_Alpha, self.Loss_Beta, self.mask_ratio = args.Loss_Alpha, args.Loss_Beta, args.mask_ratio
        self.eps, self.s_modsamp = args.eps, args.s_modsamp
        self.loss_ce, self.loss_mse = nn.CrossEntropyLoss(), nn.MSELoss()

    # ---------------- representation ----------------
    def item_table(self, m):
        return self.item_embeddings.weight if m == 'id' else self.text_proj(self.text_feat)

    def seq_rep(self, m, sequence, table):
        rep = self.layer_norms[m](self.embed_dropout(F.embedding(sequence, table)))
        return rep + self.modality_embeddings.weight[self.MODALITY_IDS[m]]

    def encode(self, sequence):
        B, M = sequence.size(0), len(self.modalities)
        tables = {m: self.item_table(m) for m in self.modalities}            # once per batch
        rep = torch.cat([self.seq_rep(m, sequence, tables[m]) for m in self.modalities], 0)
        mask = (sequence > 0).float().repeat(M, 1)
        hidden, encoded = self.encoder(rep, rep, mask)
        S = self.norm_out(self.enc_dropout(hidden))                            # [MB, L, H]
        mod_ids = torch.cat([torch.full((B,), self.MODALITY_IDS[m], device=sequence.device, dtype=torch.long)
                             for m in self.modalities])
        return types.SimpleNamespace(B=B, M=M, tables=tables, S=S, h_last=S[:, -1], enc_last=encoded[:, -1],
                                     key_pad=(sequence == 0).repeat(M, 1), mod_emb=self.modality_embeddings(mod_ids))

    def sl(self, enc, i):
        return slice(i * enc.B, (i + 1) * enc.B)

    def predict_x1(self, enc, x, t, msg=None):
        if self.head_off:
            return enc.h_last
        extra = enc.mod_emb if msg is None else enc.mod_emb + msg
        return enc.h_last + self.head(enc.S, enc.key_pad, x, t, extra)

    def sample_t(self, n, device):
        u = torch.rand(n, device=device)
        t = 1 - u - self.s_modsamp * (torch.cos((math.pi / 2) * u) ** 2 - 1 + u)   # FMRec 'mode' sampling
        return t.clamp(0, 1) * (1 - self.eps) + self.eps

    # ---------------- rollout (identical in training and inference) ----------------
    def rollout(self, enc, K, use_msg):
        MB = enc.B * enc.M
        dev = enc.S.device
        x = torch.zeros(MB, self.H, device=dev) if self.x_frozen else torch.randn(MB, self.H, device=dev)
        q = self.coupler.intent(enc.h_last[self.sl(enc, 0)]) if (self.coupling and use_msg) else None
        msg, scores = None, None
        for k in range(K):
            t = k / K
            tvec = torch.full((MB,), t, device=dev)
            skip_grad = self.training and k == 0 and K > 1 and not self.grad_step0
            with torch.no_grad() if skip_grad else nullcontext():
                x1 = self.predict_x1(enc, x, tvec, msg)
            scores = {m: x1[self.sl(enc, i)] @ enc.tables[m].t() for i, m in enumerate(self.modalities)}
            if k < K - 1:
                if self.coupling and use_msg:
                    msgs = self.coupler(scores, enc.tables, q, t)
                    msg = torch.cat([msgs[m] for m in self.modalities], 0)
                if not self.x_frozen:
                    x = x + (x1 - x) / (K * (1 - t))
        return scores

    # ---------------- trainer interface ----------------
    def compute_loss(self, sequence, tag):
        enc = self.encode(sequence)
        y = tag.squeeze(-1)
        tgt = torch.cat([enc.tables[m][y] for m in self.modalities], 0)          # [MB, H]
        if self.head_off:
            x1 = enc.h_last
        else:
            t = self.sample_t(tgt.size(0), tgt.device)
            z0 = torch.randn_like(tgt)
            x_t = t[:, None] * tgt + (1 - t[:, None]) * z0
            x1 = self.predict_x1(enc, x_t, t, None)
        decode = self.decoder(enc.enc_last) if self.Loss_Beta > 0 else None
        seq_matrix = multi_hot(sequence, self.item_num) if decode is not None else None

        total, logs = 0.0, {}
        for i, m in enumerate(self.modalities):
            s = self.sl(enc, i)
            l_fm = self.loss_mse(x1[s], tgt[s])
            l_ce = self.loss_ce(x1[s] @ enc.tables[m].t(), y)
            l = l_fm + self.Loss_Alpha * l_ce
            logs.update({f'{m}/fm_mse': l_fm.item(), f'{m}/ce': l_ce.item()})
            if decode is not None:
                l_dec = balanced_mse_loss(self.loss_mse, seq_matrix, decode[s], self.mask_ratio)
                l = l + self.Loss_Beta * l_dec
                logs[f'{m}/dec'] = l_dec.item()
            total = total + self.lam_flow * l

        if self.K_train > 0 and not self.head_off:
            scores = self.rollout(enc, self.K_train, use_msg=True)
            l_rank = self.loss_ce(sum(scores.values()), y)
            l_br = sum(self.loss_ce(scores[m], y) for m in self.modalities)
            total = total + self.lam_rank * l_rank + self.lam_branch * l_br
            logs.update({'roll/ce_fused': l_rank.item(), 'roll/ce_branch': l_br.item()})
        return total, logs

    def predict_scores(self, sequence):
        enc = self.encode(sequence)
        if self.head_off:
            scores = {m: enc.h_last[self.sl(enc, i)] @ enc.tables[m].t() for i, m in enumerate(self.modalities)}
        else:
            scores = self.rollout(enc, max(1, int(self.fm.sample_N)), use_msg=not self.msg_off_eval)
        out = dict(scores)
        out['fused'] = sum(scores.values())   # raw sum (rank-equivalent to the sum of log-softmaxes)
        return out


# ---------------- warm start ----------------
_REMAP = [('fm.xstart_model.att.', 'encoder.'), ('fm.xstart_model.norm_fm_rep.', 'norm_out.'),
          ('fm.xstart_model.decoder.', 'decoder.')]


def load_warm_start(model, path):
    """Load matching weights from an IDOM (renamed) or EH checkpoint; new modules keep their init.
    Marks loaded parameters in model.pretrained_names (used for a lower learning rate)."""
    sd = torch.load(path, map_location='cpu', weights_only=False)['model']
    renamed = {}
    for k, v in sd.items():
        for a, b in _REMAP:
            if k.startswith(a):
                k = b + k[len(a):]
                break
        renamed[k] = v
    own = model.state_dict()
    loadable = {k: v for k, v in renamed.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(loadable, strict=False)
    param_names = {n for n, _ in model.named_parameters()}
    model.pretrained_names = set(loadable) & param_names
    not_loaded = sorted(n for n in param_names if n not in loadable)
    return len(model.pretrained_names), not_loaded
