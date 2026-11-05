import torch.nn as nn
import torch
import math
from fmrec import FMRec
import torch.nn.functional as F
import numpy as np
import torch as th


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


def multi_hot(sequence, num_items):
    """B x L item ids -> B x N multi-hot matrix (padding id 0 excluded).

    Vectorised replacement of FMRec's switch_Matrix python loop; produces the same matrix."""
    m = torch.zeros(sequence.size(0), num_items, device=sequence.device)
    m.scatter_(1, sequence, 1.0)
    m[:, 0] = 0.0
    return m


def balanced_mse_loss(loss_mse, target, output, mask_ratio=1.0):
    """FMRec's auxiliary decoder loss (copied unchanged)."""
    num_ones = torch.sum(target == 1).item()
    num_zeros = torch.sum(target == 0).item()
    num_selected_zeros = int(min(num_zeros, num_ones * mask_ratio))
    zero_rows, zero_cols = (target == 0).nonzero(as_tuple=True)
    one_positions = (target == 1).nonzero(as_tuple=True)
    selected_zero_indices = torch.randint(0, num_zeros, (num_selected_zeros,), device=target.device)
    selected_zero_positions = (zero_rows[selected_zero_indices], zero_cols[selected_zero_indices])
    mask = torch.zeros_like(target)
    mask[selected_zero_positions] = 1
    mask[one_positions] = 1
    return loss_mse(output * mask, target * mask)


# ---------------------------------------------------------------------------
# FMRec baseline (ID only) - original Att_FM_model, plus a common
# compute_loss / predict_scores interface used by the trainer.
# ---------------------------------------------------------------------------
class Att_FM_model(nn.Module):
    def __init__(self, fm, args):
        super(Att_FM_model, self).__init__()
        self.emb_dim = args.hidden_size
        self.item_num = args.item_num
        self.item_embeddings = nn.Embedding(self.item_num, self.emb_dim)
        self.embed_dropout = nn.Dropout(args.emb_dropout)
        self.position_embeddings = nn.Embedding(args.max_len, args.hidden_size)  # unused, kept as in FMRec
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.dropout)
        self.fm = fm
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_mse = nn.MSELoss()
        self.mask_ratio = args.mask_ratio
        self.Loss_Alpha = args.Loss_Alpha
        self.Loss_Beta = args.Loss_Beta

    def reverse(self, item_rep, z0, mask_seq):
        return self.fm.reverse_p_sample_rf(item_rep, z0, mask_seq)

    def loss_fm_ce(self, rep_fm, labels):
        scores = torch.matmul(rep_fm, self.item_embeddings.weight.t())
        return self.loss_ce(scores, labels.squeeze(-1))

    def fm_rep_pre(self, rep_fm):
        return torch.matmul(rep_fm, self.item_embeddings.weight.t())

    def forward(self, sequence, tag, train_flag=True):
        item_embeddings = self.item_embeddings(sequence)
        item_embeddings = self.embed_dropout(item_embeddings)
        item_embeddings = self.LayerNorm(item_embeddings)
        mask_seq = (sequence > 0).float()

        if train_flag:
            tag_emb = self.item_embeddings(tag.squeeze(-1))  # B x H
            rep_fm, decode_out, t_rf_expand, t_rf, z0 = self.fm(item_embeddings, tag_emb, mask_seq)
            seq_Matrix = multi_hot(sequence, self.item_num)
            loss_dec = balanced_mse_loss(self.loss_mse, seq_Matrix, decode_out, self.mask_ratio)
            loss_FM_mse = self.loss_mse(rep_fm, tag_emb)
            return loss_dec, rep_fm, loss_FM_mse
        else:
            z0 = th.randn_like(item_embeddings[:, -1, :])
            rep_fm = self.reverse(item_embeddings, z0, mask_seq)
            return None, rep_fm, None

    # common interface -----------------------------------------------------
    def compute_loss(self, sequence, tag):
        loss_dec, rep_fm, loss_FM_mse = self.forward(sequence, tag, train_flag=True)
        loss_ce = self.loss_fm_ce(rep_fm, tag)
        total = loss_FM_mse + self.Loss_Alpha * loss_ce + self.Loss_Beta * loss_dec
        return total, {'id/fm_mse': loss_FM_mse.item(), 'id/ce': loss_ce.item(), 'id/dec': loss_dec.item()}

    def predict_scores(self, sequence):
        _, rep_fm, _ = self.forward(sequence, None, train_flag=False)
        return {'fused': self.fm_rep_pre(rep_fm)}


# ---------------------------------------------------------------------------
# IDOM (no intent anchor yet)
#   - one flow trajectory per modality (ID space, text space, ...)
#   - all trajectories share ONE vector field (the FMRec FM_xstart network);
#     a learnable modality embedding added to the condition tells it which space it is in
#   - modalities are batched together (M*B rows), so one forward pass per step
#   - each modality is trained with the FMRec objective in its own space
#   - inference: Euler ODE per modality, scores fused by weighted log-softmax
# ---------------------------------------------------------------------------
class IDOM_model(nn.Module):
    MODALITY_IDS = {'id': 0, 'text': 1}

    def __init__(self, fm, args, text_feat=None):
        super(IDOM_model, self).__init__()
        H = args.hidden_size
        self.emb_dim = H
        self.item_num = args.item_num
        self.modalities = list(args.modalities)
        for m in self.modalities:
            assert m in self.MODALITY_IDS, f'unknown modality {m}'
        self.fm = fm
        self.embed_dropout = nn.Dropout(args.emb_dropout)
        self.mask_ratio = args.mask_ratio
        self.Loss_Alpha = args.Loss_Alpha
        self.Loss_Beta = args.Loss_Beta
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_mse = nn.MSELoss()

        self.modality_embeddings = nn.Embedding(len(self.MODALITY_IDS), H)
        nn.init.normal_(self.modality_embeddings.weight, std=args.modality_emb_std)
        self.use_modality_emb = not args.no_modality_emb
        self.layer_norms = nn.ModuleDict({m: LayerNorm(H, eps=1e-12) for m in self.modalities})

        if 'id' in self.modalities:
            self.item_embeddings = nn.Embedding(self.item_num, H)
        if 'text' in self.modalities:
            assert text_feat is not None, '--text_emb is required when text is a modality'
            assert text_feat.shape[0] == self.item_num, \
                f'text_emb has {text_feat.shape[0]} rows, expected item_num={self.item_num} (row 0 = padding)'
            self.register_buffer('text_feat', text_feat.float())
            d = text_feat.shape[1]
            self.text_proj = nn.Sequential(
                nn.Linear(d, 2 * H), nn.GELU(), nn.Dropout(args.dropout), nn.Linear(2 * H, H))

        w = args.fuse_weights if args.fuse_weights else [1.0] * len(self.modalities)
        assert len(w) == len(self.modalities), '--fuse_weights must have one weight per modality'
        self.fuse_weights = [float(x) for x in w]

    # item representation table of one modality: N x H
    def item_table(self, m):
        if m == 'id':
            return self.item_embeddings.weight
        if m == 'text':
            return self.text_proj(self.text_feat)
        raise ValueError(m)

    def seq_rep(self, m, sequence, table):
        rep = F.embedding(sequence, table)
        rep = self.embed_dropout(rep)
        rep = self.layer_norms[m](rep)
        if self.use_modality_emb:
            rep = rep + self.modality_embeddings.weight[self.MODALITY_IDS[m]]
        return rep

    def _stack(self, sequence, tag=None):
        tables, reps, tags = [], [], []
        for m in self.modalities:
            table = self.item_table(m)
            tables.append(table)
            reps.append(self.seq_rep(m, sequence, table))
            if tag is not None:
                tags.append(table[tag.squeeze(-1)])
        return tables, reps, tags

    def compute_loss(self, sequence, tag):
        B, M = sequence.size(0), len(self.modalities)
        mask_seq = (sequence > 0).float()
        tables, reps, tags = self._stack(sequence, tag)
        rep_all = torch.cat(reps, 0)                # (M*B) x L x H
        tag_all = torch.cat(tags, 0)                # (M*B) x H
        mask_all = mask_seq.repeat(M, 1)
        x1_pred, decode_out, _, _, _ = self.fm(rep_all, tag_all, mask_all)

        seq_matrix = multi_hot(sequence, self.item_num)
        labels = tag.squeeze(-1)
        total, logs = 0.0, {}
        for i, m in enumerate(self.modalities):
            sl = slice(i * B, (i + 1) * B)
            l_fm = self.loss_mse(x1_pred[sl], tags[i])
            l_ce = self.loss_ce(torch.matmul(x1_pred[sl], tables[i].t()), labels)
            l_dec = balanced_mse_loss(self.loss_mse, seq_matrix, decode_out[sl], self.mask_ratio)
            total = total + l_fm + self.Loss_Alpha * l_ce + self.Loss_Beta * l_dec
            logs.update({f'{m}/fm_mse': l_fm.item(), f'{m}/ce': l_ce.item(), f'{m}/dec': l_dec.item()})
        return total, logs

    def predict_scores(self, sequence):
        B, M = sequence.size(0), len(self.modalities)
        mask_seq = (sequence > 0).float()
        tables, reps, _ = self._stack(sequence)
        rep_all = torch.cat(reps, 0)
        z0 = th.randn(M * B, self.emb_dim, device=sequence.device)
        x_all = self.fm.reverse_p_sample_rf(rep_all, z0, mask_seq.repeat(M, 1))
        out, fused = {}, 0.0
        for i, m in enumerate(self.modalities):
            s = torch.matmul(x_all[i * B:(i + 1) * B], tables[i].t())
            out[m] = s
            fused = fused + self.fuse_weights[i] * F.log_softmax(s, dim=-1)
        out['fused'] = fused
        return out


def create_model_FM(args):
    return FMRec(args)
