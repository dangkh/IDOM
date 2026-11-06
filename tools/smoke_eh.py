"""Smoke tests for the eh model (exp_spec_revise.md §8) + a quick speed comparison. Run before any eh training:

    python tools/smoke_eh.py                 # GPU if available (a few minutes); CPU works too (slower)
Exit code 1 if any test fails.
  T1 head zero-init:       x1_hat == h_last at initialisation
  T2 sampler:              x_K == x1_hat at the last step
  T3 message:              identical modality distributions -> zero message (r part)
  T4 gradient path:        loss reaches head, coupler (gate, msg_proj, scal_proj) and encoder
  T5 no target in rollout: rollout() takes no target argument
  T6 warm start:           IDOM v3 checkpoint loads; eh (head at init) ranks like IDOM (top-10 overlap)
  T7 speed:                ms per training batch, idom vs eh0 vs eh_c4 (K_train=2, coupling)
"""
import inspect
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, '..', 'src'))
sys.path.insert(0, SRC)
import torch  # noqa: E402
from config import build_parser  # noqa: E402
from utils import Data_Train, Data_Val  # noqa: E402
from model import build_model  # noqa: E402
from eh_model import load_warm_start  # noqa: E402

os.chdir(SRC)
torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
D = 'amazon_beauty'
EMB = f'../datasets/data/{D}/text_emb_all-MiniLM-L6-v2.npy'
with open(f'../datasets/data/{D}/dataset.pkl', 'rb') as f:
    d = pickle.load(f)
base = ['--dataset', D, '--modalities', 'id', 'text', '--text_emb', EMB]
results = []


def args_for(extra):
    a = build_parser().parse_args(base + extra)
    a.device, a.item_num, a.user_num = device, len(d['smap']) + 1, len(d['train']) + 1
    return a


def report(name, ok, info=''):
    results.append(ok)
    print(f'[{"PASS" if ok else "FAIL"}] {name} {info}', flush=True)


a0 = args_for(['--model', 'eh', '--K_train', '2', '--coupling'])
seq, tag = next(iter(Data_Train(d['train'], a0).get_pytorch_dataloaders()))
seq, tag = seq[:128].to(device), tag[:128].to(device)
model = build_model(a0).to(device)
print(f'device {device}; eh_c4 params {sum(p.numel() for p in model.parameters()):,}')

# T1
model.eval()
with torch.no_grad():
    enc = model.encode(seq)
    MB = enc.B * enc.M
    x1 = model.predict_x1(enc, torch.randn(MB, model.H, device=device), torch.rand(MB, device=device))
report('T1 head zero-init', torch.allclose(x1, enc.h_last, atol=1e-6), f'max|diff|={(x1 - enc.h_last).abs().max():.2e}')

# make the head state-dependent for the next tests
with torch.no_grad():
    for p in [model.head.out.weight, model.coupler.msg_proj.weight, model.coupler.scal_proj[2].weight]:
        p.normal_(0, 0.05)

# T2
with torch.no_grad():
    K, x = 3, torch.randn(MB, model.H, device=device)
    for k in range(K):
        t = k / K
        x1 = model.predict_x1(enc, x, torch.full((MB,), t, device=device))
        x = x + (x1 - x) / (K * (1 - t))
report('T2 sampler x_K == x1_hat(last)', torch.allclose(x, x1, atol=1e-5), f'max|diff|={(x - x1).abs().max():.2e}')

# T3
with torch.no_grad():
    s = enc.h_last[:enc.B] @ enc.tables['id'].t()
    probs = torch.softmax(s, -1)
    diff = (probs - probs).clamp_min(0)
    r = diff @ enc.tables['text']
report('T3 identical distributions -> zero r', float(r.abs().max()) == 0.0)

# T4
model.train()
model.zero_grad()
loss, logs = model.compute_loss(seq, tag)
loss.backward()


def gnorm(mod):
    return sum(float(p.grad.abs().sum()) for p in mod.parameters() if p.grad is not None)


g = {'head.x_in': gnorm(model.head.x_in), 'coupler.gate': gnorm(model.coupler.gate),
     'coupler.msg_proj': gnorm(model.coupler.msg_proj), 'coupler.scal_proj': gnorm(model.coupler.scal_proj),
     'encoder': gnorm(model.encoder)}
report('T4 gradient reaches head/coupler/encoder', all(v > 0 for v in g.values()),
       ' '.join(f'{k}={v:.2e}' for k, v in g.items()) + f' | loss={float(loss):.3f} ' + str({k: round(v, 3) for k, v in logs.items()}))

# T5
report('T5 rollout has no target argument', 'tag' not in inspect.signature(model.rollout).parameters
       and 'y' not in inspect.signature(model.rollout).parameters)

# T6
ck = f'ckpt/{D}/{D}_idom_id-text_seed1997_v3/best.pt'
if os.path.exists(ck):
    from argparse import Namespace
    ia = Namespace(**torch.load(ck, map_location='cpu', weights_only=False)['args'])
    ia.device = device
    idom = build_model(ia).to(device)
    idom.load_state_dict(torch.load(ck, map_location=device, weights_only=False)['model'])
    idom.eval()
    eh = build_model(args_for(['--model', 'eh'])).to(device)
    n, not_loaded = load_warm_start(eh, ck)
    eh.to(device).eval()
    vseq, _ = next(iter(Data_Val(d['train'], d['val'], a0).get_pytorch_dataloaders()))
    vseq = vseq.to(device)
    with torch.no_grad():
        si, se = idom.predict_scores(vseq), eh.predict_scores(vseq)
    ov = {m: float((torch.topk(si[m], 10).indices.unsqueeze(-1) == torch.topk(se[m], 10).indices.unsqueeze(-2))
                   .any(-1).float().mean()) for m in ['id', 'text']}
    report('T6 warm start from IDOM v3', min(ov.values()) > 0.9,
           f'{n} params loaded; new: {not_loaded[:6]}{"..." if len(not_loaded) > 6 else ""}; top-10 overlap {ov}')
else:
    print(f'[SKIP] T6 warm start: {ck} not found')

# T7
def ms_per_batch(extra, n=10):
    a = args_for(extra)
    m = build_model(a).to(device)
    m.train()
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    big_seq, big_tag = next(iter(Data_Train(d['train'], a).get_pytorch_dataloaders()))
    big_seq, big_tag = big_seq.to(device), big_tag.to(device)
    for i in range(n + 3):
        if i == 3:
            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
        opt.zero_grad()
        lo, _ = m.compute_loss(big_seq, big_tag)
        lo.backward()
        opt.step()
    if device == 'cuda':
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


t_idom = ms_per_batch(['--model', 'idom'])
t_eh0 = ms_per_batch(['--model', 'eh'])
t_c4 = ms_per_batch(['--model', 'eh', '--K_train', '2', '--coupling'])
print(f'[INFO] T7 ms/batch (batch 512): idom {t_idom:.1f} | eh0 {t_eh0:.1f} ({t_eh0 / t_idom:.2f}x) | '
      f'eh_c4 {t_c4:.1f} ({t_c4 / t_idom:.2f}x)')

print(f'\n{sum(results)}/{len(results)} tests passed')
sys.exit(0 if all(results) else 1)
