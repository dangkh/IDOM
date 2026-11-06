"""B2 baseline and rescue/harm analysis: fuse the scores of two checkpoints, weight chosen on VALIDATION.

fused = s_A + lam * s_B  (raw scores; any user-independent weighting of log-softmaxes ranks the same way).
lam is picked on validation (--select), then test is reported once for that lam.

  # separate-model ensemble: FMRec (ID) + text-only model
  python tools/ensemble_eval.py --a amazon_beauty_fmrec_id_seed1997_v3 --b amazon_beauty_idom_text_seed1997_v3
  # IDOM's own branches (same checkpoint, two score keys)
  python tools/ensemble_eval.py --a amazon_beauty_idom_id-text_seed1997_v3 --a_key id \\
                                --b amazon_beauty_idom_id-text_seed1997_v3 --b_key text
Rescue/harm (test, @10) is measured against model A's own ranking:
  rescue = #(A wrong, fused right) / #(A wrong),   harm = #(A right, fused wrong) / #(A right)
Writes src/log/<dataset>/ensemble_<A>__<B>.json
"""
import argparse
import json
import os
import pickle
import sys
from argparse import Namespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, '..', 'src'))
sys.path.insert(0, SRC)

ap = argparse.ArgumentParser()
ap.add_argument('--a', required=True, help='run name of model A (src/ckpt/<dataset>/<run>/best.pt)')
ap.add_argument('--b', required=True, help='run name of model B')
ap.add_argument('--a_key', default='fused', help="score key of A: fused | id | text")
ap.add_argument('--b_key', default='fused')
ap.add_argument('--dataset', default='amazon_beauty')
ap.add_argument('--lams', nargs='+', type=float,
                default=[0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0])
ap.add_argument('--select', default='NDCG@10')
ap.add_argument('--steps', type=int, default=None, help='Euler steps for both models (default: their saved sample_N)')
ap.add_argument('--seed', type=int, default=0)
opt = ap.parse_args()

import torch  # noqa: E402
from utils import Data_Val, Data_Test  # noqa: E402
from model import build_model  # noqa: E402
from trainer import hrs_and_ndcgs_k  # noqa: E402

os.chdir(SRC)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load(run):
    ck = torch.load(os.path.join('ckpt', opt.dataset, run, 'best.pt'), map_location=device, weights_only=False)
    a = Namespace(**ck['args'])
    a.device = device
    m = build_model(a).to(device)
    m.load_state_dict(ck['model'])
    m.eval()
    if opt.steps is not None:
        m.fm.sample_N = opt.steps
    return m, a


model_a, args_a = load(opt.a)
model_b, _ = (model_a, None) if opt.b == opt.a else load(opt.b)
with open(os.path.join('..', 'datasets', 'data', args_a.dataset, 'dataset.pkl'), 'rb') as f:
    d = pickle.load(f)
loaders = {'val': Data_Val(d['train'], d['val'], args_a).get_pytorch_dataloaders(),
           'test': Data_Test(d['train'], d['val'], d['test'], args_a).get_pytorch_dataloaders()}


def hit10(scores, labels):
    top = torch.topk(scores, 10, dim=-1).indices
    return (top == labels).any(-1).cpu().numpy()


res, per_user = {}, {}
torch.manual_seed(opt.seed)
with torch.no_grad():
    for split, loader in loaders.items():
        acc = {lam: {} for lam in opt.lams}
        hits = {'A': [], 'B': [], **{lam: [] for lam in opt.lams}}
        for seq, labels in loader:
            seq, labels = seq.to(device), labels.to(device)
            out_a = model_a.predict_scores(seq)
            out_b = out_a if model_b is model_a else model_b.predict_scores(seq)
            sa, sb = out_a[opt.a_key], out_b[opt.b_key]
            hits['A'].append(hit10(sa, labels))
            hits['B'].append(hit10(sb, labels))
            for lam in opt.lams:
                fused = sa + lam * sb
                for k, v in hrs_and_ndcgs_k(fused, labels, args_a.metric_ks).items():
                    acc[lam].setdefault(k, []).append(v)
                hits[lam].append(hit10(fused, labels))
        res[split] = {lam: {k: round(float(np.mean(v)) * 100, 4) for k, v in acc[lam].items()} for lam in opt.lams}
        per_user[split] = {k: np.concatenate(v) for k, v in hits.items()}

best = max(opt.lams, key=lambda lam: res['val'][lam][opt.select])
hA, hB, hF = per_user['test']['A'], per_user['test']['B'], per_user['test'][best]
groups = {'A right, B right': hA & hB, 'A right, B wrong': hA & ~hB,
          'A wrong, B right (oracle rescue)': ~hA & hB, 'A wrong, B wrong': ~hA & ~hB}
rescue = float((~hA & hF).sum() / max(1, (~hA).sum()))
harm = float((hA & ~hF).sum() / max(1, hA.sum()))
net = int((~hA & hF).sum() - (hA & ~hF).sum())

print(f'A = {opt.a} [{opt.a_key}]\nB = {opt.b} [{opt.b_key}]\nfused = s_A + lam * s_B\n')
print(f'{"lam":>6} | {"val HR@10":>9} {"val NDCG@10":>11} | {"test HR@10":>10} {"test NDCG@10":>12} {"test HR@20":>10} {"test NDCG@20":>12}')
for lam in opt.lams:
    v, t = res['val'][lam], res['test'][lam]
    mark = f'  <- best on val {opt.select}' if lam == best else ''
    print(f'{lam:>6} | {v["HR@10"]:>9.2f} {v["NDCG@10"]:>11.2f} | {t["HR@10"]:>10.2f} {t["NDCG@10"]:>12.2f} '
          f'{t["HR@20"]:>10.2f} {t["NDCG@20"]:>12.2f}{mark}')
print(f'\nTest @10, per user (n={len(hA)}), relative to A:')
for g, msk in groups.items():
    print(f'  {g:<34} {int(msk.sum()):>6}  fused right: {int((msk & hF).sum()):>6}')
print(f'  rescue = {rescue:.3f}   harm = {harm:.3f}   net rescue = {net:+d} users')

out = os.path.join('log', opt.dataset, f'ensemble_{opt.a}-{opt.a_key}__{opt.b}-{opt.b_key}.json')
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, 'w') as f:
    json.dump({'a': opt.a, 'a_key': opt.a_key, 'b': opt.b, 'b_key': opt.b_key, 'select': opt.select,
               'best_lam': best, 'test_at_best': res['test'][best],
               'rescue': rescue, 'harm': harm, 'net_rescue': net,
               'groups': {g: int(m.sum()) for g, m in groups.items()},
               'results': {s: {str(l): r for l, r in rr.items()} for s, rr in res.items()}}, f, indent=2)
print('\nsaved', os.path.join('src', out))
