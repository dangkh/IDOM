"""Choose the IDOM score-fusion weight on VALIDATION, from a saved best.pt (no retraining).

fused = a * log_softmax(score_id) + (1 - a) * log_softmax(score_text),  a in --alphas
a = 0.5 is what training used (weights 1:1); a = 1 / a = 0 are the single trajectories.
The ODE start noise is drawn once per batch and shared by all alphas, so they are compared fairly.

    python tools/fuse_sweep.py --run amazon_beauty_idom_id-text_seed1997
    python tools/fuse_sweep.py --run amazon_beauty_idom_id-text_seed1997 --select HR@10
Writes src/log/<dataset>/<run>_fusesweep.json
"""
import argparse
import json
import os
import pickle
import sys
from argparse import Namespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'src')
sys.path.insert(0, SRC)

ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True, help='run name = folder in src/ckpt/<dataset>/')
ap.add_argument('--dataset', default='amazon_beauty')
ap.add_argument('--alphas', nargs='+', type=float, default=[round(x, 2) for x in np.arange(0, 1.0001, 0.1)])
ap.add_argument('--select', default='NDCG@10', help='validation metric used to pick alpha')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--device', default='cuda')
opt = ap.parse_args()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from utils import Data_Val, Data_Test  # noqa: E402
from model import create_model_FM, IDOM_model  # noqa: E402
from trainer import hrs_and_ndcgs_k  # noqa: E402

os.chdir(SRC)  # paths stored in the checkpoint args are relative to src/
device = opt.device if torch.cuda.is_available() else 'cpu'
ck = torch.load(os.path.join('ckpt', opt.dataset, opt.run, 'best.pt'), map_location=device, weights_only=False)
a = Namespace(**ck['args'])
a.device = device
assert a.model == 'idom' and len(a.modalities) == 2, 'fuse sweep needs an IDOM run with two modalities'
print(f'run {opt.run}: best checkpoint from epoch {ck["epoch"]}, modalities {a.modalities}')

with open(os.path.join('..', 'datasets', 'data', a.dataset, 'dataset.pkl'), 'rb') as f:
    d = pickle.load(f)
loaders = {'val': Data_Val(d['train'], d['val'], a).get_pytorch_dataloaders(),
           'test': Data_Test(d['train'], d['val'], d['test'], a).get_pytorch_dataloaders()}
text_feat = torch.from_numpy(np.load(a.text_emb)).float() if 'text' in a.modalities else None
model = IDOM_model(create_model_FM(a), a, text_feat).to(device)
model.load_state_dict(ck['model'])
model.eval()

m0, m1 = a.modalities
torch.manual_seed(opt.seed)
res = {}
with torch.no_grad():
    for split, loader in loaders.items():
        acc = {al: {} for al in opt.alphas}
        for seq, labels in loader:
            seq, labels = seq.to(device), labels.to(device)
            out = model.predict_scores(seq)
            l0, l1 = F.log_softmax(out[m0], -1), F.log_softmax(out[m1], -1)
            for al in opt.alphas:
                for k, v in hrs_and_ndcgs_k(al * l0 + (1 - al) * l1, labels, a.metric_ks).items():
                    acc[al].setdefault(k, []).append(v)
        res[split] = {al: {k: round(float(np.mean(v)) * 100, 4) for k, v in acc[al].items()} for al in opt.alphas}

best_alpha = max(opt.alphas, key=lambda al: res['val'][al][opt.select])
print(f'\nalpha = weight of "{m0}" (1-alpha on "{m1}")')
print(f'{"alpha":>6} | ' + '  '.join(f'val {k}' for k in ('HR@10', 'NDCG@10')) + ' | '
      + '  '.join(f'test {k}' for k in ('HR@10', 'NDCG@10', 'HR@20', 'NDCG@20')))
for al in opt.alphas:
    v, t = res['val'][al], res['test'][al]
    mark = '  <- best on val ' + opt.select if al == best_alpha else ''
    print(f'{al:>6.2f} | {v["HR@10"]:>10.2f}  {v["NDCG@10"]:>11.2f} | {t["HR@10"]:>11.2f}  {t["NDCG@10"]:>12.2f}  '
          f'{t["HR@20"]:>11.2f}  {t["NDCG@20"]:>12.2f}{mark}')

out_path = os.path.join(a.log_file, opt.dataset, f'{opt.run}_fusesweep.json')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w') as f:
    json.dump({'run': opt.run, 'ckpt_epoch': ck['epoch'], 'select': opt.select, 'best_alpha': best_alpha,
               'test_at_best_alpha': res['test'][best_alpha],
               'results': {s: {str(al): r for al, r in rr.items()} for s, rr in res.items()}}, f, indent=2)
print('\nsaved', out_path)
