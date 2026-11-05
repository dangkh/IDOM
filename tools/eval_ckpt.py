"""Accuracy AND time vs number of Euler steps, from a trained checkpoint (src/ckpt/<dataset>/<run>/best.pt).

Answers: does the ODE trajectory matter, or does 1 step give the same ranking as 30?

    python tools/eval_ckpt.py --run amazon_beauty_fmrec_id_seed1997_v2
    python tools/eval_ckpt.py --run amazon_beauty_idom_id-text_seed1997_v2 --steps 1 2 5 10 30
Writes src/log/<dataset>/<run>_steps.json
"""
import argparse
import json
import os
import pickle
import sys
import time
from argparse import Namespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, '..', 'src'))
sys.path.insert(0, SRC)

ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True)
ap.add_argument('--dataset', default='amazon_beauty')
ap.add_argument('--steps', nargs='+', type=int, default=[1, 2, 5, 10, 30])
ap.add_argument('--seed', type=int, default=0)
opt = ap.parse_args()

import torch  # noqa: E402
from utils import Data_Val, Data_Test  # noqa: E402
from model import create_model_FM, Att_FM_model, IDOM_model  # noqa: E402
from trainer import evaluate  # noqa: E402

os.chdir(SRC)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ck = torch.load(os.path.join('ckpt', opt.dataset, opt.run, 'best.pt'), map_location=device, weights_only=False)
a = Namespace(**ck['args'])
a.device = device
with open(os.path.join('..', 'datasets', 'data', a.dataset, 'dataset.pkl'), 'rb') as f:
    d = pickle.load(f)
loaders = {'val': Data_Val(d['train'], d['val'], a).get_pytorch_dataloaders(),
           'test': Data_Test(d['train'], d['val'], d['test'], a).get_pytorch_dataloaders()}
fm = create_model_FM(a)
if a.model == 'fmrec':
    model = Att_FM_model(fm, a)
else:
    tf = torch.from_numpy(np.load(a.text_emb)).float() if 'text' in a.modalities else None
    model = IDOM_model(fm, a, tf)
model = model.to(device)
model.load_state_dict(ck['model'])
print(f'{opt.run}: checkpoint from epoch {ck["epoch"]}, trained with sample_N={a.sample_N}')

res = {}
for n in opt.steps:
    model.fm.sample_N = n
    res[n] = {}
    for split, loader in loaders.items():
        torch.manual_seed(opt.seed)
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = evaluate(model, loader, a)
        if device == 'cuda':
            torch.cuda.synchronize()
        res[n][split] = {'time_s': round(time.perf_counter() - t0, 1), **r}

keys = list(res[opt.steps[0]]['test'].keys() - {'time_s'})
keys = ['fused'] + sorted(k for k in keys if k != 'fused')
print(f'\n{"N":>3} | {"val time":>8} {"test time":>9} | ' + ' | '.join(f'test {k} HR@10 / NDCG@10' for k in keys))
for n in opt.steps:
    v, t = res[n]['val'], res[n]['test']
    print(f'{n:>3} | {v["time_s"]:>7.1f}s {t["time_s"]:>8.1f}s | '
          + ' | '.join(f'{t[k]["HR@10"]:>10.2f} / {t[k]["NDCG@10"]:<10.2f}' for k in keys))

out = os.path.join(a.log_file, opt.dataset, f'{opt.run}_steps.json')
with open(out, 'w') as f:
    json.dump({'run': opt.run, 'ckpt_epoch': ck['epoch'], 'results': {str(n): r for n, r in res.items()}}, f, indent=2)
print('\nsaved', os.path.join('src', out))
