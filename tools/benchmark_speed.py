"""Speed benchmark: training and inference cost of FMRec / IDOM configs on the real data.

One process, one GPU, random weights (speed does not depend on the weights). For every config and
noise mode it measures
  * training time per batch (incl. data loading) -> estimated seconds per epoch
  * inference time on the validation set for each number of Euler steps (sample_N)
  * peak GPU memory
Train/eval times are measured on a subset of batches and extrapolated to the full set.

    python tools/benchmark_speed.py                                 # all 4 configs, GPU vs CPU noise
    python tools/benchmark_speed.py --configs fmrec idom_idtext --noise gpu --steps 1 5 10 30
Writes src/log/bench/bench_<host>_<time>.json
"""
import argparse
import itertools
import json
import os
import pickle
import socket
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, '..', 'src'))
sys.path.insert(0, SRC)

ap = argparse.ArgumentParser()
ap.add_argument('--configs', nargs='+', default=['fmrec', 'idom_id', 'idom_text', 'idom_idtext'])
ap.add_argument('--noise', nargs='+', default=['gpu', 'cpu'], help='gpu = new default, cpu = original FMRec')
ap.add_argument('--steps', nargs='+', type=int, default=[1, 5, 30], help='Euler steps (sample_N) to time at inference')
ap.add_argument('--dataset', default='amazon_beauty')
ap.add_argument('--encoder', default='all-MiniLM-L6-v2')
ap.add_argument('--train_batches', type=int, default=40)
ap.add_argument('--eval_batches', type=int, default=8, help='val batches timed per setting (extrapolated to all)')
ap.add_argument('--warmup', type=int, default=5)
opt = ap.parse_args()

import torch  # noqa: E402
from config import build_parser  # noqa: E402
from utils import Data_Train, Data_Val  # noqa: E402
from model import build_model  # noqa: E402
from trainer import evaluate, optimizers  # noqa: E402

os.chdir(SRC)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
text_emb = os.path.join('..', 'datasets', 'data', opt.dataset, f'text_emb_{opt.encoder}.npy')
CONFIGS = {
    'fmrec': ['--model', 'fmrec'],
    'idom_id': ['--model', 'idom', '--modalities', 'id'],
    'idom_text': ['--model', 'idom', '--modalities', 'text', '--text_emb', text_emb],
    'idom_idtext': ['--model', 'idom', '--modalities', 'id', 'text', '--text_emb', text_emb],
    'eh_b4': ['--model', 'eh', '--modalities', 'id', 'text', '--text_emb', text_emb, '--head_off'],
    'eh0': ['--model', 'eh', '--modalities', 'id', 'text', '--text_emb', text_emb],
    'eh0_nodec': ['--model', 'eh', '--modalities', 'id', 'text', '--text_emb', text_emb, '--Loss_Beta', '0'],
    'eh_c0': ['--model', 'eh', '--modalities', 'id', 'text', '--text_emb', text_emb, '--K_train', '2'],
    'eh_c4': ['--model', 'eh', '--modalities', 'id', 'text', '--text_emb', text_emb, '--K_train', '2', '--coupling'],
}

with open(os.path.join('..', 'datasets', 'data', opt.dataset, 'dataset.pkl'), 'rb') as f:
    d = pickle.load(f)
base = build_parser().parse_args(['--dataset', opt.dataset])
tra_loader = Data_Train(d['train'], base).get_pytorch_dataloaders()
val_loader = Data_Val(d['train'], d['val'], base).get_pytorch_dataloaders()
n_train, n_val = len(tra_loader), len(val_loader)
val_sub = list(itertools.islice(val_loader, opt.eval_batches))


def sync():
    if device == 'cuda':
        torch.cuda.synchronize()


gpu = torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'
print(f'host {socket.gethostname()} | {gpu} | torch {torch.__version__} | threads {torch.get_num_threads()} '
      f'| {n_train} train batches/epoch, {n_val} val batches')

rows = []
for cfg in opt.configs:
    for noise in opt.noise:
        if noise == 'cpu' and cfg.startswith('eh'):
            continue  # eh has no FMRec noise injection
        a = build_parser().parse_args(CONFIGS[cfg] + ['--dataset', opt.dataset] + (['--cpu_noise'] if noise == 'cpu' else []))
        a.device, a.item_num, a.user_num = device, len(d['smap']) + 1, len(d['train']) + 1
        torch.manual_seed(0)
        model = build_model(a).to(device)
        opt_ = optimizers(model, a)
        if device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # ---- training: time per batch including data loading ----
        model.train()
        it, times = iter(tra_loader), []
        for i in range(opt.warmup + opt.train_batches):
            sync()
            t0 = time.perf_counter()
            try:
                seq, tag = next(it)
            except StopIteration:
                it = iter(tra_loader)
                seq, tag = next(it)
            seq, tag = seq.to(device), tag.to(device)
            opt_.zero_grad()
            loss, _ = model.compute_loss(seq, tag)
            loss.backward()
            opt_.step()
            loss.item()
            sync()
            if i >= opt.warmup:
                times.append(time.perf_counter() - t0)
        s_batch = float(np.mean(times))
        row = {'config': cfg, 'noise': noise, 'train_s_per_batch': round(s_batch, 4),
               'train_s_per_epoch': round(s_batch * n_train, 1)}

        # ---- inference on validation, per number of Euler steps ----
        evaluate(model, val_sub[:1], a)  # warm-up
        for n in opt.steps:
            model.fm.sample_N = n
            sync()
            t0 = time.perf_counter()
            evaluate(model, val_sub, a)
            sync()
            row[f'val_s_N{n}'] = round((time.perf_counter() - t0) / len(val_sub) * n_val, 1)
        row['peak_mem_MB'] = round(torch.cuda.max_memory_allocated() / 2 ** 20) if device == 'cuda' else None
        rows.append(row)
        print(json.dumps(row), flush=True)
        del model, opt_

cols = ['config', 'noise', 'train_s_per_batch', 'train_s_per_epoch'] + [f'val_s_N{n}' for n in opt.steps] + ['peak_mem_MB']
w = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
print('\n' + '  '.join(c.ljust(w[c]) for c in cols))
for r in rows:
    print('  '.join(str(r[c]).ljust(w[c]) for c in cols))
print('\nval_s_N* = seconds for one full validation pass (22,363 users) with N Euler steps; '
      'the original runs used noise=cpu, N=30.')

out_dir = os.path.join('log', 'bench')
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, f'bench_{socket.gethostname()}_{time.strftime("%Y%m%d-%H%M%S")}.json')
with open(out, 'w') as f:
    json.dump({'host': socket.gethostname(), 'gpu': gpu, 'torch': torch.__version__,
               'threads': torch.get_num_threads(), 'rows': rows}, f, indent=2)
print('saved', os.path.join('src', out))
