"""Convergence speed from finished runs (src/log/<dataset>/*.json, 'history' = validation every eval_interval).

For each config (seeds averaged) it prints
  (1) the validation curve of the chosen score at fixed epochs, and
  (2) epochs and approx. train hours needed to FIRST reach a target = frac x reference best val
      (reference = best val of --ref config, mean over its seeds).
Answers: does ID+text reach FMRec-level accuracy in fewer epochs / less time, even if it trains longer overall?

    python tools/convergence.py
    python tools/convergence.py --metric NDCG@10 --ref fmrec[id]_v3 --fracs 0.9 0.95 1.0
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--log_dir', default=os.path.join(HERE, '..', 'src', 'log'))
ap.add_argument('--dataset', default='amazon_beauty')
ap.add_argument('--metric', default='HR@10')
ap.add_argument('--score', default='fused', help='fused | id | text')
ap.add_argument('--ref', default='fmrec[id]_v3', help='config name used as the accuracy reference')
ap.add_argument('--fracs', nargs='+', type=float, default=[0.9, 0.95, 1.0])
ap.add_argument('--epochs', nargs='+', type=int, default=[20, 40, 60, 80, 100, 140, 200, 300])
args = ap.parse_args()

groups = defaultdict(list)
for f in sorted(glob.glob(os.path.join(args.log_dir, args.dataset, '*.json'))):
    if f.endswith(('_fusesweep.json', '_steps.json')) or os.path.basename(f).startswith('ensemble_'):
        continue
    with open(f) as fh:
        r = json.load(fh)
    if 'history' not in r or not r['history']:
        continue
    a = r['args']
    mods = '+'.join(a['modalities']) if a['model'] in ('idom', 'eh') else 'id'
    name = f"{a['model']}[{mods}]" + (f"_{a['tag']}" if a.get('tag') else '')
    curve = {h['epoch']: h['val'].get(args.score, h['val'].get('fused'))[args.metric] for h in r['history']}
    last = max(curve)
    groups[name].append({'curve': curve, 's_per_epoch': r['train_time_s'] / max(1, last), 'seed': a['random_seed']})

if args.ref not in groups:
    raise SystemExit(f'reference {args.ref} not found; configs: {sorted(groups)}')
ref_best = float(np.mean([max(g['curve'].values()) for g in groups[args.ref]]))
print(f'validation {args.metric} ({args.score}); reference = best of {args.ref} = {ref_best:.2f}\n')

w = max(len(n) for n in groups) + 2
print('(1) validation curve, mean over seeds')
print(f'{"config":<{w}}' + ''.join(f'{"ep" + str(e):>8}' for e in args.epochs) + f'{"best":>8}')
for name, runs in sorted(groups.items()):
    row = []
    for e in args.epochs:
        v = [g['curve'][e] for g in runs if e in g['curve']]
        row.append(f'{np.mean(v):8.2f}' if v else f'{"-":>8}')
    print(f'{name:<{w}}' + ''.join(row) + f'{np.mean([max(g["curve"].values()) for g in runs]):8.2f}')

print(f'\n(2) first epoch (and approx. train hours) reaching frac x {ref_best:.2f}; "-" = never (n = seeds)')
print(f'{"config":<{w}}' + ''.join(f'{f"{fr:.0%}":>18}' for fr in args.fracs) + f'{"s/epoch":>10}')
for name, runs in sorted(groups.items()):
    cells = []
    for fr in args.fracs:
        tgt = fr * ref_best
        eps = [min([e for e, v in g['curve'].items() if v >= tgt], default=None) for g in runs]
        ok = [(e, e * g['s_per_epoch'] / 3600) for e, g in zip(eps, runs) if e is not None]
        cells.append(f'{np.mean([o[0] for o in ok]):6.0f} ep {np.mean([o[1] for o in ok]):4.1f}h n{len(ok)}'
                     if ok else f'{"-":>18}')
    print(f'{name:<{w}}' + ''.join(f'{c:>18}' for c in cells) + f'{np.mean([g["s_per_epoch"] for g in runs]):10.1f}')
print('\nNote: hours use each run\'s measured s/epoch (train only, nodes differ between runs).')
