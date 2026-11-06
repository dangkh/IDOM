"""Aggregate tools/eval_ckpt.py outputs (src/log/<dataset>/*_steps.json): accuracy and time vs Euler steps,
mean ± std over seeds, plus the change relative to the largest N (the training default, 30).

    python tools/summarize_steps.py
    python tools/summarize_steps.py --split val --metrics HR@10 NDCG@10
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--log_dir', default=os.path.join(HERE, '..', 'src', 'log'))
ap.add_argument('--dataset', default='amazon_beauty')
ap.add_argument('--split', default='test', choices=['val', 'test'])
ap.add_argument('--metrics', nargs='+', default=['HR@10', 'NDCG@10', 'HR@20', 'NDCG@20'])
args = ap.parse_args()

groups = defaultdict(list)
for f in sorted(glob.glob(os.path.join(args.log_dir, args.dataset, '*_steps.json'))):
    with open(f) as fh:
        r = json.load(fh)
    groups[re.sub(r'_seed\d+', '', r['run'])].append(r)

for config, runs in groups.items():
    steps = sorted({int(n) for r in runs for n in r['results']})
    ref = steps[-1]
    keys = [k for k in runs[0]['results'][str(ref)][args.split] if k != 'time_s']
    keys = ['fused'] + sorted(k for k in keys if k != 'fused')
    print(f'\n=== {config}  ({len(runs)} seeds, {args.split}; Δ = change vs N={ref}, in %-points)')
    for key in keys:
        print(f'  -- score: {key}')
        print('  ' + f'{"N":>4} {"time_s":>7}  ' + '  '.join(f'{m:>20}' for m in args.metrics))
        for n in steps:
            rs = [r['results'][str(n)][args.split] for r in runs if str(n) in r['results']]
            t = np.mean([x['time_s'] for x in rs])
            cells = []
            for m in args.metrics:
                v = np.array([x[key][m] for x in rs])
                vref = np.array([r['results'][str(ref)][args.split][key][m] for r in runs if str(n) in r['results']])
                cells.append(f'{v.mean():6.2f}±{v.std():.2f} ({(v - vref).mean():+.2f})')
            print('  ' + f'{n:>4} {t:>7.1f}  ' + '  '.join(f'{c:>20}' for c in cells))
