"""Aggregate result JSONs written by src/main.py into a mean ± std table over seeds.

    python tools/summarize.py                       # src/log/amazon_beauty
    python tools/summarize.py --dataset amazon_beauty --metrics HR@10 NDCG@10 HR@20 NDCG@20 --csv results.csv
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

ap = argparse.ArgumentParser()
here = os.path.dirname(os.path.abspath(__file__))
ap.add_argument('--log_dir', default=os.path.join(here, '..', 'src', 'log'))
ap.add_argument('--dataset', default='amazon_beauty')
ap.add_argument('--metrics', nargs='+', default=['HR@10', 'NDCG@10', 'HR@20', 'NDCG@20'])
ap.add_argument('--csv', default=None)
args = ap.parse_args()

files = sorted(glob.glob(os.path.join(args.log_dir, args.dataset, '*.json')))
groups = defaultdict(list)
for f in files:
    with open(f) as fh:
        r = json.load(fh)
    a = r['args']
    mods = '+'.join(a['modalities']) if a['model'] == 'idom' else 'id'
    name = f"{a['model']}[{mods}]" + (f" {a['tag']}" if a.get('tag') else '')
    if a.get('epochs', 500) != 500:
        name += f" ep{a['epochs']}"
    groups[name].append(r)

rows = []
for name, runs in groups.items():
    for key in runs[0]['test']:            # 'fused' and, for IDOM, each modality
        row = {'config': name, 'score': key, 'n_seeds': len(runs),
               'seeds': ' '.join(str(r['args']['random_seed']) for r in runs)}
        for m in args.metrics:
            v = [r['test'][key][m] for r in runs if key in r['test']]
            row[m] = f'{np.mean(v):.2f} ± {np.std(v):.2f}' if len(v) > 1 else f'{v[0]:.2f}'
        row['train_h'] = f"{np.mean([r['train_time_s'] for r in runs]) / 3600:.2f}"
        row['test_s'] = f"{np.mean([r['test_time_s'] for r in runs]):.1f}"
        rows.append(row)

cols = ['config', 'score'] + args.metrics + ['n_seeds', 'train_h', 'test_s', 'seeds']
width = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols} if rows else {}
print(f'{len(files)} result files in {os.path.join(args.log_dir, args.dataset)}\n')
if rows:
    print('  '.join(c.ljust(width[c]) for c in cols))
    for r in rows:
        print('  '.join(str(r[c]).ljust(width[c]) for c in cols))
if args.csv and rows:
    import csv
    with open(args.csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print('\nsaved', args.csv)
