"""Summarise runs from their .out logs: works for runs that are still running or were killed by walltime.

For every run (src/log/<dataset>/<run>.out) it takes the validation epoch that is best on --select
(fused score), and reports mean ± std over seeds of:
  * VAL metrics at that epoch (always available), per score (fused / id / text)
  * TEST metrics: from the final test of a finished run, otherwise from the "Test epoch E" line at the
    best val epoch (the new trainer tests every --test_interval=10 epochs); '-' if the run has neither.
VAL-only numbers are fine for comparing our configs with each other (same protocol for all),
but the paper must report TEST numbers.

    python tools/summarize_logs.py
    python tools/summarize_logs.py --select HR@10 --csv results_logs.csv
"""
import argparse
import ast
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
ap.add_argument('--select', default='NDCG@10', help='val metric (fused) used to pick the best epoch')
ap.add_argument('--metrics', nargs='+', default=['HR@10', 'NDCG@10', 'HR@20', 'NDCG@20'])
ap.add_argument('--csv', default=None)
args = ap.parse_args()

VAL = re.compile(r'^Val epoch (\d+) \([\d.]+s\): (\{.*\})\s*$')
TEST_AT = re.compile(r'^Test(?:@best)? epoch (\d+) \([\d.]+s\): (\{.*\})\s*$')  # periodic test lines
EPOCH = re.compile(r'^Epoch (\d+) avg loss')
FINAL = re.compile(r'^\s*(\w+): (\{.*\})\s*$')  # lines after "Test-----" in a finished run


def parse(path):
    vals, tests_at, final, last_epoch, in_test, stopped = {}, {}, {}, -1, False, None
    with open(path, errors='replace') as f:
        for line in f:
            if m := VAL.match(line):
                vals[int(m.group(1))] = json.loads(m.group(2))
            elif m := TEST_AT.match(line):
                tests_at[int(m.group(1))] = json.loads(m.group(2))
            elif m := EPOCH.match(line):
                last_epoch = int(m.group(1))
            elif line.startswith('early stop at epoch'):
                stopped = int(line.split()[-1])
            elif line.startswith('Test-----'):
                in_test, final = True, {}
            elif in_test and (m := FINAL.match(line)):
                final[m.group(1)] = ast.literal_eval(m.group(2))
            elif line.startswith('Best val:'):
                in_test = False
    return vals, tests_at, final, last_epoch, stopped


groups = defaultdict(list)
for path in sorted(glob.glob(os.path.join(args.log_dir, args.dataset, '*.out'))):
    run = os.path.basename(path)[:-4]
    vals, tests_at, final, last_epoch, stopped = parse(path)
    if not vals:
        continue
    best_ep = max(vals, key=lambda e: vals[e]['fused'][args.select])
    if final:
        test, test_src = final, 'final'
    else:
        cands = [e for e in tests_at if e <= best_ep]  # test logged at the best val epoch (or the closest before)
        test, test_src = (tests_at[max(cands)], f'@{max(cands)}') if cands else (None, '-')
    status = 'done' if final else (f'stopped@{stopped}' if stopped is not None else f'running/killed@{last_epoch}')
    config = re.sub(r'_seed\d+', '', run)
    seed = (re.search(r'_seed(\d+)', run) or [None, '?'])[1]
    groups[config].append(dict(seed=seed, best_ep=best_ep, last=last_epoch, status=status,
                               val=vals[best_ep], test=test, test_src=test_src))


def fmt(vs):
    vs = [v for v in vs if v is not None]
    if not vs:
        return '-'
    return f'{np.mean(vs):.2f} ± {np.std(vs):.2f}' if len(vs) > 1 else f'{vs[0]:.2f}'


rows = []
for config, runs in groups.items():
    for key in runs[0]['val']:  # fused, then per-modality scores
        for split in ('val', 'test'):
            row = {'config': config, 'score': key, 'split': split.upper(), 'n': len(runs)}
            for m in args.metrics:
                row[m] = fmt([(r[split] or {}).get(key, {}).get(m) for r in runs])
            if split == 'val':
                row['best_ep'] = ' '.join(str(r['best_ep']) for r in runs)
                row['status'] = ' '.join(f"s{r['seed']}:{r['status']}" for r in runs)
            else:
                row['best_ep'] = ' '.join(r['test_src'] for r in runs)
                row['status'] = ''
            rows.append(row)

cols = ['config', 'score', 'split'] + args.metrics + ['n', 'best_ep', 'status']
print(f'best epoch per run = max VAL fused {args.select}; mean ± std over seeds\n')
if rows:
    w = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print('  '.join(c.ljust(w[c]) for c in cols))
    prev = None
    for r in rows:
        if prev and r['config'] != prev:
            print()
        print('  '.join(str(r[c]).ljust(w[c]) for c in cols))
        prev = r['config']
if args.csv and rows:
    import csv
    with open(args.csv, 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)
    print('\nsaved', args.csv)
