"""Build item text embeddings for FMRec's amazon_beauty dataset.

FMRec's amazon_beauty/dataset.pkl is exactly S3Rec's Beauty.txt (verified: 22,363/22,363 user
sequences identical), whose item ids were assigned by S3Rec's data_process.py from the raw
Amazon 2014 file reviews_Beauty_5.json.gz. This script re-runs that id assignment to recover
id -> ASIN, CHECKS that every user sequence matches dataset.pkl exactly, then builds a text for each
item from meta_Beauty.json.gz and encodes it with a sentence-transformers model.

Raw files (Amazon review data 2014, McAuley et al.), download once:
  reviews_Beauty_5.json.gz  (5-core reviews)
  meta_Beauty.json.gz       (product metadata)
  from https://jmcauley.ucsd.edu/data/amazon/links.html  ("Beauty" row)

Usage:
  python scripts/prepare_beauty_text.py --reviews path/reviews_Beauty_5.json.gz \
      --meta path/meta_Beauty.json.gz --out_dir datasets/data/amazon_beauty
Outputs (in out_dir):
  item_asin.json            id -> asin
  item_text.json            id -> text actually encoded
  text_emb_<model>.npy      float32 array (item_num+1, d), row 0 = padding zeros, rows L2-normalised
"""
import argparse
import ast
import gzip
import json
import os
import pickle
import re
from collections import defaultdict


def parse(path):
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as g:
        for line in g:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:  # meta files of the 2014 dump are python dict literals
                yield ast.literal_eval(line)


# ---------------- replicated from S3Rec data/data_process.py (logic kept verbatim) -------------
def amazon_interactions(review_file, rating_score=0.0):
    datas = []
    for inter in parse(review_file):
        if float(inter['overall']) <= rating_score:
            continue
        datas.append((inter['reviewerID'], inter['asin'], int(inter['unixReviewTime'])))
    return datas


def get_interaction(datas):
    user_seq = {}
    for user, item, t in datas:
        user_seq.setdefault(user, []).append((item, t))
    for user, item_time in user_seq.items():
        item_time.sort(key=lambda x: x[1])  # stable sort, as in S3Rec
        user_seq[user] = [x[0] for x in item_time]
    return user_seq


def check_Kcore(user_items, user_core, item_core):
    user_count, item_count = defaultdict(int), defaultdict(int)
    for user, items in user_items.items():
        for item in items:
            user_count[user] += 1
            item_count[item] += 1
    for _, num in user_count.items():
        if num < user_core:
            return user_count, item_count, False
    for _, num in item_count.items():
        if num < item_core:
            return user_count, item_count, False
    return user_count, item_count, True


def filter_Kcore(user_items, user_core, item_core):
    user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    while not isKcore:
        for user, num in user_count.items():
            if user_count[user] < user_core:
                user_items.pop(user)
            else:
                for item in user_items[user]:  # (S3Rec removes while iterating; kept for exactness)
                    if item_count[item] < item_core:
                        user_items[user].remove(item)
        user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    return user_items


def id_map(user_items):
    user2id, item2id, final = {}, {}, {}
    for user, items in user_items.items():
        if user not in user2id:
            user2id[user] = len(user2id) + 1
        iids = []
        for item in items:
            if item not in item2id:
                item2id[item] = len(item2id) + 1
            iids.append(item2id[item])
        final[user2id[user]] = iids
    return final, user2id, item2id
# -----------------------------------------------------------------------------------------------


def verify(final, pkl_path):
    with open(pkl_path, 'rb') as f:
        d = pickle.load(f)
    umap = d.get('umap', {u + 1: u for u in d['train']})  # raw S3Rec user id -> FMRec user index
    n_ok, bad = 0, []
    for raw_u, u in umap.items():
        seq = d['train'][u] + d['val'][u] + d['test'][u]
        if final.get(raw_u) == seq:
            n_ok += 1
        elif len(bad) < 5:
            bad.append(raw_u)
    n_items = len(d['smap'])
    return n_ok, len(umap), bad, n_items


def clean(s, max_words=None):
    s = re.sub(r'<[^>]+>', ' ', str(s))
    s = re.sub(r'&amp;', '&', s)
    s = re.sub(r'\s+', ' ', s).strip()
    if max_words:
        s = ' '.join(s.split()[:max_words])
    return s


def item_text(info, desc_words):
    parts = []
    if info.get('title'):
        parts.append('Title: ' + clean(info['title']))
    if info.get('brand'):
        parts.append('Brand: ' + clean(info['brand']))
    cats = []
    for path in info.get('categories', []) or []:
        p = [clean(c) for c in path[1:] if c]  # drop the root "Beauty", as S3Rec does for attributes
        if p:
            cats.append(' > '.join(p))
    if cats:
        parts.append('Categories: ' + ' | '.join(dict.fromkeys(cats)))
    if desc_words > 0 and info.get('description'):
        parts.append('Description: ' + clean(info['description'], desc_words))
    return '. '.join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reviews', required=True, help='reviews_Beauty_5.json.gz')
    ap.add_argument('--meta', required=True, help='meta_Beauty.json.gz')
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument('--out_dir', default=os.path.join(here, '..', 'datasets', 'data', 'amazon_beauty'))
    ap.add_argument('--encoder', default='sentence-transformers/all-MiniLM-L6-v2',
                    help='any sentence-transformers model, e.g. sentence-transformers/all-mpnet-base-v2')
    ap.add_argument('--desc_words', type=int, default=64, help='words of description to keep (0 = none)')
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--skip_encode', action='store_true', help='only rebuild/verify mapping and texts')
    args = ap.parse_args()
    pkl = os.path.join(args.out_dir, 'dataset.pkl')

    print('1/4 rebuilding S3Rec id mapping from', args.reviews)
    user_items = get_interaction(amazon_interactions(args.reviews))
    user_items = filter_Kcore(user_items, 5, 5)
    final, user2id, item2id = id_map(user_items)
    print(f'    users={len(user2id)} items={len(item2id)}')

    print('2/4 verifying against', pkl)
    n_ok, n_users, bad, n_items = verify(final, pkl)
    print(f'    identical user sequences: {n_ok}/{n_users}; items {len(item2id)} vs {n_items}')
    if n_ok != n_users or len(item2id) != n_items:
        raise SystemExit(f'MAPPING MISMATCH (first bad users: {bad}). Wrong raw file? Expected the 2014 '
                         f'reviews_Beauty_5.json.gz. Nothing written.')
    id2asin = {i: a for a, i in item2id.items()}

    print('3/4 reading metadata', args.meta)
    asins = set(item2id)
    meta = {}
    for info in parse(args.meta):
        if info.get('asin') in asins:
            meta[info['asin']] = info
    missing = [a for a in asins if a not in meta]
    print(f'    meta found for {len(meta)}/{len(asins)} items')
    texts = {}
    for i in range(1, n_items + 1):
        t = item_text(meta.get(id2asin[i], {}), args.desc_words)
        texts[i] = t if t else 'Unknown beauty product'
    empty = sum(1 for t in texts.values() if t == 'Unknown beauty product')
    print(f'    items without usable text: {empty} (missing meta: {len(missing)})')
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'item_asin.json'), 'w') as f:
        json.dump(id2asin, f)
    with open(os.path.join(args.out_dir, 'item_text.json'), 'w', encoding='utf-8') as f:
        json.dump(texts, f, ensure_ascii=False, indent=0)
    print('    example:', texts[1][:200])

    if args.skip_encode:
        return
    print('4/4 encoding with', args.encoder)
    import numpy as np
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.encoder)
    emb = model.encode([texts[i] for i in range(1, n_items + 1)], batch_size=args.batch_size,
                       normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)
    full = np.zeros((n_items + 1, emb.shape[1]), dtype=np.float32)
    full[1:] = emb
    name = args.encoder.split('/')[-1]
    out = os.path.join(args.out_dir, f'text_emb_{name}.npy')
    np.save(out, full)
    print(f'    saved {out} shape={full.shape}')


if __name__ == '__main__':
    main()
