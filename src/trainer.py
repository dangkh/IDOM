import copy
import datetime
import json
import os
import random
import time

import numpy as np
import torch
import torch.optim as optim


def optimizers(model, args):
    if args.optimizer.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    raise ValueError(args.optimizer)


# ---- metrics: identical to FMRec (per-batch mean, then mean over batches) ----
def cal_hr(label, predict, ks):
    _, topk_predict = torch.topk(predict, k=max(ks), dim=-1)
    hit = label == topk_predict
    return [hit[:, :k].sum().item() / label.size()[0] for k in ks]


def cal_hr_topk(label, topk_predict, ks):
    hit = label == topk_predict
    return [hit[:, :k].sum().item() / label.size()[0] for k in ks]


def cal_ndcg_topk(label, topk_predict, ks):
    hit = (label == topk_predict).int()
    out = []
    for k in ks:
        max_dcg = dcg(torch.tensor([1] + [0] * (k - 1)))
        out.append((dcg(hit[:, :k]) / max_dcg).mean().item())
    return out


def dcg(hit):
    log2 = torch.log2(torch.arange(1, hit.size()[-1] + 1) + 1).unsqueeze(0)
    return (hit / log2).sum(dim=-1)


def cal_ndcg(label, predict, ks):
    _, topk_predict = torch.topk(predict, k=max(ks), dim=-1)
    hit = (label == topk_predict).int()
    out = []
    for k in ks:
        max_dcg = dcg(torch.tensor([1] + [0] * (k - 1)))
        out.append((dcg(hit[:, :k]) / max_dcg).mean().item())
    return out


def hrs_and_ndcgs_k(scores, labels, ks):
    # top-k on the device the scores live on (GPU); then only the B x max(k) indices go to the CPU.
    # Same numbers as FMRec's version (which moved the full B x N score matrix to the CPU first).
    _, topk = torch.topk(scores.detach(), k=max(ks), dim=-1)
    labels, topk = labels.detach().cpu(), topk.cpu()
    metrics = {}
    for k, n, h in zip(ks, cal_ndcg_topk(labels, topk, ks), cal_hr_topk(labels, topk, ks)):
        metrics[f'HR@{k}'] = h
        metrics[f'NDCG@{k}'] = n
    return metrics


@torch.no_grad()
def evaluate(model, loader, args):
    """Returns {score_key: {metric: value(%)}}; score keys are 'fused' and, for IDOM, each modality."""
    model.eval()
    acc = {}
    for batch in loader:
        seq, labels = [x.to(args.device) for x in batch]
        for key, scores in model.predict_scores(seq).items():
            for mk, v in hrs_and_ndcgs_k(scores, labels, args.metric_ks).items():
                acc.setdefault(key, {}).setdefault(mk, []).append(v)
    return {key: {mk: round(float(np.mean(v)) * 100, 4) for mk, v in d.items()} for key, d in acc.items()}


def _rng_state():
    st = {'torch': torch.get_rng_state(), 'numpy': np.random.get_state(), 'python': random.getstate()}
    if torch.cuda.is_available():
        st['cuda'] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st):
    torch.set_rng_state(st['torch'])
    np.random.set_state(st['numpy'])
    random.setstate(st['python'])
    if 'cuda' in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st['cuda'])


def _atomic_save(obj, path):
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def model_train(tra_data_loader, val_data_loader, test_data_loader, model, args, logger):
    def log(msg):
        print(msg, flush=True)
        logger.info(msg)

    model = model.to(args.device)
    optimizer = optimizers(model, args)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.gamma)
    ks = args.metric_ks
    best = {f'{m}@{k}': 0.0 for k in ks for m in ('HR', 'NDCG')}
    best_epoch = {key: -1 for key in best}
    bad_count, history, train_time, start_epoch = 0, [], 0.0, 0
    best_model = copy.deepcopy(model)

    # ---- checkpoint / resume ----------------------------------------------------------------
    ckpt_dir = args.ckpt_dir
    last_path = os.path.join(ckpt_dir, 'last.pt') if ckpt_dir else None
    best_path = os.path.join(ckpt_dir, 'best.pt') if ckpt_dir else None
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
    if ckpt_dir and args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=args.device, weights_only=False)
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        lr_scheduler.load_state_dict(ck['scheduler'])
        best, best_epoch, bad_count = ck['best'], ck['best_epoch'], ck['bad_count']
        history, train_time, start_epoch = ck['history'], ck['train_time'], ck['epoch'] + 1
        if os.path.exists(best_path):
            best_model.load_state_dict(torch.load(best_path, map_location=args.device, weights_only=False)['model'])
        _set_rng_state(ck['rng'])
        log('RESUMED from {} at epoch {} (best so far {} at {}, bad_count {})'.format(
            last_path, start_epoch, best, best_epoch, bad_count))

    def save_last(epoch):
        if not ckpt_dir:
            return
        _atomic_save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                      'scheduler': lr_scheduler.state_dict(), 'epoch': epoch, 'best': best,
                      'best_epoch': best_epoch, 'bad_count': bad_count, 'history': history,
                      'train_time': train_time, 'rng': _rng_state()}, last_path)

    stopped = start_epoch >= args.epochs or bad_count >= args.patience
    for epoch in range(start_epoch, args.epochs):
        if stopped:
            break
        model.train()
        t0 = time.time()
        losses, n_batches = [], len(tra_data_loader)
        for i, batch in enumerate(tra_data_loader):
            seq, tag = [x.to(args.device) for x in batch]
            optimizer.zero_grad()
            loss, parts = model.compute_loss(seq, tag)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if i % int(n_batches / 5 + 1) == 0:
                log('[%d/%d] Loss: %.4f %s' % (i, n_batches, loss.item(),
                                               ' '.join(f'{k}={v:.4f}' for k, v in parts.items())))
        train_time += time.time() - t0
        lr_scheduler.step()
        log('Epoch {} avg loss {:.4f} ({:.1f}s)'.format(epoch, float(np.mean(losses)), time.time() - t0))

        if epoch != 0 and epoch % args.eval_interval == 0:
            t1 = time.time()
            res = evaluate(model, val_data_loader, args)
            log('Val epoch {} ({:.1f}s): {}'.format(epoch, time.time() - t1, json.dumps(res)))
            history.append({'epoch': epoch, 'val': res})
            improved = False
            for key, v in res['fused'].items():  # model selection on the fused score, as in FMRec
                if v > best[key]:
                    improved, best[key], best_epoch[key] = True, v, epoch
            if improved:
                bad_count = 0
                best_model = copy.deepcopy(model)
                if ckpt_dir:
                    _atomic_save({'model': best_model.state_dict(), 'epoch': epoch, 'best': best,
                                  'args': {k: v for k, v in vars(args).items()
                                           if isinstance(v, (int, float, str, list, bool, type(None)))}},
                                 best_path)
                log('best val: {} at {}'.format(best, best_epoch))
            else:
                bad_count += 1
            if bad_count >= args.patience:
                log('early stop at epoch {}'.format(epoch))
                stopped = True
            save_last(epoch)
        elif args.ckpt_every > 0 and epoch % args.ckpt_every == 0:
            save_last(epoch)

        # periodic TEST of the current model (logged only, never used for model selection)
        if args.test_interval > 0 and epoch != 0 and epoch % args.test_interval == 0:
            t3 = time.time()
            tres = evaluate(model, test_data_loader, args)
            log('Test epoch {} ({:.1f}s): {}'.format(epoch, time.time() - t3, json.dumps(tres)))

    if args.eval_interval > args.epochs:
        best_model = copy.deepcopy(model)

    t2 = time.time()
    test_res = evaluate(best_model, test_data_loader, args)
    test_time = time.time() - t2
    log('Test------------------------------------------------------')
    for key, v in test_res.items():
        log('{:>6s}: {}'.format(key, v))
    log('Best val: {} at {}'.format(best, best_epoch))
    log('train time {:.1f}s total, test inference {:.1f}s'.format(train_time, test_time))

    if args.eval_train:
        log('Train metrics: {}'.format(evaluate(best_model, tra_data_loader, args)))

    if ckpt_dir and os.path.exists(last_path):
        os.remove(last_path)  # run finished: keep only best.pt (used by tools/fuse_sweep.py)

    summary = {'args': {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, list, bool, type(None)))},
               'test': test_res, 'best_val': best, 'best_epoch': best_epoch,
               'train_time_s': train_time, 'test_time_s': test_time, 'history': history,
               'finished': datetime.datetime.now().isoformat()}
    return best_model, test_res, summary
