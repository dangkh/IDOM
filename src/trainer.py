import copy
import datetime
import json
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
    labels, scores = labels.detach().cpu(), scores.detach().cpu()
    metrics = {}
    for k, n, h in zip(ks, cal_ndcg(labels, scores, ks), cal_hr(labels, scores, ks)):
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


def model_train(tra_data_loader, val_data_loader, test_data_loader, model, args, logger):
    def log(msg):
        print(msg)
        logger.info(msg)

    model = model.to(args.device)
    optimizer = optimizers(model, args)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.gamma)
    ks = args.metric_ks
    best = {f'{m}@{k}': 0.0 for k in ks for m in ('HR', 'NDCG')}
    best_epoch = {key: -1 for key in best}
    best_model, bad_count = copy.deepcopy(model), 0
    history = []
    train_time = 0.0

    for epoch in range(args.epochs):
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
                log('best val: {} at {}'.format(best, best_epoch))
            else:
                bad_count += 1
            if bad_count >= args.patience:
                log('early stop at epoch {}'.format(epoch))
                break

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

    summary = {'args': {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, list, bool, type(None)))},
               'test': test_res, 'best_val': best, 'best_epoch': best_epoch,
               'train_time_s': train_time, 'test_time_s': test_time, 'history': history,
               'finished': datetime.datetime.now().isoformat()}
    return best_model, test_res, summary
