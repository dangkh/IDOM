import argparse
import json
import logging
import os
import pickle
import random
import time

import numpy as np

parser = argparse.ArgumentParser()
# ---- model selection (new) ----
parser.add_argument('--model', default='fmrec', choices=['fmrec', 'idom'],
                    help='fmrec = original ID-only FMRec; idom = parallel per-modality flows (no intent anchor yet)')
parser.add_argument('--modalities', nargs='+', default=['id', 'text'], help="IDOM modalities, e.g. id text | text | id")
parser.add_argument('--text_emb', default=None, help='.npy item text embeddings, shape (item_num+1, d), row 0 = padding')
parser.add_argument('--fuse_weights', nargs='+', type=float, default=None,
                    help='per-modality weights for log-softmax score fusion at inference (default: all 1)')
parser.add_argument('--modality_emb_std', type=float, default=0.02, help='init std of modality embeddings')
parser.add_argument('--no_modality_emb', action='store_true', help='ablation: do not tell the shared field the modality')
parser.add_argument('--gpu_id', default=None, help='sets CUDA_VISIBLE_DEVICES if given (FMRec hard-coded "7")')
parser.add_argument('--eval_train', action='store_true', help='also report metrics on the training set (slow)')
parser.add_argument('--tag', default='', help='free-text tag appended to the result file name')
# ---- original FMRec arguments (defaults unchanged) ----
parser.add_argument('--dataset', default='amazon_beauty', help='Dataset name: amazon_beauty, steam, ml-100k, yelp')
parser.add_argument('--log_file', default='log/', help='log dir path')
parser.add_argument('--random_seed', type=int, default=1997, help='Random seed')
parser.add_argument('--max_len', type=int, default=50, help='The max length of sequence')
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
parser.add_argument('--num_gpu', type=int, default=1, help='Number of GPU')
parser.add_argument('--batch_size', type=int, default=512, help='Batch Size')
parser.add_argument("--hidden_size", default=128, type=int, help="hidden size of model")
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout of representation')
parser.add_argument('--emb_dropout', type=float, default=0.3, help='Dropout of item embedding')
parser.add_argument("--hidden_act", default="gelu", type=str)
parser.add_argument('--num_blocks', type=int, default=4, help='Number of Transformer blocks')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs for training')
parser.add_argument('--decay_step', type=int, default=100, help='Decay step for StepLR')
parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for StepLR')
parser.add_argument('--metric_ks', nargs='+', type=int, default=[5, 10, 20], help='ks for Metric@k')
parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'])
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='L2 regularization')
parser.add_argument('--momentum', type=float, default=None, help='SGD momentum')
parser.add_argument('--lambda_uncertainty', type=float, default=0.001, help='uncertainty weight')
parser.add_argument('--eval_interval', type=int, default=20, help='the number of epoch to eval')
parser.add_argument('--patience', type=int, default=5, help='the number of epoch to wait before early stop')
parser.add_argument('--eps', type=float, default=0.001, help='start step')
parser.add_argument('--sample_N', type=int, default=30, help='Euler calculate steps')
parser.add_argument('--lambda_t', type=float, default=0.001, help='scale of condition')
parser.add_argument('--dropout_c', type=float, default=0.1, help='scale of condition')
parser.add_argument('--eps_reverse', type=float, default=0.001, help='reverse start step')
parser.add_argument('--m_logNorm', type=float, default=1.0, help='Logit-Normal Sampling mean')
parser.add_argument('--s_logNorm', type=float, default=0.6, help='Logit-Normal Sampling Variance')
parser.add_argument('--s_modsamp', type=float, default=1.0, help='Mode_sample_timestep scale parameter')
parser.add_argument('--last', type=int, default=2, help='last H Get')
parser.add_argument('--mask_ratio', type=float, default=1.0, help='Balanced positive-negative class ratio')
parser.add_argument('--sampling_method', type=str, default='mode', choices=['uniform', 'logit_normal', 'mode', 'cosmap'])
parser.add_argument('--Loss_Alpha', type=float, default=0.2, help='Loss parameter')
parser.add_argument('--Loss_Beta', type=float, default=0.4, help='Loss parameter')
args = parser.parse_args()

if args.gpu_id is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

import torch  # noqa: E402  (import after CUDA_VISIBLE_DEVICES is set)
import torch.backends.cudnn as cudnn  # noqa: E402
from utils import Data_Train, Data_Val, Data_Test  # noqa: E402
from model import create_model_FM, Att_FM_model, IDOM_model  # noqa: E402
from trainer import model_train  # noqa: E402

if args.device == 'cuda' and not torch.cuda.is_available():
    print('CUDA not available -> falling back to CPU')
    args.device = 'cpu'

HERE = os.path.dirname(os.path.abspath(__file__))
run_name = '{}_{}_{}_seed{}{}'.format(
    args.dataset, args.model, '-'.join(args.modalities) if args.model == 'idom' else 'id',
    args.random_seed, ('_' + args.tag) if args.tag else '')
log_dir = os.path.join(args.log_file, args.dataset)
os.makedirs(log_dir, exist_ok=True)
stamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
logging.basicConfig(level=logging.INFO, filename=os.path.join(log_dir, f'{run_name}_{stamp}.log'),
                    datefmt='%Y/%m/%d %H:%M:%S', format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')
logger = logging.getLogger(__name__)
print(args)
logger.info(args)


def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def main(args, logger):
    fix_random_seed_as(args.random_seed)
    path_data = os.path.join(HERE, '..', 'datasets', 'data', args.dataset, 'dataset.pkl')
    with open(path_data, 'rb') as f:
        data_raw = pickle.load(f)

    args.item_num = len(data_raw['smap']) + 1
    args.user_num = len(data_raw['train']) + 1

    tra_loader = Data_Train(data_raw['train'], args).get_pytorch_dataloaders()
    val_loader = Data_Val(data_raw['train'], data_raw['val'], args).get_pytorch_dataloaders()
    test_loader = Data_Test(data_raw['train'], data_raw['val'], data_raw['test'], args).get_pytorch_dataloaders()

    fm = create_model_FM(args)
    if args.model == 'fmrec':
        model = Att_FM_model(fm, args)
    else:
        text_feat = None
        if 'text' in args.modalities:
            assert args.text_emb, '--text_emb is required for the text modality (run scripts/prepare_beauty_text.py)'
            text_feat = torch.from_numpy(np.load(args.text_emb)).float()
        model = IDOM_model(fm, args, text_feat)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'model={args.model} trainable params={n_params:,}')
    logger.info(f'model={args.model} trainable params={n_params:,}')

    _, test_res, summary = model_train(tra_loader, val_loader, test_loader, model, args, logger)
    summary['n_params'] = n_params
    out = os.path.join(log_dir, f'{run_name}_{stamp}.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print('results saved to', out)


if __name__ == '__main__':
    main(args, logger)
