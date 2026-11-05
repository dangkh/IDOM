import json
import logging
import os
import pickle
import random
import time

import numpy as np

from config import build_parser

args = build_parser().parse_args()

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
args.ckpt_dir = None if args.no_ckpt else os.path.join('ckpt', args.dataset, run_name)
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
