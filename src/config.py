"""All command-line options (shared by main.py and the tools)."""
import argparse


def build_parser():
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--test_interval', type=int, default=10,
                        help='evaluate (and log) the TEST set every N epochs, so a run killed by walltime still has '
                             'test numbers; 0 = only at the end')
    parser.add_argument('--resume', action='store_true', help='continue from ckpt/<run>/last.pt if it exists')
    parser.add_argument('--no_ckpt', action='store_true', help='do not write checkpoints')
    parser.add_argument('--ckpt_every', type=int, default=0,
                        help='also save last.pt every N epochs (0 = only at each validation, together with best.pt)')
    parser.add_argument('--cpu_noise', action='store_true',
                        help='original FMRec: draw the uncertainty noise on CPU (slow). Default: same distribution on GPU')
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
    return parser
