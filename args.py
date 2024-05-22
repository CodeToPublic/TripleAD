import argparse
import torch
import time

# python main.py --dataset acm --lambd 0.8
# python main.py --dataset citeseer --lambd 0.4

# Set argument
parser = argparse.ArgumentParser(description='TripleAD')
parser.add_argument('--name', type=str, default="testrun", help='Provide a test name.')

parser.add_argument('--dataset', type=str, default='acm')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--embedding_dim', type=int, default=128)
parser.add_argument('--repeats', type=int, default=100)
parser.add_argument('--lambd', type=float, default=0.8)
parser.add_argument('--learning_rate', type=float, default=1e-0)
parser.add_argument('--attMask', type=float, default=0.1)
parser.add_argument('--strMask', type=float, default=0.1)
parser.add_argument('--weight_decay_att', type=float, default=0)
parser.add_argument('--weight_decay_str', type=float, default=0)

parser.add_argument('--eta1', type=float, default=0.7)
parser.add_argument('--eta2', type=float, default=0.6)
parser.add_argument('--margin', type=int, default=7)

parser.add_argument('--multi_scale_L', type=int ,default=5)
parser.add_argument('--restart_prob_a', type=float, default=0.05)
parser.add_argument('--att_epoch', type=int, default=400)

parser.add_argument('--k', type=int, default=4)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--restart_prob_b', type=float, default=0.05)
parser.add_argument('--propagation_iteration_T', type=int, default=5)
parser.add_argument('--str_epoch', type=int, default=400)


parser.add_argument('--detla', type=float, default=0.5)

parser.add_argument('--no_cuda', action='store_false', default=True, help='Disables CUDA training.')

args = parser.parse_args()
args.device = torch.device('cuda:0' if args.no_cuda and torch.cuda.is_available() else 'cpu')
args.name = args.name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')