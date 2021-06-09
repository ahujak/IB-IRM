import argparse
import os
import sys


def add_path(path):
    if path not in sys.path:
        print('Adding {}'.format(path))
        sys.path.append(path)


parser = argparse.ArgumentParser(description='Colored MNIST')
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--l2_regularizer_weight', type=float, default=0.001)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--n_restarts', '--nr', type=int, default=1)
parser.add_argument('--penalty_anneal_iters', '--p_step', type=int, default=100)
parser.add_argument('--penalty_weight', '--p', type=float, default=10000.0)
parser.add_argument('--steps', type=int, default=501)
parser.add_argument('--grayscale_model', action='store_true')
parser.add_argument('--d', type=int, default=0)

parser.add_argument('--method', type=str, choices=['irm', 'erm', 'vrex'], default='irm')
parser.add_argument('--prob_flip', default=0.25, type=float)
parser.add_argument('--nn', choices=['cnn', 'mlp'], default='mlp')
parser.add_argument('--val', default=0, type=float, help="the ratio of valid dataset in test domain")

parser.add_argument('--ib_lambda', default=0, type=float)
parser.add_argument('--ib_step', default=0, type=int)
parser.add_argument('--class_condition', '--cc', action='store_true')
parser.add_argument('--inter', default=0, type=int, help='use which layer to compute variance')
parser.add_argument("--test_valid", type=int, default=0, choices=[0, 1])
args = parser.parse_args()