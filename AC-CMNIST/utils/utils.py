import numpy as np
import math
import os
from typing import Tuple, List, Dict
import torch
import sys

import time

import torch.nn as nn
import torch.nn.init as init


def add_path(path):
    if path not in sys.path:
        print('Adding {}'.format(path))
        sys.path.append(path)


def torch_accuracy(output, target, topk=(1,)) -> List[torch.Tensor]:
    '''
    param output, target: should be torch Variable
    '''
    # assert isinstance(output, torch.cuda.Tensor), 'expecting Torch Tensor'
    # assert isinstance(target, torch.Tensor), 'expecting Torch Tensor'
    # print(type(output))

    topn = max(topk)
    batch_size = output.size(0)

    _, pred = output.topk(topn, 1, True, True)
    pred = pred.t()

    is_correct = pred.eq(target.view(1, -1).expand_as(pred))

    ans = []
    for i in topk:
        is_correct_i = is_correct[:i].view(-1).float().sum(0, keepdim=True)
        ans.append(is_correct_i.mul_(100.0 / batch_size))

    return ans


def mkdir(path):
    if not os.path.exists(path):
        print('creating dir {}'.format(path))
        os.mkdir(path)
    else:
        print('{} already exists.'.format(path))


def pretty_print(*values, col_width=13):
    # col_width = 13

    def format_val(v):
        if not isinstance(v, str):
            v = np.array2string(v, precision=5, floatmode='fixed')
        return v.ljust(col_width)

    str_values = [format_val(v) for v in values]
    print("   ".join(str_values))


class AvgMeter(object):
    '''
    Computing mean
    '''
    name = 'No name'

    def __init__(self, name='No name'):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = 0
        self.mean = 0
        self.num = 0
        self.now = 0

    def update(self, mean_var, count=1):
        if math.isnan(mean_var):
            mean_var = 1e6
            print('Avgmeter getting Nan!')
        self.now = mean_var
        self.num += count

        self.sum += mean_var * count
        self.mean = float(self.sum) / self.num


def make_symlink(source, link_name):
    '''
    Note: overwriting enabled!
    '''
    if os.path.exists(link_name):
        print("Link name already exist! Removing '{}' and overwriting".format(link_name))
        os.remove(link_name)
    if os.path.exists(source):
        os.symlink(source, link_name)
        return
    else:
        print('Source path not exists')


def to_onehot(inp, num_dim=10):
    # inp: (bs,) int
    # ret: (bs, num_dim) float
    # assert inp.dtype == torch.long

    batch_size = inp.shape[0]
    y_onehot = torch.FloatTensor(batch_size, num_dim).to(inp.device)
    y_onehot.zero_()
    y_onehot.scatter_(1, inp.reshape(batch_size, 1), 1)

    return y_onehot