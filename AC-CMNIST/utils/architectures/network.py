import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torchvision.models

import sys
import pdb


class MLP(nn.Module):
    '''
    oringal model from IRM paper
    '''

    def __init__(self, flags):
        super(MLP, self).__init__()
        self.flags = flags
        if flags.grayscale_model:
            self.lin1 = nn.Linear(14 * 14, flags.hidden_dim)
        else:
            self.lin1 = nn.Linear(2 * 14 * 14, flags.hidden_dim)
            self.lin2 = nn.Linear(flags.hidden_dim, flags.hidden_dim)
        self.lin3 = nn.Linear(flags.hidden_dim, 1)
        for lin in [self.lin1, self.lin2, self.lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self._main = nn.Sequential(self.lin1, nn.ReLU(True), self.lin2, nn.ReLU(True), self.lin3)

    def forward(self, input, inter=0):
        if self.flags.grayscale_model:
            out = input.view(input.shape[0], 2, 14 * 14).sum(dim=1)
        else:
            out = input.view(input.shape[0], 2 * 14 * 14)

        out = self.lin1(out)
        if inter == 1:
            inter_out = out
        out = F.relu(out, True)
        if inter == 2:
            inter_out = out
        out = self.lin2(out)
        if inter == 3:
            inter_out = out
        out = F.relu(out, True)
        if inter == 4:
            inter_out = out
        out = self.lin3(out)
        if inter == 5:
            inter_out = out

        # out = self._main(out)
        if inter != 0:
            return out, inter_out
        else:
            return out


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


def net():
    return nn.Sequential(nn.Conv2d(1, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2, 2),
                         nn.Conv2d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2, 2), Flatten(),
                         nn.Linear(7 * 7 * 64, 1024), nn.ReLU(), nn.Linear(1024, 10))


class cmnist_cnn(nn.Module):
    '''
    '''

    def __init__(self, ):
        super(cmnist_cnn, self).__init__()
        self._main = nn.Sequential(nn.Conv2d(2, 32, 5, padding=2), nn.ReLU(True),
                                   nn.MaxPool2d(2, 2),
                                   nn.Conv2d(32, 64, 5, padding=2), nn.ReLU(True),
                                   # nn.MaxPool2d(2, 2),
                                   nn.Conv2d(64, 32, 3, padding=0), nn.ReLU(True),
                                   Flatten(),
                                   nn.Linear(32 * 5 * 5, 256), nn.ReLU(True),
                                   nn.Linear(256, 1),
                                   # nn.Linear(7 * 7 * 64, 256), nn.ReLU(True), nn.Linear(256, 1),
                                   )

    def forward(self, input):
        # out = input.view(input.shape[0], 2, 14, 14)
        out = self._main(input)
        return out
