import torchvision
from torch import nn
import torch
from torchvision import transforms
from d2l import torch as d2l
from torch.utils import data
import numpy as np
import torch.nn.functional as F


def try_gpu(i=0):
    if torch.cuda.device_count() > i:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_allGpu():
    devices = torch.device([f'cuda{i}' for i in range(torch.cuda.device_count())])
    return devices if devices else [torch.device('cpu')]


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 2)
        self.rand_weight = torch.rand((256, 256)) + 1

    def forward(self, X):
        X = self.hidden(X)
        X = torch.matmul(X, self.rand_weight)
        return F.relu(self.out(X))


class mySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for block in args:
            self._modules[block] = block

    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X


class complexSequential(nn.Module):

    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(nn.Linear(20, 15), nn.ReLU(), nn.Linear(15, 20), nn.ReLU())
        self.block2 = nn.Sequential()
        self.getBlock2()
        self.block3 = nn.Sequential(self.block2, nn.Linear(20, 2))

    def getBlock2(self):
        for i in range(3):
            self.block2.add_module(str(i) + "_block", self.block1)

    def forward(self, X):
        return self.block3(X)


def init_params(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight)


class myLinearLayer(nn.Module):
    def __init__(self, inSize, outSize):
        super().__init__()
        self.weight = nn.Parameter(torch.rand((inSize, outSize)))
        self.bias = nn.Parameter(torch.zeros(outSize))

    def forward(self, X):
        linear = torch.matmul(X, self.weight) + self.bias
        return F.relu(linear)


# linear = myLinearLayer(256, 2)
# netSequential = nn.Sequential(nn.Linear(128, 256), nn.ReLU())
# netSequential.state_dict()
# netSequential.apply(init_params)
# print(netSequential(torch.arange(100, dtype=torch.float32).reshape((20, -1))))
# net = MLP()
# print(net(torch.rand((5, 20))))
net = complexSequential()
print(net(torch.rand((2, 20))))

# print(net.state_dict())
# print(netSequent.weight)
# print(netSequential(torch.rand(5, 128)))
