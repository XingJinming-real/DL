import torch
import torchvision
from torchvision import datasets
from torch import nn
import pandas as pd
import numpy as np
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
from torchsummary import summary
import matplotlib.pyplot as plt
import time


def initParameters():
    for l in net:
        if type(l) == nn.Conv2d or type(l) == nn.Linear:
            nn.init.xavier_uniform_(l.weight)


def vgg_block(num_convolutions, in_channels, out_channels):
    layers = []
    for _ in range(num_convolutions):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(2, stride=2))
    return nn.Sequential(*layers)


conv_arch = ((1, 64), (2, 128))


def VGG():
    convolutionBlock = []
    in_channels = 1
    out_channels = 1
    for (convolutionNum, out_channels) in conv_arch:
        convolutionBlock.append(vgg_block(convolutionNum, in_channels, out_channels))
        in_channels = out_channels
    return nn.Sequential(*convolutionBlock, nn.Flatten(), nn.Linear(out_channels * 7 * 7, 4096),
                         nn.ReLU(), nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(),
                         nn.Dropout(0.5), nn.Linear(4096, 10))


def accuracy(X, y):
    mask = torch.argmax(net(X), 1) == y
    mask = mask.type(torch.float)
    return torch.sum(mask).detach().item()


batchSize = 8
lr = 0.3
numEpoch = 10
trans = [transforms.ToTensor()]
trans.insert(0, transforms.Resize((28, 28)))
trans = transforms.Compose(trans)
trainData = datasets.MNIST(root='../data', transform=trans)
trainIter = data.DataLoader(trainData, batch_size=batchSize, shuffle=True)
testData = datasets.MNIST(root='../data', transform=trans, train=False)
testIter = data.DataLoader(testData, shuffle=False)
net = VGG()
# initParameters()
net.to('cuda')
loss = nn.CrossEntropyLoss()
updater = torch.optim.SGD(net.parameters(), lr)
accumulator = d2l.Accumulator(3)
for epoch in range(numEpoch):
    num = 0
    begin = time.time()
    print("***")
    for X, y in trainIter:
        X = X.cuda()
        y = y.cuda()
        updater.zero_grad()
        l = loss(net(X), y)
        predict = net(X)
        l.backward()
        updater.step()
        num += len(y)
        end = time.time()
        if end - begin > 10:
            break
        accumulator.add(len(y) * float(l), accuracy(X, y), len(y))
    print(num / 10)
    input()
    print("loss{}, accu{}".format(accumulator[0] / accumulator[2],
                                  accumulator[1] / accumulator[2]))
# X = torch.randn(size=(1, 1, 224, 224))
# for blk in net:
#     X = blk(X)
#     print(blk.__class__.__name__, 'output shape:\t', X.shape)
