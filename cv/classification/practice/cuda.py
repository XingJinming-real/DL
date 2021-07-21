import torch
import torchvision
from torchvision import datasets
from torch import nn
import pandas as pd
import numpy as np
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

epochNum = 200
lr = 0.3
weightDecay = 0.01
# momentum = 0.8
batchSize = 64
inputSize = 28 * 28
l1 = int(inputSize * 0.8)
l2 = int(inputSize * 0.5)
l3 = int(inputSize * 0.2)


class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(nn.Flatten(),
                                   nn.Linear(inputSize, l1),
                                   nn.ReLU(),
                                   # nn.Dropout(0.5),
                                   nn.Linear(l1, l2),
                                   nn.ReLU(),
                                   # nn.Dropout(0.1),
                                   nn.Linear(l2, l3),
                                   nn.ReLU(),
                                   nn.Linear(l3, 10))

    def forward(self, X):
        return self.block(X)


def init_params(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight)
        m.bias.data.zero_()


def getData():
    global inputSize
    trans = [transforms.ToTensor()]
    trans = transforms.Compose(trans)
    # trainD = torchvision.datasets.CIFAR10('../data', download=True)
    # trainD = torchvision.datasets.CIFAR10('../data', train=False, download=True)
    trainD = torchvision.datasets.FashionMNIST(root='../data', train=True, transform=trans, download=False)
    testD = torchvision.datasets.FashionMNIST(root='../data', train=False, transform=trans, download=False)
    inputSize = trainD.data[0].numel()
    return data.DataLoader(trainD, batchSize, shuffle=True), data.DataLoader(testD, batchSize)


def trainEpoch():
    for X, y in trainIter:
        X = X.cuda()
        y = y.cuda()
        opt.zero_grad()
        yHat = net(X)
        L = loss(yHat, y)
        L.backward()
        opt.step()
        metricTrain.add(float(L) * len(y), d2l.accuracy(yHat, y), len(y))
    print("trainLoss: ", metricTrain[0] / metricTrain[2], 'trainAcc: ', metricTrain[1] / metricTrain[2])


if __name__ == "__main__":

    torch.device('cuda:0')
    metricTrain = d2l.Accumulator(3)
    metricTest = d2l.Accumulator(2)
    net = model()
    net.apply(init_params)
    net.to('cuda:0')
    loss = nn.CrossEntropyLoss()
    # loss.to('cuda:0')
    opt = torch.optim.SGD(net.parameters(), lr,
                          weight_decay=weightDecay)
    trainIter, testIter = getData()
    for epoch in range(epochNum):
        print(f'epoch=: {epoch}')
        for X, y in trainIter:
            X = X.cuda()
            y = y.cuda()
            opt.zero_grad()
            l = loss(net(X), y)
            l.backward()
            opt.step()
            metricTrain.add(float(l) * len(y), d2l.accuracy(net(X), y), len(y))
        print("loss", metricTrain[0] / metricTrain[2],
              "accu", metricTrain[1] / metricTrain[2])
    net.save('fashionMnist')
