import torch
import torchvision
from torchvision import datasets
from torch import nn
import pandas as pd
import numpy as np
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l


def getData():
    trans = transforms.ToTensor()
    trainData = datasets.FashionMNIST(root='../../data', train=True, transform=trans)
    testData = datasets.FashionMNIST(root='../../data', train=False, transform=trans)
    trainIter = data.DataLoader(trainData, batch_size=1, shuffle=True)
    testIter = data.DataLoader(testData, batch_size=1)
    return trainIter, testIter


class leNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(1, 6, kernel_size=(3, 3)),
                                   nn.Sigmoid(),
                                   nn.AvgPool2d(kernel_size=(3, 3), stride=2),
                                   nn.Conv2d(6, 16, kernel_size=(3, 3)),
                                   nn.Sigmoid(),
                                   nn.AvgPool2d(kernel_size=(3, 3), stride=2),
                                   nn.Flatten(),
                                   nn.Linear(256, 120),
                                   nn.Sigmoid(),
                                   nn.Linear(120, 84),
                                   nn.Sigmoid(),
                                   nn.Linear(84, 10))
        # 注意输入的X为4维，分别为输入通道，批量大小维，长，宽

    def forward(self, X):
        X = X.reshape((-1, 1, 28, 28))
        return self.block(X)


def loop(model):
    X = torch.rand((1, 1, 28, 28))
    for m in model:
        X = m(X)
        print(X.shape)


def evaluate_accuracy(model, testIter):
    metricTest = d2l.Accumulator(2)
    for X, y in testIter:
        X, y = X.cuda(), y.cuda()
        y_hat = torch.argmax(model(X).reshape((1, -1)), axis=1)
        accu = int((y_hat == y).sum())
        metricTest.add(accu, len(y_hat))
    return metricTest[0] / metricTest[1]


def init_params(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


def getDevice():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    return device


def main():
    device = getDevice()
    trainIter, testIter = getData()
    model = leNet()
    model.apply(init_params)
    model.to(device)
    loss = nn.CrossEntropyLoss()
    # 分类问题使用CrossEntropyLoss，回归问题使用SME
    optimizer = torch.optim.SGD(model.parameters(), lr=0.99)
    metricTrain = d2l.Accumulator(3)
    for epoch in range(200):
        for X, y in trainIter:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            L = loss(model(X), y)
            L.backward()
            optimizer.step()
            metricTrain.add(float(L) * len(y), d2l.accuracy(model(X), y), len(y))
        print(epoch, "--:  trainAccu: ", metricTrain[1] / metricTrain[2])


main()
