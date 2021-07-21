import torchvision
from torch import nn
import torch
from torchvision import transforms
from d2l import torch as d2l
from torch.utils import data
import numpy as np


def getData(batchSize):
    trans = [transforms.ToTensor()]
    trans = transforms.Compose(trans)
    trainIter = torchvision.datasets.FashionMNIST(root='../data', train=True, transform=trans)
    testIter = torchvision.datasets.FashionMNIST(root='../data', train=False, transform=trans)
    return data.DataLoader(trainIter, batchSize, shuffle=True), \
           data.DataLoader(testIter, batchSize)


inputSize = 784
l1 = 256
l1DropoutP = 0.5
l2 = 128
l2DropoutP = 0.1
outSize = 10
lr = 0.3
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(inputSize, l1),
                    nn.Sigmoid(),
                    nn.Dropout(l1DropoutP),
                    # nn.Softmax(1),
                    nn.Linear(l1, l2),
                    nn.Tanh(),
                    nn.Dropout(l2DropoutP),
                    # nn.Softmax(1),
                    nn.Linear(l2, outSize))


def init_params(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, np.sqrt(2 / (m.in_features + m.out_features)))
        m.bias.data.zero_()


def accuracy(X, y):
    mask = torch.argmax(net(X), 1) == y
    mask = mask.type(torch.float)
    return torch.sum(mask).detach().item()


net.apply(init_params)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr)
epochNum = 10
batchSize = 256
trainIter, testIter = getData(batchSize)
metricTrain = d2l.Accumulator(3)
net.to('cuda')
for epoch in range(epochNum):
    for X, y in trainIter:
        X, y = X.cuda(), y.cuda()
        optimizer.zero_grad()
        l = loss(net(X), y)
        l.backward()
        optimizer.step()
        metricTrain.add(float(l) * len(y), accuracy(X, y), len(y))
    print("loss", metricTrain[0] / metricTrain[2],
          "accu", metricTrain[1] / metricTrain[2])
