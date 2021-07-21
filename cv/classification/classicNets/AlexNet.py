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


def initParameters(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def main():
    trans = [transforms.ToTensor()]
    trans.insert(0, transforms.Resize((224, 224)))
    trans = transforms.Compose(trans)
    trainData = datasets.CIFAR10(root='../../data', transform=trans)
    trainIter = data.DataLoader(trainData, batch_size=64, shuffle=True)
    testData = datasets.CIFAR10(root='../../data', transform=trans, train=False)
    testIter = data.DataLoader(testData, shuffle=False)
    net = nn.Sequential(nn.Conv2d(3, 96, kernel_size=(11, 11), stride=(4, 4)),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=3, stride=2),
                        nn.Conv2d(96, 256, kernel_size=(5, 5), padding=(2, 2)),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=3, stride=2),
                        nn.Conv2d(256, 384, kernel_size=(3, 3), padding=(1, 1)),
                        nn.ReLU(),
                        nn.Conv2d(384, 384, kernel_size=(3, 3), padding=(1, 1)),
                        nn.ReLU(),
                        nn.Conv2d(384, 256, kernel_size=(3, 3), padding=(1, 1)),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=3, stride=2),
                        nn.Flatten(),
                        nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(0.5),
                        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
                        nn.Linear(4096, 10))
    loss = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), lr=0.0001)
    # net.to('cuda')
    Epoch = 20
    for epoch in range(Epoch):
        for X, y in trainIter:
            # X = X.cuda()
            # y = y.cuda()
            opt.zero_grad()
            l = loss(net(X), y)
            l.backward()
            opt.step()
            print(f'Loss {float(l) * len(y)}')
        print("accu{}".format(accuracyTest(net, testIter)))


def accuracyTest(net, testIter):
    for X, y in testIter:
        y_hat = net(X)
    return int(torch.sum(y_hat == y)) / len(y)


def loopModel(net):
    tempX = torch.rand((64, 3, 224, 224))
    for m in net:
        tempX = m(tempX)
        print(m.__class__.__name__, tempX.shape)


if __name__ == "__main__":
    main()
