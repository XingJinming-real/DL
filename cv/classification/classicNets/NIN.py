import d2l.torch
import numpy as np
import torch.nn as nn
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils import data
import matplotlib.pyplot as plt


def NINBlock(inChannel, outChannel, kernelSize, stride, padding):
    blk = [nn.Conv2d(inChannel, outChannel, (kernelSize, kernelSize), stride=(stride, stride), padding=padding),
           nn.ReLU(), nn.Conv2d(outChannel, outChannel, (1, 1), (1, 1)), nn.ReLU(),
           nn.Conv2d(outChannel, outChannel, (1, 1), (1, 1)), nn.ReLU()]
    return nn.Sequential(*blk)


# blkDict = [(32, 3, 1), (64, 3, 1), (32, 3, 1), (10, 3, 1)]

def initParameters(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


def NIN():
    block = []
    net = nn.Sequential(NINBlock(1, 96, 11, 4, 0), nn.MaxPool2d(3, stride=2),
                        NINBlock(96, 256, 5, 1, 2), nn.MaxPool2d(3, stride=2),
                        NINBlock(256, 384, 3, 1, 1), nn.MaxPool2d(3, stride=2),
                        nn.Dropout(0.5), NINBlock(384, 10, 3, 1, 1), nn.AdaptiveAvgPool2d(1),
                        nn.Flatten(), nn.Softmax(dim=0)
                        )
    return net


def getData():
    trans = [transforms.ToTensor()]
    trans.insert(0, transforms.Resize((224, 224)))
    trans = transforms.Compose(trans)
    trainData = datasets.MNIST(root='../data', transform=trans, train=True)
    testData = datasets.MNIST(root='../data', transform=trans, train=False)
    trainIter = data.DataLoader(trainData, batch_size=batchSize, shuffle=True)
    testIter = data.DataLoader(testData, batch_size=1)
    return trainIter, testIter


def accuracy(predict, y):
    mask = torch.ones(len(y))
    mask = mask[predict.max(axis=1).indices == y]
    return float(sum(mask) / len(y))


lr = 0.23
batchSize = 128
epochNum = 20
if torch.cuda.is_available():
    net = NIN().to('cuda')
else:
    net = NIN()


def main():
    net.apply(initParameters)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr, momentum=0.9)
    trainIter, testIter = getData()
    pltX = []
    pltY = []
    plt.ion()
    for epoch in range(epochNum):
        L = 0
        accumulator = [0, 0, 0]
        for X, y in trainIter:
            if torch.cuda.is_available():
                X, y = X.cuda(), y.cuda()
            optimizer.zero_grad()
            predict = net(X)
            l = loss(predict, y)
            l.backward()
            optimizer.step()
            L += l
            accumulator[0] += float(l)
            accumulator[1] += accuracy(predict, y)
            accumulator[2] += len(y)
        print(f'loss on train {accumulator[0] / accumulator[2]}, accu on train {accumulator[1] / accumulator[2]}')
        pltX.append(epoch)
        pltY.append(accumulator[0] / accumulator[2])
        plt.plot(pltX, pltY)
    plt.ioff()
    plt.show()


main()
