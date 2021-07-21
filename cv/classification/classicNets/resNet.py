import time
import d2l.torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils import data
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from PIL import Image


class residual(nn.Module):
    def __init__(self, inputChannels, outputChannels, stride=1):
        super(residual, self).__init__()
        self.conv1 = nn.Conv2d(inputChannels, outputChannels, kernel_size=(3, 3), stride=stride, padding=(1, 1))
        self.conv2 = nn.Conv2d(outputChannels, outputChannels, kernel_size=(3, 3), padding=(1, 1))
        if stride != 1:
            self.conv3 = nn.Conv2d(inputChannels, outputChannels, stride=stride, kernel_size=(1, 1))
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(outputChannels)
        self.bn2 = nn.BatchNorm2d(outputChannels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.relu(y)


b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(2, 2)),
                   nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1))


def resnetBlock(inputChannel, outputChannel, numResidual, first=False):
    blk = []
    for _ in range(numResidual):
        if not first and _ == 0:
            blk.append(residual(inputChannel, outputChannel, 2))
        else:
            blk.append(residual(outputChannel, outputChannel))
    return blk


b2 = nn.Sequential(*resnetBlock(64, 64, 2, first=True))
b3 = nn.Sequential(*resnetBlock(64, 128, 2))
b4 = nn.Sequential(*resnetBlock(128, 256, 2))
b5 = nn.Sequential(*resnetBlock(256, 512, 2))
net = nn.Sequential(b1, b2, b3, b4, b5, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(512, 10),
                    nn.Softmax(dim=0))
x = torch.rand((1, 1, 224, 224))


def getData():
    trainIdx = pd.read_csv('../data/leavesClassification/train.csv')
    le = LabelEncoder()
    label = le.fit_transform(trainIdx['label'])
    return data.DataLoader(datasets.FashionMNIST('../../data', transform=transforms.ToTensor()), batch_size=batchSize,
                           shuffle=True), \
           data.DataLoader(datasets.FashionMNIST('../../data', transform=transforms.ToTensor(), train=False))


def accuracy(predict, y):
    mask = torch.ones(len(y))
    mask = mask[predict.max(axis=1).indices == y]
    return float(sum(mask) / len(y))


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

lr, numEpoch, batchSize = 0.4, 100, 256
# trainIter, testIter = getData()
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr, momentum=0.9)
pltX = []
pltY = []
plt.ion()
net.to(device)
bestScore = 0
print(net(torch.rand((1, 1, 224, 224))))
# for epoch in range(numEpoch):
#     L = 0
#     accumulator = [0, 0, 0]
#     # num = 0
#     begin = time.time()
#     for X, y in trainIter:
#         if torch.cuda.is_available():
#             X, y = X.to(device), y.to(device)
#         optimizer.zero_grad()
#         predict = net(X)
#         l = loss(predict, y)
#         l.backward()
#         optimizer.step()
#         L += l * len(y)
#         # num += len(y)
#         accumulator[0] += float(l) * len(y)
#         accumulator[1] += accuracy(predict, y)
#         accumulator[2] += len(y)
#     print(f'loss on train {accumulator[0] / accumulator[2]}, accu on train {accumulator[1] / accumulator[2]}')
#     # if accumulator[0] / accumulator[2] > bestScore:
#     #     bestScore = accumulator[0] / accumulator[2]
#     #     torch.save(net.state_dict(), './resNet18' + str(epoch) + '.pt')
#     pltX.append(epoch)
#     pltY.append(accumulator[0] / accumulator[2])
#     plt.plot(pltX, pltY)
#     plt.show()
#     plt.pause(0.5)
# plt.ioff()
# # plt.show()
