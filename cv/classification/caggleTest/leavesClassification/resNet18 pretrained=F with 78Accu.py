import time
import d2l.torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import torchsummary
import torchvision.io
from torchvision import datasets
from torchvision import transforms
from torch.utils import data
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import os
import kaggle


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


def initParameters(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(2, 2)),
                   nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1))
b1.apply(initParameters)


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
b2.apply(initParameters)
b3.apply(initParameters)
b4.apply(initParameters)
b5.apply(initParameters)
net = nn.Sequential(b1, b2, b3, b4, b5, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                    nn.Linear(512, 176), nn.Softmax(dim=0))
net.apply(initParameters)


class DataSets(torch.utils.data.Dataset):
    def __init__(self, isTrain):
        self.features, self.labels = getData(isTrain)
        self.isTrain = isTrain

    def __getitem__(self, item):
        if self.isTrain:
            return self.features[item].float(), self.labels[item]
        else:
            return self.features[item].float()

    def __len__(self):
        return len(self.features)


labelNum = 0


def getData(isTrain):
    global labelNum
    trainIdx = pd.read_csv('/beginner/classification/data/leavesClassification\\miniImages.csv')
    testIdx = pd.read_csv('/data/leavesClassification/test.csv')
    le = LabelEncoder()
    label = torch.tensor(le.fit_transform(trainIdx['label']), dtype=torch.long)
    labelNum = len(label.unique())
    dataBase = 'D:\\torchProjects\\data\\leavesClassification\\'
    Data = []
    if isTrain:
        idx = trainIdx.set_index('image')
    else:
        label = None
        idx = testIdx.set_index('image')
    for Idx, (imgName, target) in enumerate(idx.iterrows()):
        if not Idx % 1000:
            print(Idx)
        Data.append(torchvision.io.read_image(dataBase + imgName))
    return Data, label


def loadData(train=True):
    Iter = data.DataLoader(DataSets(train), 64 if not train else batchSize)
    return Iter


def accuracy(predict, y):
    mask = torch.ones(len(y))
    mask = mask[predict.max(axis=1).indices == y]
    return float(sum(mask))


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
lr, numEpoch, batchSize = 0.22, 200, 64


# newNet = torchvision.models.resnet50(True)
# newNet.fc = nn.Linear(newNet.fc.in_features, labelNum)
# nn.init.xavier_uniform_(newNet.fc.weight)


def main():
    # torchsummary.summary(net, (3, 64, 64), device='cpu')
    # input()
    trainIter = loadData(True)
    # trainIter = [100, 10]
    print("已读取完数据,使用{}".format(device))
    loss = nn.CrossEntropyLoss()
    # paramList = [param for name, param in newNet.named_parameters() if name not in ['fc.weight', 'fc.bias']]
    # optimizer = torch.optim.SGD([{'params': paramList}, {'params': newNet.fc.parameters(), 'lr': 0.1}], lr=0.01)
    net.load_state_dict(torch.load('./savedModel11.pkl'))
    # net[7] = nn.Sequential(nn.Linear(512, 256),
    #                        nn.BatchNorm1d(256),
    #                        nn.ReLU(),
    #                        nn.Dropout(),
    #                        nn.Linear(256, 176))
    # net[7].apply(initParameters)
    newParamsId = list(map(id, net[7].parameters()))
    baseParams = filter(lambda x: id(x) not in newParamsId, net.parameters())
    # net.to(device)
    flag = 0
    lrList = [0.002, 0.0005]
    threshList = [50]
    lastScore = 0
    for epoch in range(numEpoch):
        L = 0
        accumulator = [0, 0, 0]
        optimizer = torch.optim.SGD([{'params': baseParams},
                                     {'params': net[7].parameters(), 'lr': lrList[flag] * 200}],
                                    lr=lrList[flag])
        if flag < len(threshList) and epoch > threshList[flag]:
            flag += 1
            optimizer = torch.optim.SGD([{'params': baseParams},
                                         {'params': net[7].parameters(), 'lr': lrList[flag] * 200}],
                                        lr=lrList[flag])
        for X, y in trainIter:
            # X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            predict = torch.argmax(net(X), dim=1)
            l = loss(predict, y)
            l.backward()
            optimizer.step()
            L += l * len(y)
            accumulator[0] += float(l) * len(y)
            accumulator[1] += accuracy(predict, y)
            accumulator[2] += len(y)
        print(f'loss on train {accumulator[0] / accumulator[2]}, accu on train {accumulator[1] / accumulator[2]}')
        if accumulator[1] / accumulator[2] > 0.85 and accumulator[1] / accumulator[2] > lastScore:
            lastScore = accumulator[1] / accumulator[2]
            torch.save(net.state_dict(), './savedModel' + str(epoch) + '.pkl')
        # net.load_state_dict(torch.load('./savedModel.pkl'))
        # if accumulator[0] / accumulator[2] > bestScore:
        #     bestScore = accumulator[0] / accumulator[2]
        #     torch.save(net.state_dict(), './resNet18' + str(epoch) + '.pt')
        # pltX.append(epoch)
        # pltY.append(accumulator[0] / accumulator[2])
        # pltAccu.append(accumulator[1] / accumulator[2])
        # plt.plot(pltX, pltY, 'k')
        # plt.show()
        # plt.pause(0.01)


def test():
    net.load_state_dict(torch.load('./savedModel11.pkl', map_location='cpu'))
    # net.to('cuda')
    trainCSV = pd.read_csv('/data/leavesClassification/train.csv')
    testIter = loadData(False)
    le = LabelEncoder()
    label = pd.DataFrame(le.fit_transform(trainCSV['label']))
    benchmark = pd.concat((label, trainCSV), axis=1)
    benchmark = benchmark.drop('image', axis=1).set_index(0)
    with open('D:\\testResult.txt', 'w') as f:
        for X in testIter:
            # X = X.to('cuda')
            y = torch.argmax(net(X), dim=1).tolist()
            for perSample in y:
                f.write(benchmark.loc[perSample].iloc[0].label + '\n')
    # plt.show()


if __name__ == "__main__":
    # aaa = torchvision.models.resnet50()
    # a = torchsummary.summary(aaa, (3, 224, 224), device='cpu')
    # print(a)
    main()
    # test()
