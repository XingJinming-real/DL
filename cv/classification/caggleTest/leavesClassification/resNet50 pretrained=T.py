import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from torchvision import models
from torch.utils import data
import torchvision.transforms as transforms
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from torchsummary import summary


class dataSets(torch.utils.data.Dataset):
    # 继承自torch.utils.data.Dataset类，我们必须要实现__getitem__()和__len__方法

    def __init__(self, isTrain, trans):
        self.trans = trans
        self.data, self.label = getData(isTrain)
        self.isTrain = isTrain

    def __getitem__(self, item):
        if self.isTrain:
            return self.trans(self.data[item].float()), self.label[item]
        return self.data[item]

    def __len__(self):
        return len(self.data)


def getData(isTrain):
    features, label = [], []
    trainIdx = pd.read_csv('/beginner/classification/data/leavesClassification\\miniImages.csv')
    testIdx = pd.read_csv('/data/leavesClassification/test.csv')
    trainIdx = trainIdx.set_index('image')
    testIdx = testIdx.set_index('image')
    le = LabelEncoder()
    label = torch.tensor(le.fit_transform(trainIdx['label']), dtype=torch.long)
    for i in range(len(label)):
        book[label[i]] = trainIdx['label'].iloc[i]
    imgBase = 'D:\\torchProjects\\data\\leavesClassification\\'
    if isTrain:
        whichIdx = trainIdx
    else:
        whichIdx = testIdx
    for imgIdx, name in whichIdx.iterrows():
        features.append(torchvision.io.read_image(imgBase + imgIdx))
    if isTrain:
        return features, label
    return features, None


def getDevice():
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def initParams(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


device = getDevice()
book = {}
batchSize = 16
lr = 0.001
epoch = 200
net = models.resnet50(pretrained=True)
net.fc = nn.Sequential(nn.Linear(net.fc.in_features, 1024),
                       nn.BatchNorm1d(1024),
                       nn.ReLU(),
                       nn.Linear(1024, 512),
                       nn.BatchNorm1d(512),
                       nn.ReLU(),
                       nn.Linear(512, 176),
                       nn.Softmax(dim=0))
# net.fc = nn.Linear(net.fc.in_features, 176)
net.fc.apply(initParams)
net.to(device)
fcParams = list(map(id, net.fc.parameters()))
baseParams = filter(lambda x: id(x) not in fcParams, net.parameters())
lFunc = nn.CrossEntropyLoss()
opti = torch.optim.Adam(
    [{'params': net.fc.parameters(), 'lr': lr * 3},
     {'params': baseParams, 'lr': lr}])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opti, 200, 0)
kf = KFold(n_splits=10, shuffle=True)
trans = torchvision.transforms.Compose(
    [transforms.Normalize([0.485, 0.456, 0.406], [0.299, 0.224, 0.255]),
     transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()])


def accu(predict, y):
    mask = torch.ones(len(predict))
    accuNum = int(mask[torch.argmax(predict, dim=1) == y].sum())
    return accuNum


def main():
    # summary(net, (3, 224, 224), device='cuda')
    trainIter = data.DataLoader(dataSets(True, trans), batch_size=batchSize, shuffle=True)
    for perEpoch in range(epoch):
        # print(f'当前lr={scheduler.get_last_lr()}')
        trainAccu = []
        validAccu = []
        dashBoardTrain = [0, 0, 0]
        dashBoardValid = [0, 0, 0]
        # for trainIter, validIter in kf.split(trainIter):
        for X, y in trainIter:
            for i in range(4):
                X, y = X.to(device), y.to(device)
                opti.zero_grad()
                predict = net(X)
                loss = lFunc(predict, y)
                loss.backward()
                opti.step()
                dashBoardTrain[0] += len(y) * float(loss)
                dashBoardTrain[1] += accu(predict, y)
                dashBoardTrain[2] += len(y)
        # for X, y in validIter:
        #     X, y = X.to(device), y.to(device)
        #     predict = net(X)
        #     loss = lFunc(predict, y)
        #     dashBoardValid[0] += len(y) * float(loss)
        #     dashBoardValid[1] += accu(predict, y)
        #     dashBoardValid[2] += len(y)
        scheduler.step()
        # validAccu.append(dashBoardValid[1] / dashBoardValid[2])
        # plt.plot(range(1, perEpoch + 1), trainAccu, ':ro', range(1, perEpoch + 1), validAccu, ':ko')
        print(f'loss={dashBoardTrain[0] / dashBoardTrain[1]},accu={dashBoardTrain[1] / dashBoardTrain[2]}')
        # plt.plot(range(1, perEpoch + 2), trainAccu, ':ro')
        # plt.pause(0.5)


if __name__ == '__main__':
    main()
