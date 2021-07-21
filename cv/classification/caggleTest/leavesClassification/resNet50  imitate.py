# 此代码为阅读相关范例代码后重新写的
import random
import torchvision
import torch
import pandas as pd
import timm
import torch.nn as nn
from torch.utils import data
import torchvision.transforms as transform
from sklearn.model_selection import KFold
import numpy as np

stringToNum = {}
numToString = {}


class dataSet(torch.utils.data.Dataset):
    # 继承自torch.utils.data.Dataset类，我们必须要实现__getitem__()和__len__方法

    def __init__(self, X, y, isTrain, trans):
        self.feature = X
        self.label = y
        self.isTrain = isTrain
        self.trans = trans

    def __getitem__(self, item):
        if self.isTrain:
            return self.trans(self.feature[item].float()), self.label[item]
        else:
            return self.trans(self.feature[item].float())

    def __len__(self):
        return len(self.feature)


def getData(imgIdx, isTrain):
    global stringToNum, numToString
    trainCsv = pd.read_csv('/data/leavesClassification/mini.csv').reset_index(drop=True)
    feature = []
    for i in imgIdx:
        feature.append(torchvision.io.read_image('D:/torchProjects/data/leavesClassification/' +
                                                 trainCsv['image'].iloc[i]).float())
        # 注意要加上.float()变成float，不然就会导致归一化时出现零
    if isTrain:
        label = list(map(lambda x: stringToNum[x], trainCsv['label'].iloc[imgIdx]))
        return feature, label
    return feature, None


def getTrainValidIter(k=5):
    global stringToNum, numToString, uniformTrans
    trainCsv = pd.read_csv('/data/leavesClassification/mini.csv').reset_index(drop=True)
    testCsv = pd.read_csv('/data/leavesClassification/test.csv').reset_index(drop=True)
    label = np.unique(trainCsv['label'].values)
    labelNum = len(label)
    stringToNum = dict(zip(label, range(labelNum)))
    numToString = dict(zip(range(labelNum), label))

    sampleNum = range(len(trainCsv))
    kf = KFold(k)
    # 在使用kf时，注意对标号进行分割，然后根据标号读取数据，注意使用torchvision.io.read_img()就不要再加上toTensor了
    for trainIdx, validIdx in kf.split(sampleNum):
        trainIter = data.DataLoader(dataSet(*getData(trainIdx, True), isTrain=True, trans=uniformTrans),
                                    batch_size=batchSize, pin_memory=True)
        validIter = data.DataLoader(dataSet(*getData(validIdx, True), isTrain=True, trans=uniformTrans),
                                    batch_size=batchSize, pin_memory=True)
        # 注意是yield
        yield trainIter, validIter


def mixUp(x, y, lam, alpha=1):
    # 注意使用mixup的时候
    # 要求lam服从beta分布，同时标签不变，只是在计算梯度时变成了该样本进行对两个标签的计算，然后加权平均
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 0.5
    idx = torch.randperm(len(y))
    xShuffle = x[idx, :, :, :]
    # 在同一个batch内进行混合
    xNew = lam * x + (1 - lam) * xShuffle
    yNew = y[idx].long()
    return xNew, yNew, lam


def getFragment(x):
    w, h = x.shape[-2], x.shape[-1]
    cx = random.randint(0, w)
    cy = random.randint(0, h)
    hw = random.uniform(0, 0.5)
    hh = random.uniform(0, 0.5)
    x1, y1, x2, y2 = int(np.clip(cx - hw * w, 0, w)), int(np.clip(cy - hh * h, 0, h)), \
                     int(np.clip(cx + hw * w, 0, w)), int(np.clip(cy + hh * h, 0, h))

    return x1, y1, x2, y2


def cutFix(x, y):
    xNewIdx = torch.randperm(len(y))
    x1, y1, x2, y2 = getFragment(x)
    x[:, :, x1:x2, y1:y2] = x[xNewIdx, :, x1:x2, y1:y2]
    # 注意是4维向量
    lam = (x2 - x1) * (y2 - y1) / x.shape[-2] * x.shape[-1]
    yNew = y[xNewIdx]
    return x, yNew, 1 - lam


def randomRotation(x, y, lam, minR=0, maxR=360):
    xNew = torchvision.transforms.RandomRotation((minR, maxR))(x)
    return xNew, y, lam


def initParams(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def getDevice():
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


# device = 'cpu'
device = getDevice()
net = timm.create_model('resnet50', True)
optimizer = torch.optim.AdamW(filter(lambda x: x.requires_grad, net.parameters()), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0)
loss = nn.CrossEntropyLoss()
epochNum = 50
batchSize = 16
uniformTrans = transform.Compose([transform.CenterCrop(200), transform.Resize((224, 224)),
                                  transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                                                      inplace=True)])


def train():
    # net.cuda()
    net.to(device)
    net.fc = nn.Sequential(nn.Linear(2048, 512), nn.BatchNorm1d(512),
                           nn.Dropout(0.5), nn.ReLU(), nn.Linear(512, 176))
    net.fc.apply(initParams)
    for k, layer in enumerate(net.children()):
        if k == 6:
            break
        layer.requires_grad = False
    for epoch in range(epochNum):
        trainLoss = []
        validLoss = []
        trainAccu = []
        validAccu = []
        for k, (trainIter, validIter) in enumerate(getTrainValidIter(5)):
            tempLoss = 0
            tempAccuNum = 0
            tempLen = 0
            for x, y in trainIter:
                key = random.uniform(0, 1)
                # key = 0.7
                if key < 0.25:
                    xNew, yNew, lam = randomRotation(*mixUp(x, y, 1))
                elif key < 0.5:
                    xNew, yNew, lam = randomRotation(*cutFix(x, y))
                elif key < 0.75:
                    xNew, yNew, lam = randomRotation(x, y, 0)
                else:
                    xNew, yNew, lam = x, y, 0
                x, y, xNew, yNew = x.to(device), y.to(device), xNew.to(device), yNew.to(device)
                xNew, yNew = map(torch.autograd.Variable, (xNew, yNew))
                net.to(device)
                predict = net(x)
                l1 = loss(predict, y) * lam
                l2 = loss(net(xNew), yNew) * (1 - lam)
                l = l1 + l2
                l.backward()
                tempLoss += l.item()
                predict = predict.argmax(dim=1)
                tempAccuNum += torch.sum((predict == y)).cpu().item()
                # 注意要加上cpu
                tempLen += len(x)
            trainLoss.append(tempLoss)
            trainAccu.append(tempAccuNum / tempLen)
            tempLoss = 0
            tempAccuNum = 0
            tempLen = 0
            with torch.no_grad():
                # 不计算梯度
                for x, y in validIter:
                    x, y = x.to(device), y.to(device)
                    predict = net(x)
                    tempLoss += loss(predict, y).item()
                    predict = predict.argmax(dim=1)
                    tempAccuNum += torch.sum(predict == y).item()
                    tempLen += len(x)
            validLoss.append(tempLoss)
            validAccu.append(tempAccuNum / tempLen)
        print(f'第{epoch}个epoch\ntrainLoss   {np.mean(trainLoss)},validLoss   {np.mean(validLoss)}')
        print(f'trainAccu   {np.mean(trainAccu)},validAccu  {np.mean(validAccu)}')
        print('-' * 20)


def getTestIter():
    testCsv = pd.read_csv('/data/leavesClassification/test.csv')
    testNum = len(testCsv)
    testData = []
    for i in range(testNum):
        testData.append(
            torchvision.io.read_image('D:/torchProjects/data/leavesClassification/images/' + testNum['image'].iloc[i]))
    testIter = data.DataLoader(*dataSet(testData, None, False, trans=uniformTrans))
    return testIter


def test():
    k = 1
    net.load_state_dict(torch.load(f'./saved{k}.pth'))
    sampleSubmission = pd.read_csv('/data/leavesClassification/sample_submission.csv')
    testIter = getTestIter()
    predict = []
    with torch.no_grad():
        for img in getTestIter():
            predict.append(numToString[torch.argmax(net(img), dim=1)])
    predict = pd.DataFrame(predict)
    sampleSubmission = pd.concat((sampleSubmission, predict), axis=1)
    sampleSubmission.to_csv('D:/torchProjects/data/leavesClassification/submission.csv')


if __name__ == '__main__':
    train()
