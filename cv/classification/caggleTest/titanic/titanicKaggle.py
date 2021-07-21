import random
from sklearn.impute import SimpleImputer
import pandas as pd
import torchvision
from torchvision import transforms
import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt
import numpy as np
from torch.utils import data
from torch import nn
import torchvision
import os
from sklearn.preprocessing import scale

batchSize = 64
numEpoch = 200
inputSize = 784
outputSize = 2


def getDataIter(trainArray):
    dataSet = data.TensorDataset(*trainArray)
    return data.DataLoader(dataSet, batch_size=batchSize, shuffle=True)


def preprocessData(mode='mean'):
    global inputSize
    # trainData = pd.read_csv("E://kaggle//titanic//train.csv")
    # label = torch.tensor(trainData.Survived.values.copy())
    # trainData.drop('Survived', axis=1, inplace=True)
    # trainData.drop(['PassengerId', 'Name'], axis=1, inplace=True)
    # testData = pd.read_csv("E://kaggle//titanic//test.csv")
    # testData.drop(['PassengerId', 'Name'], axis=1, inplace=True)
    # Data = pd.concat([trainData, testData], ignore_index=True)
    # trainShape = trainData.shape
    # inputSize = Data.shape[1]
    # Data = pd.get_dummies(Data)
    # oriColumns = Data.columns
    # Data.columns = np.arange(len(Data.columns))
    # for perCol in range(Data.shape[1]):
    #     if Data[perCol].isna().any():
    #         Data[perCol] = Data[perCol].fillna(Data[perCol].mean())
    # Data.columns = oriColumns
    # trainData = Data.iloc[:trainShape[0]].copy()
    # trainData['label'] = label
    # trainData.to_csv('./trainData.csv')
    # Data.iloc[trainShape[0]:].to_csv('./testData.csv')
    trainData = pd.read_csv('./titanic/trainData.csv')
    label = trainData['label']
    trainData = trainData.drop(['label'], axis=1)
    testData = pd.read_csv('./titanic/testData.csv')
    Data = pd.concat([trainData, testData], axis=0)
    Data = scale(Data)
    trainData = torch.tensor(Data[:trainData.shape[0]]).to(torch.float32)
    testData = torch.tensor(Data[trainData.shape[0]:]).to(torch.float32)
    trainIter = getDataIter((trainData, torch.tensor(label).to(torch.long)))
    inputSize = testData.shape[1]
    return trainIter, testData
    # le = LabelEncoder()
    # for i in range(trainShape[1]):
    #     if Data[i].isna().any():
    #         if Data[i].dtype == 'object':
    #             impInput = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    #             Data[i] = impInput.fit_transform(Data[i].values.reshape(-1, 1))
    #             Data[i] = le.fit_transform(Data[i])
    #         else:
    #             impInput = SimpleImputer(missing_values=np.nan, strategy=mode)
    #             Data[i] = impInput.fit_transform(Data[i].values.reshape(-1, 1))
    #     else:
    #         if Data[i].dtype == 'object':
    #             Data[i] = le.fit_transform(Data[i])
    # # Data = Data.astype(np.float32)
    # trainData = torch.tensor(Data.iloc[:trainShape[0]].values).to(torch.float32)
    # testData = torch.tensor(Data.iloc[trainShape[0]:].values).to(torch.float32)
    # trainIter = getDataIter((trainData, label))
    # return trainIter, testData


def getData(batchSize):
    trans = [transforms.ToTensor()]
    trans = transforms.Compose(trans)
    trainIter = torchvision.datasets.FashionMNIST(root='../data', train=True, transform=trans)
    testData = torchvision.datasets.FashionMNIST(root='../data', train=False, transform=trans)
    return data.DataLoader(trainIter, batchSize, shuffle=True), testData


def model():
    p1 = 0.5
    p2 = 0.1
    l1 = int(0.8 * inputSize)
    l2 = int(0.5 * inputSize)
    l3 = int(0.2 * inputSize)
    net = nn.Sequential(
        nn.Linear(inputSize, l1),
        nn.ReLU(),
        nn.Dropout(p1),
        nn.Linear(l1, l2),
        nn.ReLU(),
        nn.Linear(l2, l3),
        nn.ReLU(),
        nn.Dropout(p2),
        nn.Linear(l3, outputSize))
    return net


def init_params(m):
    if m == nn.Linear:
        nn.init.normal_(m.weight, np.sqrt(2 / (m.input_features + m.out_features)))
        m.bias.data.rand_(0)


def main():
    global inputSize, outputSize
    lr = 0.3
    outputSize = 2
    loss = nn.CrossEntropyLoss()
    trainIter, testData = preprocessData()
    net = model()
    net.apply(init_params)
    optimizer = torch.optim.SGD(net.parameters(), lr, weight_decay=1e-4)
    metric = d2l.Accumulator(3)
    for epoch in range(numEpoch):
        for X, y in trainIter:
            l = loss(net(X), y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            metric.add(float(l) * len(y), d2l.accuracy(net(X), y), len(y))
        print("loss: ", metric[0] / metric[2], "trainAccu:", metric[1] / metric[2])
    result = pd.DataFrame(np.argmax(net(testData).detach().numpy(), axis=1))
    try:
        result.index = np.arange(892, 892 + result.shape[0])
        result.to_csv('./titanic/submitTitanicTest.csv')
    except:
        result.to_csv('./titanic/submitTitanicTest.csv')


if __name__ == "__main__":
    main()
