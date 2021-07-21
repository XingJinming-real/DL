import torch
from torch import nn
from d2l import torch as d2l
from torch.utils import data


def load_data(data_array, batch_size):
    dataSet = data.TensorDataset(*data_array)
    return data.DataLoader(dataSet, batch_size, shuffle=True)


def train_concise():
    features, labels = d2l.synthetic_data(trueW, trueD, numTrain)
    trainArray = load_data((features, labels), batchSize)
    net = nn.Sequential(nn.Linear(input_size, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss()

    net[0].bias.data.zero_()
    trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
    for epoch in range(num_epoch):
        for X, y in trainArray:
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        print("epoch: ", epoch, "loss", loss(net(features), labels))


def l1Penalty(W):
    return torch.sum(torch.abs(W))


def l2Penalty(W):
    return torch.sum(W.pow(2)) / 2


def sgd(params, lr, batchSize):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batchSize
            param.grad.zero_()


def net(X, W, b):
    return torch.matmul(X, W) + b


def loss(y_hat, y):
    return torch.sum(torch.pow(y_hat - y, 2)) + l1Penalty(W)


def withoutPenalty(y_hat, y):
    return torch.sum(torch.pow(y_hat - y, 2))+l2Penalty(W)


num_epoch = 50
lr = 0.01
wd = 1e-3
numTrain = 20
input_size = 200
batchSize = 5
output_size = 1
trueW = torch.ones((input_size, 1))
trueD = torch.tensor(0.5)

features, labels = d2l.synthetic_data(trueW, trueD, numTrain)
trainIter = load_data((features, labels), batchSize)


def initParameters():
    return torch.randn((200, 1), requires_grad=True), torch.zeros(1, requires_grad=True)


W, b = initParameters()
for epoch in range(num_epoch):
    for X, y in trainIter:
        l = loss(net(X, W, b), y)
        l.backward()
        sgd([W, b], lr, batchSize)
    print("loss", loss(net(features, W, b), labels))

W, b = initParameters()
for epoch in range(num_epoch):
    for X, y in trainIter:
        l = loss(net(X, W, b), y)
        l.backward()
        sgd([W, b], lr, batchSize)
    print("lossL2", loss(net(features, W, b), labels))
# train_concise()
