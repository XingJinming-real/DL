import random
import torchvision
from torchvision import transforms
import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt
import numpy as np
from torch.utils import data
from torch import nn
import torchvision
import cv2
from IPython import display
import keras as kr

# import d2lzh


def accuracy(y_hat, y):
    y_hat = y_hat.type(y.dtype)
    if len(y_hat) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()  # 告诉pytorch不要计算梯度
    metric = d2l.Accumulator(2)  # 含有两个元素的累加器
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_epoch(net, train_iter, loss, optimiser):
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = d2l.Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(optimiser, torch.optim.Optimizer):
            optimiser.zero_grad()
            l.backward()
            optimiser.step()
            metric.add(float(l) * len(y), accuracy(y_hat, y),
                       y.numel())
        else:
            l.sum().backward()
            optimiser([W, b], lr=0.8, batch_size=batch_size)
            metric.add(float(l.sum()), accuracy(y_hat, y),
                       y.numel())
    return metric


def load_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root='../data'
                                                    , train=True,
                                                    transform=trans,
                                                    )
    mnist_test = torchvision.datasets.FashionMNIST(root='../data',
                                                   train=False,
                                                   transform=trans,
                                                   )
    return data.DataLoader(mnist_train, batch_size, shuffle=True, ) \
        , data.DataLoader(mnist_test, batch_size, False)


def softMax_self():
    def softMax(X):
        X_exp = torch.exp(X)
        partition = torch.sum(X_exp, dim=1, keepdim=True)
        return X_exp / partition

    def net(X):
        return torch.matmul(X.reshape((-1, W.shape[0])), W) + b

    def sgd(params, lr, batch_size):
        with torch.no_grad():
            for param in params:
                param -= lr * param.grad / batch_size
                param.grad.zero_()
            # 这样下一次计算梯度不会和上一次相关
        pass

    def cross_entropy(y_hat, y):
        return -torch.log(y_hat[range(len(y_hat)), y])

    loss = cross_entropy
    optimizer = sgd
    trainData, testData = load_fashion_mnist(batch_size)

    for epoch in range(epoch_num):
        metric = train_epoch(net, trainData, loss, optimizer)
        # print("epoch{},总损失{},总正确数{},总学习数{}".format(epoch, metric[0], metric[1], metric[2]))
        print("*" * 20)
        print("loss ", metric[0] / metric[2], "trainAccuracy",
              metric[1] / metric[2])
        print("testAccuracy", evaluate_accuracy(net, testData))


def softMax_torch():
    trainIter, testIter = load_fashion_mnist(batch_size)
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, out_put_size))

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, 0, 1)

    net.apply(init_weights)
    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(net.parameters(), lr=0.3)
    for epoch in range(epoch_num):
        metric = train_epoch(net, trainIter, loss, trainer)
        # print("epoch{},总损失{},总正确数{},总学习数{}".format(epoch, metric[0], metric[1], metric[2]))
        print("*" * 20)
        print("loss ", metric[0] / metric[2], "trainAccuracy",
              metric[1] / metric[2])
        print("testAccuracy", evaluate_accuracy(net, testIter))


# 相当于使用该函数初始化net


if __name__ == "__main__":
    batch_size = 256
    out_put_size = 10
    epoch_num = 256
    lr = 0.8
    num_inputs = 784
    W = torch.normal(0, 1, (num_inputs, out_put_size), requires_grad=True)
    b = torch.randn(out_put_size, requires_grad=True)
    softMax_torch()
    # softMax_self()
