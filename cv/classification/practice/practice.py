import random
from torchvision import transforms
import torchvision
import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt
import numpy as np
from torch.utils import data
from torch import nn

"""手动实现单层线性回归"""
"""因为该优化函数为凸函数，故一定有最优解"""


def synthetic_data(w, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w)
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


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


true_w = torch.tensor([2, -3.4, 4.2])
features, labels = synthetic_data(true_w, 100)
batch_size = 256


def dataYield(batch_size, xSub, ySub):
    num_examples = len(xSub)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        choose_indices = indices[i:min(i + batch_size, num_examples)]
        yield xSub[choose_indices], ySub[choose_indices]


def main_self():
    batch_size = 256
    inputSize = 784
    outputSize = 10

    W = torch.normal(0, 0.01, size=(inputSize, outputSize), requires_grad=True)
    b = torch.zeros(outputSize, requires_grad=True)

    def softMax(x):
        x_exp = torch.exp(x)
        partition = x_exp.sum(dim=1, keepdim=True)
        return x_exp / partition

    def linReg(X, w):
        return softMax(torch.matmul(X.reshape((-1, w.shape[0])), w) + b)

    def cross_entropy(y_hat, y):
        return -torch.log(y_hat[range(batch_size), y])

    def squared_loss(y_hat, y):
        return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

    def sgd(params, lr, batch_size):
        with torch.no_grad():
            for param in params:
                param -= lr * param.grad / batch_size
                param.grad.zero_()
                # 这样下一次计算梯度不会和上一次相关

    lr = 0.03
    num_epoch = 10
    net = linReg
    # loss = squared_loss
    loss = cross_entropy
    trainData, testData = load_fashion_mnist(batch_size)
    for epoch in range(num_epoch):
        # for X, y in dataYield(batch_size, features, labels):
        for X, y in trainData:
            l = loss(net(X, W), y)
            l.sum().backward()
            sgd([W, b], lr, batch_size)
        # with torch.no_grad():
        #     train_l = loss(net(testData, W), labels)
        #     print("epoch{}:,loss={}".format(epoch, train_l.mean()))
    print(W)


"""掉包实现单层线性回归"""


def main_torch():
    def load_array(data_arrays, batch_size, is_train=True):
        dataset = data.TensorDataset(*data_arrays)
        return data.DataLoader(dataset, batch_size, shuffle=is_train)

    data_iter = load_array((features, labels), batch_size)
    # next(iter(data_iter))
    # 注意使用torch自带的TensorDataset来自动处理数据，要用*运算符来分开features和labels
    # 然后使用iter和next来变成python的iter，next(iter(data_iter))
    # 最后使用
    # for X,y in data_iter:
    #     此时X为features，y为labels
    net = nn.Sequential(nn.Linear(3, 1))
    # 定义一个全连接层来模拟线性回归，放在一个sequential容器里面方便后面加入不同的层
    # net[0].weight.data.normal_(0, 0.01)
    # 可以手动设也可以不用
    # 对net中的第零层其weight设置为正态分布
    net[0].bias.data.fill_(0)
    loss = nn.MSELoss()
    # 使用平方范数
    optimizer = torch.optim.SGD(net.parameters(), lr=0.03)
    # 定义优化器，SGD为随机梯度下降
    num_epoch = 10
    for epoch in range(num_epoch):
        for X, y in data_iter:
            l = loss(net(X), y)
            optimizer.zero_grad()
            # 梯度清零，因为torch里面设计就是这样，梯度会累加
            # 这样是如对于一个非常大的batch如4000，把它分4个1000然后分别
            # 计算grad然后再加起来，这样可以尽可能大的处理大batch
            l.backward()
            # torch已经进行sum求和了
            # 一般我们会对标量求导数而不对向量求导数
            optimizer.step()
            # 进行一次模型更新
        l = loss(net(features), labels)
        print("epoch", epoch, ":", l)


if __name__ == "__main__":
    main_self()
    # main_torch()