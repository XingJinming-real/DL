import torch
import torchvision
import numpy as np
from d2l import torch as d2l
from torch import nn

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
num_inputs, num_outputs, num_hidden = 784, 10, 256


def test(net, params):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params, lr)
    metric = d2l.Accumulator(3)
    for epoch in range(epoch_num):
        for X, y in train_iter:
            optimizer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            metric.add(float(l) * len(y), d2l.accuracy(y_hat, y), y.detach().numel())
        print("loss: ", metric[0] / metric[2], "accu: ", metric[1] / metric[2])
        print("testAccu:", d2l.evaluate_accuracy(net, test_iter))


def with_hand():
    W1 = nn.Parameter(
        torch.randn((num_inputs, num_hidden), requires_grad=True)
    )
    b1 = nn.Parameter(
        torch.randn(num_hidden, requires_grad=True)
    )
    W2 = nn.Parameter(
        torch.randn((num_hidden, num_outputs), requires_grad=True)

    )
    b2 = nn.Parameter(
        torch.randn(num_outputs, requires_grad=True)
    )

    def relu(X):
        a = torch.zeros_like(X)
        return torch.max(X, a)

    def net(X):
        X = X.reshape((-1, num_inputs))
        H = relu(torch.matmul(X, W1) + b1)
        return H.mm(W2) + b2

    params = [W1, b1, W2, b2]
    test(net, params)


def with_torch():
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256),
                        nn.ReLU(), nn.Linear(256, 10))

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=1)

    net.apply(init_weights)
    test(net, net.parameters())


if __name__ == "__main__":
    epoch_num = 10
    lr = 0.3
    with_torch()
