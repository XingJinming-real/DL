import torchvision
from torch import nn
import torch
from torchvision import transforms
from d2l import torch as d2l
from torch.utils import data
import numpy as np
from torchsummary import summary
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as pat

"""
imgNet 使用了normalize，具体3通道均值为【0.485，0.456，0.406】方差为【0.299，0.224，0.225】
故我们在使用imgNet上预训练的模型时要在transform中加入Normalize，如main中所示
"""


def main():
    img = Image.open('../../testImg.jpg')
    fig, ax = plt.subplots(1, 1)
    ax.imshow(img)
    rec = pat.Rectangle((200, 200), 200, 200, fill=False)
    ax.add_patch(rec)
    ax.imshow(img)
    plt.show()
    normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.299, 0.224, 0.255])
    trans = torchvision.transforms.Compose(
        [transforms.RandomResizedCrop((224, 224)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    dataset = torchvision.datasets.CIFAR10(root='../../data', transform=trans, download=False)
    trainIter = data.DataLoader(dataset, batch_size=64, shuffle=True)

    # 首先导入预训练模型
    pretrainedNet = torchvision.models.resnet18(True)
    # print(pretrainedNet.state_dict())
    # summary(pretrainedNet, (3, 224, 224), device='cpu')
    pretrainedNet.fc = nn.Linear(pretrainedNet.fc.in_features, 10)
    # 将与训练网络的全连接层fully connected替换为自己定义的全连接层
    nn.init.xavier_uniform_(pretrainedNet.fc.weight)
    # 并初始化
    flag = input('请输入flag')
    if flag:
        param_ls = [
            param for name, param in pretrainedNet.named_parameters()
            if name not in ['fc.weight', 'fc.bias']]
        trainer = torch.optim.SGD([{'params': param_ls},
                                   {'params': pretrainedNet.fc.parameters(), 'lr': 0.1}
                                   ], lr=0.01)
        # pretrainedNet.parameters()中含有weight和bias
    else:
        trainer = torch.optim.SGD(pretrainedNet.parameters(), lr=0.001)
    pretrainedNet.to('cuda')
    loss = nn.CrossEntropyLoss()
    epochNum = 10
    for epoch in range(epochNum):
        lSum = 0
        for X, y in trainIter:
            X, y = X.to('cuda'), y.to('cuda')
            trainer.zero_grad()
            l = loss(pretrainedNet(X), y)
            l.backward()
            trainer.step()
            lSum += l * len(y)
        print(f'loss={lSum}')


if __name__ == '__main__':
    main()
