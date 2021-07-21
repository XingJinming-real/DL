import random

import cv2
from torchvision import transforms
import torchvision
import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt
import numpy as np
from torch.utils import data
from torch import nn


def upSample(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    temp = torch.tensor((1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor))
    res = torch.zeros((21, 3, 64, 64))
    for i in range(21):
        for j in range(3):
            res[i, j, :, :] = temp
    return res


numClass = 21
net = torchvision.models.resnet18(pretrained=True)
net = nn.Sequential(*list(net.children())[:-2])
# 这样就去掉了最后两层
# 我们知道，resNet系系列将图形缩小了32倍，我们转置卷积就要回复原图像根据公式就要stride=32
net.add_module('final_conv', nn.Conv2d(512, numClass, kernel_size=(1, 1)))
# 此处直接将通道降到21，是方便计算，但精度会有所下降
convT = nn.ConvTranspose2d(numClass, 3, (64, 64), stride=(32, 32), padding=(16, 16))
convT.weight.data = upSample(64)
net.add_module('transpose_conv', convT)
img = torchvision.io.read_image('../../testImg.jpg').unsqueeze(0).float()
imgNew = net(img)
print(imgNew.shape)
cv2.imshow('fff', imgNew.reshape((1088, 1920, 3)))
