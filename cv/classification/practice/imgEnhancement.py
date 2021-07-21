import cv2
import torch
import torchvision
from matplotlib import pyplot as plt
from torchvision import datasets
from torch import nn
import pandas as pd
import numpy as np
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
from PIL import Image

img = Image.open('../../testImg.jpg')


def apply(img, aug, numRows, numCols, scale=1.5):
    fig, ax = plt.subplots(numRows, numCols)
    for i in range(numRows):
        for j in range(numCols):
            imgNew = aug(img)
            ax[i][j].imshow(imgNew)
    pass


# apply(img, torchvision.transforms.RandomResizedCrop((200, 200)), 2, 2)
aug = torchvision.transforms.Compose([
    torchvision.transforms.CenterCrop((200, 200)),
    torchvision.transforms.RandomResizedCrop((200, 200)),
    torchvision.transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
    torchvision.transforms.RandomHorizontalFlip()])
apply(img, aug, 4, 4)
plt.show()
