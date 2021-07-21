import os
import pandas as pd
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


def readBananasData(is_train=True):
    dataDir = d2l.download_extract('banana-detection')
    csvFName = os.path.join(dataDir, 'bananasTrain' if is_train else
    '')


def drawBox(img, box):
    numObjects = box.shape[1]
    fig, ax = plt.subplots(8, int(len(box) / 8))
    ax = ax.flatten()
    for i in range(len(box)):
        for j in range(numObjects):
            rec = pat.Rectangle((box[i][j][1], box[i][j][2]), box[i][j][3] - box[i][j][1], box[i][j][4] - box[i][j][2])
            ax[i].add_patch(rec)
        channel, width, height = img[i].shape
        imgTempShape = np.reshape(img[i], (width, height, channel)) / 255
        ax[i].imshow(imgTempShape)
    plt.show()


d2l.DATA_HUB['banana-detection'] = (
    d2l.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72')


def read_data_bananas(is_train=True):
    """读取香蕉检测数据集中的图像和标签。"""
    data_dir = '/data/banana-detection\\'
    csv_fname = os.path.join(data_dir,
                             'bananas_train' if is_train else 'bananas_val',
                             'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(
            torchvision.io.read_image(
                os.path.join(data_dir,
                             'bananas_train' if is_train else 'bananas_val',
                             'images', f'{img_name}')))
        # Here `target` contains (class, upper-left x, upper-left y,
        # lower-right x, lower-right y), where all the images have the same
        # banana class (index 0)
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256


class BananasDataset(torch.utils.data.Dataset):
    """一个用于加载香蕉检测数据集的自定义数据集。"""

    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (
            f' training examples' if is_train else f' validation examples'))

    def __getitem__(self, idx):
        return self.features[idx].float(), self.labels[idx]

    def __len__(self):
        return len(self.features)


def load_data_bananas(batch_size):
    train_iter = torch.utils.data.DataLoader(BananasDataset(True), batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(BananasDataset(False), batch_size)
    return train_iter, test_iter


batchSize, edgeSize = 32, 256
trainIter, testIter = load_data_bananas(batch_size=batchSize)
batch = next(iter(trainIter))
# batch是一个list，有两个元素，前者是图片，后者是标签
# 使用next(iter(要迭代的对象))获得第一个对象
imgs = (batch[0][0:10].permute(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][0:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * 256], colors=['w'])
plt.show()
plt.pause(10)
