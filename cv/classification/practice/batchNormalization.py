import torch
import torchvision
from torchvision import datasets
from torch import nn
import pandas as pd
import numpy as np
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
import torch.functional as F


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(1, 256, (3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d((3, 3)),
            nn.Conv2d(256, 512, (3, 3), (2, 2), (1, 1)),
            nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d((3, 3)),
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, 10)
        )
