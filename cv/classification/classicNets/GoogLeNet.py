import torch
import torchvision
from torchvision import datasets
from torch import nn
import pandas as pd
import numpy as np
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
from torchsummary import summary
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F


class googleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(200, 10)
        )

    def forward(self, x):
        return self.net(x)
