import torch
import numpy as np
import matplotlib.pyplot as plt

y = torch.cos(torch.arange(0, int(50 * np.pi))) + torch.rand(int(50 * np.pi))

