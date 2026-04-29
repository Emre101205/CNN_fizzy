import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
from datetime import datetime
import numpy as np






class IMUNet(nn.Module):
    def __init__(self, in_channels=6, n_classes=4, window_length=128):
        super().__init__()