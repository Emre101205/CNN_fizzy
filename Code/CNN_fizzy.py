import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from datetime import datetime
from torch.utils.data import random_split

INPUT_AMOUNT = 6
CLASSES_AMOUNT = 4


# ===== MODEL =====
class IMUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(INPUT_AMOUNT, 16, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(32)

        self.fc = nn.Linear(32, CLASSES_AMOUNT)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 4)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 4)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.avg_pool1d(x, 4)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x



