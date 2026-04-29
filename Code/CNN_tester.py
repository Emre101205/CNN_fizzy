import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import GestureDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===== MODEL (same as in train.py) =====
class IMUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(6, 16, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc = nn.Linear(32, 4)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))); x = F.max_pool1d(x, 4)
        x = F.relu(self.bn2(self.conv2(x))); x = F.max_pool1d(x, 4)
        x = F.relu(self.bn3(self.conv3(x))); x = F.avg_pool1d(x, 8)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ===== LOAD TRAINED MODEL =====
net = IMUNet().to(device)
net.load_state_dict(torch.load('trained_imu_20260429_155155.pth', map_location=device))
net.eval()

# ===== LOAD NEW DATA =====
test_dataset = GestureDataset()   # points at whatever folder data.py is set to
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class_names = ['idle', 'shake', 'tap', 'updown']

# ===== EVALUATE =====
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')