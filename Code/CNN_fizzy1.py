import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from datetime import datetime
from torch.utils.data import random_split

from data import GestureDataset

# ===== DIAGNOSTICS =====
print(f'Python: {sys.executable}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===== DATA =====
full_dataset = GestureDataset()
n_test = int(0.2 * len(full_dataset))
n_train = len(full_dataset) - n_test
train_data, test_data = random_split(full_dataset, [n_train, n_test])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

class_names = ['idle', 'shake', 'tap', 'spin']

# ===== MODEL =====
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
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 4)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 4)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.avg_pool1d(x, 8)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

net = IMUNet().to(device)
print(f'Model on: {next(net.parameters()).device}')

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# ===== TRAINING =====
for epoch in range(500):
    t0 = time.time()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch}: loss={running_loss/len(train_loader):.4f}, time={time.time()-t0:.1f}s')

filename = f'trained_imu_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
torch.save(net.state_dict(), filename)
print(f'Saved to {filename}')

# ===== EVALUATION =====
net.eval()
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