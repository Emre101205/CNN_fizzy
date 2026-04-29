import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from datetime import datetime

# ===== DIAGNOSTICS =====
print(f'Python: {sys.executable}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===== DATA =====
# train_loader and test_loader come from your data pipeline (CSVs)

class_names = ['tap', 'shake', 'spin', 'idle']

# ===== MODEL =====
class IMUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Block 1: 128 -> 32 timesteps
        self.conv1 = nn.Conv1d(6, 16, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        # Block 2: 32 -> 8 timesteps
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        # Block 3: 8 -> 1 timestep
        self.conv3 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(32)

        self.fc = nn.Linear(32, 10)

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
for epoch in range(50):
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