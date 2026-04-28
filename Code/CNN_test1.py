import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
from datetime import datetime

# ===== DIAGNOSTICS =====
print(f'Python: {sys.executable}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Quick GPU benchmark
a = torch.randn(5000, 5000, device=device)
b = torch.randn(5000, 5000, device=device)
torch.matmul(a, b)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(10):
    c = torch.matmul(a, b)
torch.cuda.synchronize()
print(f'GPU benchmark: {time.time() - t0:.2f}s (should be under 2s)')
print('-' * 40)

# ===== DATA =====
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class_names = ['Shake','Tap','Spin']

# ===== MODEL =====
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 24, 5)
        self.fc1 = nn.Linear(24*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = NeuralNet().to(device)
print(f'Model on: {next(net.parameters()).device}')

loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# ===== TRAINING =====
for epoch in range(5):
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

filename = f'trained_net_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
torch.save(net.state_dict(), filename)
print(f'Saved to {filename}')

# ===== EVALUATION =====
net.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')