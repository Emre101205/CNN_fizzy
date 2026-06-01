import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from data import GestureDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INPUT_AMOUNT = 7
CLASSES_AMOUNT = 6

model_number = '026'

FILE_PATH  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(FILE_PATH, 'Models', f'Trained_{model_number}.pth')




# ===== MODEL (same as in train.py) =====
class IMUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(INPUT_AMOUNT, 16, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        # self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(32, CLASSES_AMOUNT) #MAX POOL 4
        # self.fc = nn.Linear(256, CLASSES_AMOUNT) #MAX POOL 2


# Met batchnorm


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        # x = F.max_pool1d(x, 2)
        x = F.max_pool1d(x,4)

        x = F.relu(self.bn2(self.conv2(x)))
        # x = F.max_pool1d(x, 2)
        x = F.max_pool1d(x,4)

        x = F.relu(self.bn3(self.conv3(x)))
        # x = F.avg_pool1d(x, 2)
        x = F.avg_pool1d(x,4)

        x = torch.flatten(x, 1)
        
        x = self.fc(x)
        return x


# ===== LOAD TRAINED MODEL =====
net = IMUNet().to(device)
net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
net.eval()

# ===== LOAD NEW DATA =====
test_dataset = GestureDataset()   # points at whatever folder data.py is set to
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class_names = ['idle', 'shake', 'tap', 'drop', 'lift', 'kick']
# class_names = ['idle',  'lift', 'kick']

# ===== EVALUATE =====
CONFIDENCE_THRESHOLD = 0.90
num_classes = CLASSES_AMOUNT
matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)

total_correct = 0
total_count = 0
conf_correct = 0
conf_count = 0
noconf_correct = 0
noconf_count = 0

net.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        # inputs = inputs[:, :INPUT_AMOUNT, :]  # trim to channels this model was trained on
        probs = F.softmax(net(inputs), dim=1)
        max_probs, predicted = torch.max(probs, dim=1)
        is_confident = max_probs >= CONFIDENCE_THRESHOLD

        correct = predicted == labels
        total_correct += correct.sum().item()
        total_count += len(labels)

        conf_mask = is_confident
        conf_correct += (correct & conf_mask).sum().item()
        conf_count += conf_mask.sum().item()

        noconf_mask = ~is_confident
        noconf_correct += (correct & noconf_mask).sum().item()
        noconf_count += noconf_mask.sum().item()

        for t, p, c in zip(labels.cpu(), predicted.cpu(), is_confident.cpu()):
            if c:
                matrix[t.item(), p.item()] += 1

unsure_count = noconf_count
print(f"\n{unsure_count} predictions marked unsure (below {CONFIDENCE_THRESHOLD})")

print(f"\nAccuracy (total):          {total_correct}/{total_count} = {100*total_correct/total_count:.1f}%")
if conf_count > 0:
    print(f"Accuracy (confident):      {conf_correct}/{conf_count} = {100*conf_correct/conf_count:.1f}%")
else:
    print("Accuracy (confident):      no confident predictions")
if noconf_count > 0:
    print(f"Accuracy (no confidence):  {noconf_correct}/{noconf_count} = {100*noconf_correct/noconf_count:.1f}%")
else:
    print("Accuracy (no confidence):  no low-confidence predictions")

print("\nConfusion Matrix (confident predictions only)")
print(f"{'':>8}" + "".join(f"{name:>8}" for name in class_names))
for i, name in enumerate(class_names):
    row = "".join(f"{matrix[i, j].item():>8}" for j in range(num_classes))
    print(f"{name:>8}{row}")