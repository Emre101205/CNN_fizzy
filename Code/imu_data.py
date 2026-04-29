"""
IMUData - a dataset class for IMU gesture recordings.

Use it just like torchvision.datasets.CIFAR10:

    train_data = IMUData(root='./recordings', train=True)
    test_data  = IMUData(root='./recordings', train=False)
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
"""
import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


CLASS_NAMES = ['Tap', 'Shake', 'UpDown', 'Idle']
CHANNEL_COLUMNS = ['Acc X', 'Acc Y', 'Acc Z', 'Gyro X', 'Gyro Y', 'Gyro Z']

TARGET_RATE = 100      # Hz - resample everything to this
WINDOW_SIZE = 128      # samples per window
STRIDE = 32            # how far to slide between windows
TRIM_SECONDS = 1.0     # drop first and last second (fumble buffer)


class IMUData(Dataset):
    def __init__(self, root, train=True, train_fraction=0.8):
        # Load all CSVs and turn them into windows + labels
        windows, labels = self._load_all(root)
        
        # Normalize the data
        mean = windows.mean(axis=(0, 2), keepdims=True)
        std = windows.std(axis=(0, 2), keepdims=True) + 1e-8
        windows = (windows - mean) / std
        
        # Split into train and test (same seed every time for reproducibility)
        rng = np.random.default_rng(seed=42)
        indices = rng.permutation(len(windows))
        split_at = int(train_fraction * len(windows))
        
        if train:
            keep = indices[:split_at]
        else:
            keep = indices[split_at:]
        
        self.windows = torch.tensor(windows[keep], dtype=torch.float32)
        self.labels = torch.tensor(labels[keep], dtype=torch.long)
        
        # Save normalization stats so you can apply them on the ESP32
        self.mean = mean.squeeze()
        self.std = std.squeeze()
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]
    
    def _load_all(self, root):
        all_windows = []
        all_labels = []
        
        for filepath in sorted(glob.glob(os.path.join(root, '*.csv'))):
            class_name = os.path.basename(filepath).split('_')[0]
            if class_name not in CLASS_NAMES:
                continue
            
            label = CLASS_NAMES.index(class_name)
            data = self._read_csv(filepath)
            data = self._trim(data)
            for w in self._make_windows(data):
                all_windows.append(w)
                all_labels.append(label)
        
        return np.array(all_windows, dtype=np.float32), np.array(all_labels, dtype=np.int64)
    
    def _read_csv(self, filepath):
        """Read CSV and resample onto a fixed time grid at TARGET_RATE."""
        df = pd.read_csv(filepath)
        times = df['timestamp'].values
        
        n = int((times[-1] - times[0]) * TARGET_RATE)
        target_times = np.linspace(times[0], times[-1], n)
        
        channels = []
        for col in CHANNEL_COLUMNS:
            values = np.nan_to_num(df[col].values.astype(np.float32), nan=0.0)
            channels.append(np.interp(target_times, times, values))
        
        return np.array(channels, dtype=np.float32)
    
    def _trim(self, data):
        n = int(TRIM_SECONDS * TARGET_RATE)
        return data[:, n:-n]
    
    def _make_windows(self, data):
        windows = []
        start = 0
        while start + WINDOW_SIZE <= data.shape[1]:
            windows.append(data[:, start:start+WINDOW_SIZE])
            start += STRIDE
        return windows
