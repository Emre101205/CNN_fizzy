"""
data.py — Load gesture CSV recordings and prepare them for CNN training.

What this script does, in plain English:
1. Finds every CSV file in the 'recordings' folder.
2. Reads each one into a table.
3. Keeps only the columns that have real data.
4. Cuts each recording into fixed-size "windows" (short clips).
5. Labels each window based on the filename (Idle, Shake, Tap, UpDown).
6. Packages everything into a PyTorch Dataset that the CNN can train on.
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# =============================================================
# STEP 1: Settings you might want to change
# =============================================================

# Folder where your CSV files live (next to this script, in 'recordings/')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RECORDINGS_DIR = os.path.join(SCRIPT_DIR, 'recordings_v3')

# Map gesture names to numbers. The CNN works with numbers, not text.
# Idle = 0, Shake = 1, Tap = 2, UpDown = 3
LABELS = {'IDLE' : 0, 'SHAKE': 1, 'TAP': 2, 'SPIN': 3}

# Which columns from the CSV to actually use as inputs.
# We skip 'timestamp' (not useful as a feature) and the Gyro/Magnitude
# columns (they were full of NaN / missing values in your data).
FEATURE_COLUMNS = ['Roll', 'Pitch', 'Yaw', 'Acc X', 'Acc Y', 'Acc Z', 'Gyro X', 'Gyro Y', 'Gyro Z']

# How long each "window" is, in samples.
# Your data is recorded at ~100 samples per second,
# so 200 samples ≈ 2 seconds of motion.
WINDOW_SIZE = 64

# How far we slide the window forward each time.
# 100 means each new window starts halfway through the previous one
# (50% overlap). This gives us more training examples.
WINDOW_STRIDE = 32


# =============================================================
# STEP 2: Helper function — load ONE csv file
# =============================================================

def load_one_csv(file_path):
    """
    Read one CSV file and return just the feature columns
    as a NumPy array of shape (number_of_rows, 6).
    """
    # pandas reads the CSV into a table (DataFrame)
    df = pd.read_csv(file_path)

    # Pick only the 6 columns we care about
    df = df[FEATURE_COLUMNS]

    # Convert to a NumPy array of 32-bit floats (what PyTorch likes)
    return df.to_numpy(dtype=np.float32)


# =============================================================
# STEP 3: Helper function — cut a recording into windows
# =============================================================

def cut_into_windows(recording, window_size, stride):
    """
    Take one full recording (shape: time × features) and cut it
    into many overlapping windows (shape: window_size × features).

    Example: a 1000-row recording with window_size=200, stride=100
    gives us 9 windows.
    """
    windows = []
    start = 0

    # Slide a window across the recording
    while start + window_size <= len(recording):
        window = recording[start : start + window_size]
        windows.append(window)
        start += stride

    # If the recording was too short, return an empty array
    if len(windows) == 0:
        return np.empty((0, window_size, len(FEATURE_COLUMNS)), dtype=np.float32)

    # Stack all windows into one big array
    return np.stack(windows)


# =============================================================
# STEP 4: The Dataset class — what PyTorch will use for training
# =============================================================

class GestureDataset(Dataset):
    """
    A PyTorch Dataset that holds all our gesture windows + their labels.

    Once built, you can ask it for example #5 with: dataset[5]
    and you'll get back (window_tensor, label).
    """

    def __init__(self):
        # We'll collect all windows and labels here, then combine at the end.
        all_windows = []
        all_labels = []

        # Loop through each gesture type (Idle, Shake, Tap, UpDown)
        for gesture_name, label_number in LABELS.items():

            # Find every CSV that starts with this gesture name
            # e.g. 'Idle_001.csv', 'Idle_002.csv', ...
            pattern = os.path.join(RECORDINGS_DIR, f'{gesture_name}_*.csv')
            file_paths = sorted(glob.glob(pattern))

            print(f'Found {len(file_paths)} files for "{gesture_name}"')

            # Process each file
            for path in file_paths:
                recording = load_one_csv(path)
                windows = cut_into_windows(recording, WINDOW_SIZE, WINDOW_STRIDE)

                # Make a label for each window (all the same gesture)
                labels = np.full(len(windows), label_number, dtype=np.int64)

                all_windows.append(windows)
                all_labels.append(labels)

        # Combine everything into two big arrays
        # X shape: (total_windows, window_size, num_features) e.g. (N, 200, 6)
        # y shape: (total_windows,)
        X = np.concatenate(all_windows, axis=0)
        y = np.concatenate(all_labels, axis=0)

        # PyTorch's Conv1d expects shape (batch, channels, time)
        # Right now we have (batch, time, channels), so we swap the last two axes.
        X = X.transpose(0, 2, 1)   # now shape: (N, 6, 200)

        # Normalize: subtract mean, divide by std deviation, per channel.
        # This makes training much more stable — all features end up
        # roughly in the same range (mean 0, std 1).
        mean = X.mean(axis=(0, 2), keepdims=True)
        std = X.std(axis=(0, 2), keepdims=True) + 1e-8   # +tiny number to avoid /0
        X = (X - mean) / std

        # Convert NumPy arrays to PyTorch tensors and store them
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        # Tells PyTorch how many examples we have
        return len(self.y)

    def __getitem__(self, index):
        # Tells PyTorch how to fetch example number `index`
        return self.X[index], self.y[index]


# =============================================================
# STEP 5: Run this file directly to test it
# =============================================================

if __name__ == '__main__':
    # Build the dataset
    dataset = GestureDataset()

    print()
    print(f'Total windows: {len(dataset)}')
    print(f'Input shape:   {dataset.X.shape}   (windows, channels, timesteps)')
    print(f'Label shape:   {dataset.y.shape}')
    print(f'Examples per class: {torch.bincount(dataset.y).tolist()}')
    print(f'   (in order: Idle, Shake, Tap, UpDown)')

    # Wrap in a DataLoader — this is what you'll loop over during training.
    # batch_size=32 means it gives you 32 windows at a time.
    # shuffle=True means it mixes them up each epoch (important for training).
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Grab one batch just to confirm it works
    batch_X, batch_y = next(iter(loader))
    print()
    print(f'One batch of inputs: {batch_X.shape}')
    print(f'One batch of labels: {batch_y.shape}')