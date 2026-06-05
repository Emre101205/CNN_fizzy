"""
data_test.py — Load gesture CSV recordings from the Test_01..04 folders
and prepare them for CNN testing.

Same as data.py, but with ONE difference that matters:
the Test folders store CSVs in nested sub-folders, e.g.

    Test_01/IDLE/Idle/Idle1.csv
    Test_01/RANDOM/Kick/Kick3.csv

The normal data.py uses a non-recursive glob ('*.csv'), which only looks
at the top level of the folder and therefore finds ZERO files in here.
This file uses a RECURSIVE glob ('**/*.csv') so it finds every CSV no
matter how deep it is nested.
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

# Which test folders to load. All of these are combined into one dataset.
# Comment out any you don't want, or trim the list to a single folder.
TEST_FOLDERS = ['Test_01', 'Test_02', 'Test_03', 'Test_04']

# Which half of each test folder to use: 'IDLE' or 'RANDOM' (never both).
# Each Test folder has two top sub-folders, e.g. Test_03 has IDLE03 and
# RANDOM03. We match by prefix, so 'IDLE' picks IDLE/IDLE02/IDLE03/IDLE04
# and 'RANDOM' picks RANDOM/RANDOM02/...
SUBSET = 'RANDOM'

# Folder where your CSV files live (next to this script).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Map gesture names to numbers. Must match the model you are testing.
# (Same labels as v12, v13, 16, 20, 21 — the 6-class model.)
LABELS = {'idle': 0, 'shake': 1, 'tap': 2, 'drop': 3, 'lift': 4, 'kick': 5}

# Which columns from the CSV to use as inputs.
# FEATURE_COLUMNS = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
FEATURE_COLUMNS = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'motor_input']
# Window length (samples) and how far we slide each step.
WINDOW_SIZE = 64
WINDOW_STRIDE = 32


# =============================================================
# STEP 2: Helper function — load ONE csv file
# =============================================================

def load_one_csv(file_path):
    """Read one CSV and return just the feature columns as (rows, 6)."""
    df = pd.read_csv(file_path)
    df = df[FEATURE_COLUMNS]
    return df.to_numpy(dtype=np.float32)


# =============================================================
# STEP 3: Helper function — cut a recording into windows
# =============================================================

def cut_into_windows(recording, window_size, stride):
    """Cut one recording (time x features) into overlapping windows."""
    windows = []
    start = 0
    while start + window_size <= len(recording):
        windows.append(recording[start: start + window_size])
        start += stride

    if len(windows) == 0:
        return np.empty((0, window_size, len(FEATURE_COLUMNS)), dtype=np.float32)
    return np.stack(windows)


# =============================================================
# STEP 4: The Dataset class
# =============================================================

class GestureDataset(Dataset):
    """PyTorch Dataset holding all gesture windows + labels from a Test folder."""

    def __init__(self):
        all_windows = []
        all_labels = []

        # Collect CSVs from every chosen test folder and combine them.
        all_csv_paths = []
        for folder in TEST_FOLDERS:
            recordings_dir = os.path.join(SCRIPT_DIR, folder)

            # recursive=True + '**' walks every sub-folder, so nested CSVs
            # like Test_01/IDLE/Idle/Idle1.csv are found.
            paths = glob.glob(
                os.path.join(recordings_dir, '**', '*.csv'), recursive=True
            )

            # Keep only the chosen half (IDLE* or RANDOM*). The top sub-folder
            # is the first path part after the test folder; match it by prefix.
            kept = []
            for p in paths:
                rel = os.path.relpath(p, recordings_dir)
                top = rel.split(os.sep)[0]        # e.g. 'IDLE03' or 'RANDOM03'
                if top.upper().startswith(SUBSET.upper()):
                    kept.append(p)
            print(f'  {folder}: {len(kept)} CSV files in "{SUBSET}" half')
            all_csv_paths.extend(kept)

        print(f'Using SUBSET="{SUBSET}" across {TEST_FOLDERS}: {len(all_csv_paths)} CSV files total')

        # Label each file by its GESTURE sub-folder (the folder directly
        # containing the csv), e.g. .../Idle/Idle1.csv -> 'idle'.
        # We do NOT match against the full path, because the top folder is
        # literally named 'IDLE' and would tag everything as idle.
        for gesture_name, label_number in LABELS.items():
            file_paths = sorted([
                p for p in all_csv_paths
                if os.path.basename(os.path.dirname(p)).lower() == gesture_name.lower()
            ])

            print(f'Found {len(file_paths)} files for "{gesture_name}"')

            for path in file_paths:
                recording = load_one_csv(path)
                windows = cut_into_windows(recording, WINDOW_SIZE, WINDOW_STRIDE)
                labels = np.full(len(windows), label_number, dtype=np.int64)
                all_windows.append(windows)
                all_labels.append(labels)

        X = np.concatenate(all_windows, axis=0)
        y = np.concatenate(all_labels, axis=0)

        # (batch, time, channels) -> (batch, channels, time)
        X = X.transpose(0, 2, 1)

        # Normalize per channel (mean 0, std 1).
        mean = X.mean(axis=(0, 2), keepdims=True)
        std = X.std(axis=(0, 2), keepdims=True) + 1e-8
        X = (X - mean) / std

        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


# =============================================================
# STEP 5: Run this file directly to test the loading
# =============================================================

if __name__ == '__main__':
    dataset = GestureDataset()

    print()
    print(f'Total windows: {len(dataset)}')
    print(f'Input shape:   {dataset.X.shape}   (windows, channels, timesteps)')
    print(f'Label shape:   {dataset.y.shape}')
    print(f'Examples per class: {torch.bincount(dataset.y).tolist()}')

    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    batch_X, batch_y = next(iter(loader))
    print()
    print(f'One batch of inputs: {batch_X.shape}')
    print(f'One batch of labels: {batch_y.shape}')
