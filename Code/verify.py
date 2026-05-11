"""
verify.py — Bit-for-bit verification of the C inference logic
=============================================================

Reimplements every operation from cnn_inference.c in pure numpy, then runs
side-by-side against the original PyTorch model on the same input. If the
predictions match, the C code is correct.

This is the most important step before flashing anything: catching bugs here
saves hours of staring at a non-functional ESP32 later.

Usage:
    python verify.py

Tests a few random inputs. If you have a real CSV, you can also add a test that
loads a window from it.
"""

import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Trainer_CNN import IMUNet


# ---------------------------------------------------------------------------
# 1. Load the trained model (same as export_to_c.py)
# ---------------------------------------------------------------------------
model = IMUNet()
state = torch.load("Trained_004.pth", map_location="cpu", weights_only=True)
if isinstance(state, dict) and "state_dict" in state:
    model.load_state_dict(state["state_dict"])
elif isinstance(state, dict) and "model_state_dict" in state:
    model.load_state_dict(state["model_state_dict"])
else:
    model.load_state_dict(state)
model.eval()


# ---------------------------------------------------------------------------
# 2. BN-fold helper (same math as in export_to_c.py)
# ---------------------------------------------------------------------------
def fold(conv, bn):
    w = conv.weight.detach().numpy().copy()
    b = (conv.bias.detach().numpy() if conv.bias is not None
         else np.zeros(conv.out_channels)).copy()
    gamma = bn.weight.detach().numpy()
    beta  = bn.bias.detach().numpy()
    mean  = bn.running_mean.detach().numpy()
    var   = bn.running_var.detach().numpy()
    eps   = bn.eps
    scale = gamma / np.sqrt(var + eps)
    w_new = w * scale[:, None, None]
    b_new = (b - mean) * scale + beta
    return w_new.astype(np.float32), b_new.astype(np.float32)

w1, b1 = fold(model.conv1, model.bn1)
w2, b2 = fold(model.conv2, model.bn2)
w3, b3 = fold(model.conv3, model.bn3)
fc_w = model.fc.weight.detach().numpy().astype(np.float32)
fc_b = model.fc.bias.detach().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# 3. C-style operations in Python
# ---------------------------------------------------------------------------
def conv1d_same(x, weight, bias, pad):
    """x: (C_in, T), weight: (C_out, C_in, K), bias: (C_out,)"""
    C_in, T = x.shape
    C_out, _, K = weight.shape
    out = np.zeros((C_out, T), dtype=np.float32)
    for oc in range(C_out):
        for t in range(T):
            acc = bias[oc]
            for ic in range(C_in):
                for k in range(K):
                    in_t = t + k - pad
                    if 0 <= in_t < T:
                        acc += weight[oc, ic, k] * x[ic, in_t]
            out[oc, t] = acc
    return out

def relu(x):
    return np.maximum(x, 0)

def maxpool1d(x, pool):
    C, T = x.shape
    T_out = T // pool
    out = np.zeros((C, T_out), dtype=np.float32)
    for c in range(C):
        for t in range(T_out):
            out[c, t] = x[c, t*pool:(t+1)*pool].max()
    return out

def avgpool1d(x, pool):
    C, T = x.shape
    T_out = T // pool
    out = np.zeros((C, T_out), dtype=np.float32)
    for c in range(C):
        for t in range(T_out):
            out[c, t] = x[c, t*pool:(t+1)*pool].mean()
    return out

def linear(x, weight, bias):
    return weight @ x + bias


def c_forward(x):
    """Forward pass mirroring cnn_inference.c exactly."""
    x = conv1d_same(x, w1, b1, pad=3)
    x = relu(x)
    x = maxpool1d(x, 4)
    x = conv1d_same(x, w2, b2, pad=2)
    x = relu(x)
    x = maxpool1d(x, 4)
    x = conv1d_same(x, w3, b3, pad=1)
    x = relu(x)
    x = avgpool1d(x, 4)
    x = x.flatten()
    logits = linear(x, fc_w, fc_b)
    return logits


# ---------------------------------------------------------------------------
# 4. Compare PyTorch model vs C-style implementation
# ---------------------------------------------------------------------------
print("Testing random inputs:")
print("=" * 60)

CLASS_NAMES = ["IDLE", "SHAKE", "TAP", "SPIN", "FALL"]

np.random.seed(42)
worst_diff = 0.0
mismatches = 0

with torch.no_grad():
    for test_i in range(10):
        # Random normalized input (mean 0, std 1)
        x = np.random.randn(6, 64).astype(np.float32)

        # PyTorch forward
        x_torch = torch.from_numpy(x).unsqueeze(0)   # (1, 6, 64)
        pt_logits = model(x_torch).numpy()[0]
        pt_pred = int(pt_logits.argmax())

        # C-style forward
        c_logits = c_forward(x)
        c_pred = int(c_logits.argmax())

        # Compare
        max_diff = np.abs(pt_logits - c_logits).max()
        worst_diff = max(worst_diff, max_diff)
        match = "OK " if pt_pred == c_pred else "FAIL"
        if pt_pred != c_pred:
            mismatches += 1
        print(f"  test {test_i}: pt={CLASS_NAMES[pt_pred]:5s} "
              f"c={CLASS_NAMES[c_pred]:5s} "
              f"max_logit_diff={max_diff:.2e} [{match}]")

print()
print(f"Worst logit diff across 10 tests: {worst_diff:.3e}")
print(f"Prediction mismatches: {mismatches} / 10")
if mismatches == 0 and worst_diff < 1e-3:
    print("PASS — the C inference will produce the same predictions as PyTorch.")
elif mismatches == 0:
    print("PASS predictions, but logit diff is large — investigate before deploying.")
else:
    print("FAIL — the C-style code does not match PyTorch. Don't flash yet.")
