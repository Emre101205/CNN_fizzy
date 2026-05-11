"""
export_to_c.py
==============

One-shot script: takes Trained_004.pth (your trained PyTorch model) and writes
out a single C header file containing the model weights, ready to drop into the
ESP-IDF project.

WHAT IT DOES
------------
1. Loads the IMUNet from CNN_fizzy.py
2. Loads the trained weights from Trained_004.pth
3. Folds each BatchNorm layer into the preceding Conv1d layer.
   (BN at inference is a linear transform. Combining it with the Conv that
    came right before gives numerically identical predictions but means the
    C code never has to implement BatchNorm itself.)
4. Loads imu_mean.npy and imu_std.npy and bakes them in as constants.
5. Writes model_weights.h.

HOW TO RUN
----------
Put this script in the same folder as:
    CNN_fizzy.py
    Trained_004.pth
    imu_mean.npy
    imu_std.npy

Then run:
    python export_to_c.py

Output:
    model_weights.h    (drop this into the C component)
"""

import numpy as np
import torch
import sys
import os

# Make sure we can import IMUNet from CNN_fizzy.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Trainer_CNN import IMUNet  # noqa


# ---------------------------------------------------------------------------
# 1. Load the trained model
# ---------------------------------------------------------------------------
PTH_PATH = "Trained_004.pth"
MEAN_PATH = "imu_mean.npy"
STD_PATH = "imu_std.npy"
OUT_PATH = "model_weights.h"

model = IMUNet()
state = torch.load(PTH_PATH, map_location="cpu", weights_only=True)

# .pth files vary in how they store weights — handle the common cases
if isinstance(state, dict) and "state_dict" in state:
    model.load_state_dict(state["state_dict"])
elif isinstance(state, dict) and "model_state_dict" in state:
    model.load_state_dict(state["model_state_dict"])
else:
    model.load_state_dict(state)

model.eval()
print(f"Loaded {PTH_PATH}")


# ---------------------------------------------------------------------------
# 2. Fold each Conv1d + BatchNorm1d pair into a single equivalent Conv1d
# ---------------------------------------------------------------------------
#
# BatchNorm at inference: y = gamma * (x - running_mean) / sqrt(running_var + eps) + beta
#                          = gamma/s * x  +  (beta - gamma*running_mean/s)        where s = sqrt(var+eps)
# Conv1d output (channel c): y[c] = sum_k(W[c,k] * x[k]) + b[c]
#
# After folding into Conv:
#     W_new[c] = W[c] * (gamma[c] / s[c])
#     b_new[c] = (b[c] - running_mean[c]) * (gamma[c] / s[c]) + beta[c]
#
# Result: identical predictions, but the C code only needs Conv (no BN).
# ---------------------------------------------------------------------------

def fold_conv_bn(conv: torch.nn.Conv1d, bn: torch.nn.BatchNorm1d):
    w = conv.weight.detach().clone()              # (out_channels, in_channels, kernel)
    b = conv.bias.detach().clone() if conv.bias is not None \
        else torch.zeros(conv.out_channels)
    gamma = bn.weight.detach().clone()
    beta  = bn.bias.detach().clone()
    mean  = bn.running_mean.detach().clone()
    var   = bn.running_var.detach().clone()
    eps   = bn.eps
    scale = gamma / torch.sqrt(var + eps)          # per-channel scalar
    # Multiply each output channel of W and b by `scale[c]`
    w_new = w * scale.view(-1, 1, 1)
    b_new = (b - mean) * scale + beta
    return w_new.numpy(), b_new.numpy()

w1, b1 = fold_conv_bn(model.conv1, model.bn1)
w2, b2 = fold_conv_bn(model.conv2, model.bn2)
w3, b3 = fold_conv_bn(model.conv3, model.bn3)
fc_w = model.fc.weight.detach().numpy()           # (5, 32)
fc_b = model.fc.bias.detach().numpy()             # (5,)
print("Folded BN into Conv layers")


# ---------------------------------------------------------------------------
# 3. Load normalization constants
# ---------------------------------------------------------------------------
mean_arr = np.load(MEAN_PATH).flatten().astype(np.float32)  # shape (6,)
std_arr  = np.load(STD_PATH).flatten().astype(np.float32)   # shape (6,)
assert mean_arr.shape == (6,), f"Expected mean shape (6,), got {mean_arr.shape}"
assert std_arr.shape  == (6,), f"Expected std shape (6,), got {std_arr.shape}"
print("Loaded normalization constants")


# ---------------------------------------------------------------------------
# 4. Pretty-print a numpy array as a C float[] initializer
# ---------------------------------------------------------------------------
def c_array(name, arr):
    """Return a C const float array literal for `arr`."""
    flat = arr.flatten().astype(np.float32)
    shape_str = "][".join(str(d) for d in arr.shape)
    out = [f"static const float {name}[{shape_str}] = "]
    out.append("{")
    # Format with 8 numbers per line, 9 significant digits
    for i in range(0, len(flat), 8):
        chunk = ", ".join(f"{x: .9e}f" for x in flat[i:i+8])
        out.append("    " + chunk + ",")
    # Strip trailing comma on last value
    out[-1] = out[-1].rstrip(",")
    out.append("};\n")
    return "\n".join(out)


def c_array_1d(name, arr):
    """1D variant — useful for mean, std, biases."""
    flat = arr.flatten().astype(np.float32)
    out = [f"static const float {name}[{len(flat)}] = " + "{"]
    for i in range(0, len(flat), 8):
        chunk = ", ".join(f"{x: .9e}f" for x in flat[i:i+8])
        out.append("    " + chunk + ",")
    out[-1] = out[-1].rstrip(",")
    out.append("};\n")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 5. Write the header
# ---------------------------------------------------------------------------
with open(OUT_PATH, "w") as f:
    f.write("// =============================================================\n")
    f.write("// model_weights.h\n")
    f.write("// Auto-generated by export_to_c.py — do not edit by hand.\n")
    f.write("//\n")
    f.write("// Model: IMUNet (Conv1d x3 + BN folded + Linear)\n")
    f.write("// Input:  (6 channels, 64 timesteps), already normalized\n")
    f.write("// Output: 5 logits  (IDLE=0, SHAKE=1, TAP=2, SPIN=3, FALL=4)\n")
    f.write("// =============================================================\n\n")
    f.write("#ifndef MODEL_WEIGHTS_H\n")
    f.write("#define MODEL_WEIGHTS_H\n\n")

    # Shape constants
    f.write("// --- shape constants ---\n")
    f.write(f"#define INPUT_CHANNELS   {6}\n")
    f.write(f"#define INPUT_TIMESTEPS  {64}\n")
    f.write(f"#define NUM_CLASSES      {5}\n\n")
    f.write(f"#define CONV1_OUT_CH  {w1.shape[0]}\n")
    f.write(f"#define CONV1_KERNEL  {w1.shape[2]}\n")
    f.write(f"#define CONV1_PAD     {3}\n")
    f.write(f"#define CONV2_OUT_CH  {w2.shape[0]}\n")
    f.write(f"#define CONV2_KERNEL  {w2.shape[2]}\n")
    f.write(f"#define CONV2_PAD     {2}\n")
    f.write(f"#define CONV3_OUT_CH  {w3.shape[0]}\n")
    f.write(f"#define CONV3_KERNEL  {w3.shape[2]}\n")
    f.write(f"#define CONV3_PAD     {1}\n")
    f.write(f"#define POOL_SIZE     {4}\n\n")

    # Normalization
    f.write("// --- per-channel normalization (from training) ---\n")
    f.write(c_array_1d("MODEL_INPUT_MEAN", mean_arr))
    f.write(c_array_1d("MODEL_INPUT_STD",  std_arr))
    f.write("\n")

    # Convolutional layers
    f.write("// --- conv1 (folded with bn1) ---\n")
    f.write(c_array("CONV1_WEIGHT", w1))     # (16, 6, 7)
    f.write(c_array_1d("CONV1_BIAS", b1))    # (16,)
    f.write("\n")

    f.write("// --- conv2 (folded with bn2) ---\n")
    f.write(c_array("CONV2_WEIGHT", w2))     # (32, 16, 5)
    f.write(c_array_1d("CONV2_BIAS", b2))    # (32,)
    f.write("\n")

    f.write("// --- conv3 (folded with bn3) ---\n")
    f.write(c_array("CONV3_WEIGHT", w3))     # (32, 32, 3)
    f.write(c_array_1d("CONV3_BIAS", b3))    # (32,)
    f.write("\n")

    # Fully connected layer
    f.write("// --- fc (linear) ---\n")
    f.write(c_array("FC_WEIGHT", fc_w))      # (5, 32)
    f.write(c_array_1d("FC_BIAS", fc_b))     # (5,)
    f.write("\n")

    f.write("#endif  // MODEL_WEIGHTS_H\n")

# Report sizing
n_params = (w1.size + b1.size + w2.size + b2.size + w3.size + b3.size
            + fc_w.size + fc_b.size + mean_arr.size + std_arr.size)
print(f"Wrote {OUT_PATH}")
print(f"Total constants: {n_params} floats = {n_params*4} bytes ({n_params*4/1024:.1f} KB)")
print()
print("Drop model_weights.h into your ESP-IDF component alongside cnn_inference.c/h")
