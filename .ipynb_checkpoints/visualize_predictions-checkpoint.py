import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    EnsureTyped,
)
from matplotlib.patches import Patch

from model import build_model

# ======================
# Args
# ======================
parser = argparse.ArgumentParser()
parser.add_argument("--variant", type=str, default="baseline",
                    choices=["baseline", "wavelet_a", "wavelet_a_plus", "wavelet_b", "wavelet_a_higher_level"])
parser.add_argument("--patient", type=str, default="P1",
                    help="Patient ID, e.g. P1")
parser.add_argument("--timepoint", type=str, default="T1",
                    help="Timepoint, e.g. T1")
args = parser.parse_args()

VARIANT = args.variant
ROOT = "MSLesSeg_Dataset"

# ======================
# CONFIG
# ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = f"best_{VARIANT}.pth"

P = args.patient
T = args.timepoint
IMAGE_PATH = f"{ROOT}/train/{P}/{T}/{P}_{T}_FLAIR.nii.gz"
LABEL_PATH = f"{ROOT}/train/{P}/{T}/{P}_{T}_MASK.nii.gz"

# IMAGE_PATH = f"{ROOT}/test/{P}/{P}_FLAIR.nii.gz"
# LABEL_PATH = f"{ROOT}/test/{P}/{P}_MASK.nii.gz"

ROI_SIZE = (96, 96, 96)
SW_BATCH_SIZE = 2

# ======================
# MONAI transforms (for inference ONLY)
# ======================
val_transform = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(
        keys=["image", "label"],
        pixdim=(1, 1, 1),
        mode=("bilinear", "nearest"),
    ),
    NormalizeIntensityd(keys="image", nonzero=True),
    EnsureTyped(keys=["image", "label"]),
])

print(f"Variant : {VARIANT}")
# print(f"Patient : {P}, Timepoint: {T}")
print(f"Image   : {IMAGE_PATH}")

# ======================
# Load data
# ======================
data = val_transform({"image": IMAGE_PATH, "label": LABEL_PATH})
image_tensor = data["image"].unsqueeze(0).to(DEVICE)  # [1,1,D,H,W]

# ======================
# Load model
# ======================
model = build_model(VARIANT)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
)

model.to(DEVICE)
model.eval()

# ======================
# Predict
# ======================
with torch.no_grad():
    logits = sliding_window_inference(
        image_tensor, ROI_SIZE, SW_BATCH_SIZE, model,
    )
    probs = torch.sigmoid(logits)

pred_bin = (probs > 0.3).cpu().numpy()[0, 0]

flair_raw = data["image"].cpu().numpy()[0]
gt_raw = data["label"].cpu().numpy()[0]

print("Pred min/max:", probs.min().item(), probs.max().item())
print("Pred voxels:", pred_bin.sum())

# ======================
# Select lesion-rich slice
# ======================
if gt_raw.ndim == 3:
    lesion_slices = np.where(gt_raw.sum(axis=(1, 2)) > 0)[0]
    slice_idx = (
        lesion_slices[len(lesion_slices) // 2]
        if len(lesion_slices) > 0
        else gt_raw.shape[0] // 2
    )
    flair_slice = flair_raw[slice_idx]
    gt_slice    = gt_raw[slice_idx]
    pred_slice  = pred_bin[slice_idx]
else:
    flair_slice = flair_raw
    gt_slice    = gt_raw
    pred_slice  = pred_bin

# ======================
# Contrast stretching
# ======================
p2, p98 = np.percentile(flair_slice, (2, 98))
flair_slice = np.clip((flair_slice - p2) / (p98 - p2), 0, 1)

# ======================
# Plot
# ======================
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("FLAIR")
plt.imshow(flair_slice, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Ground Truth")
plt.imshow(flair_slice, cmap="gray")
plt.imshow(gt_slice, cmap="Reds", alpha=0.6)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Prediction")
plt.imshow(flair_slice, cmap="gray")
plt.imshow(pred_slice, cmap="Greens", alpha=0.6)
plt.axis("off")

plt.tight_layout()
plt.legend(
    handles=[
        Patch(facecolor="red",   edgecolor="red",   label="Ground Truth"),
        Patch(facecolor="green", edgecolor="green", label="Prediction"),
    ],
    loc="lower center", ncol=2, frameon=False,
    bbox_to_anchor=(0.5, -0.05),
)

out_path = f"visualize_predictions_{VARIANT}_{P}.png"
plt.savefig(out_path, dpi=300)
print(f"Saved: {out_path}")
plt.show()
