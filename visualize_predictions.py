import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt

from monai.networks.nets import SwinUNETR
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


# ======================
# CONFIG
# ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "best_swinunetr.pth"

ID1= "P1"
ID2= "T1"

# IMAGE_PATH = f"MSLesSeg Dataset/train/{ID1}/{ID1}_{ID2}_FLAIR.nii.gz"
# LABEL_PATH = f"MSLesSeg Dataset/train/{ID1}/{ID1}_{ID2}_MASK.nii.gz"
IMAGE_PATH = "/home/dima/projects/mslesseg-swinunetr/MSLesSeg Dataset/train/P25/T1/P25_T1_FLAIR.nii.gz"
LABEL_PATH = "/home/dima/projects/mslesseg-swinunetr/MSLesSeg Dataset/train/P25/T1/P25_T1_MASK.nii.gz"

ROI_SIZE = (96, 96, 96)
SW_BATCH_SIZE = 2

# ======================
# MONAI transforms (for inference ONLY)
# ======================
val_transform = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
    Spacingd(
        keys=["image", "label"],
        pixdim=(1, 1, 1),
        mode=("bilinear", "nearest"),
    ),
    NormalizeIntensityd(keys="image", nonzero=True),
    EnsureTyped(keys=["image", "label"]),
])

# ======================
# Load data for inference
# ======================
data = val_transform({"image": IMAGE_PATH, "label": LABEL_PATH})

image_tensor = data["image"].unsqueeze(0).to(DEVICE)  # [1,1,D,H,W]

# ======================
# Load model
# ======================
model = SwinUNETR(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    feature_size=48,
)
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
        image_tensor,
        ROI_SIZE,
        SW_BATCH_SIZE,
        model,
    )
    probs = torch.sigmoid(logits)

pred_bin = (probs > 0.3).cpu().numpy()[0, 0]  # [D,H,W] or [H,W]

# ======================
# Load RAW NIfTI for visualization
# ======================
flair_raw = data["image"].cpu().numpy()[0]
gt_raw = data["label"].cpu().numpy()[0]

print("Pred min/max:", probs.min().item(), probs.max().item())
print("Pred voxels:", pred_bin.sum())



# ======================
# Select slice (lesion-rich if possible)
# ======================
if gt_raw.ndim == 3:
    lesion_slices = np.where(gt_raw.sum(axis=(1, 2)) > 0)[0]
    slice_idx = (
        lesion_slices[len(lesion_slices) // 2]
        if len(lesion_slices) > 0
        else gt_raw.shape[0] // 2
    )

    flair_slice = flair_raw[slice_idx]
    gt_slice = gt_raw[slice_idx]
    pred_slice = pred_bin[slice_idx]

else:
    # 2D case
    flair_slice = flair_raw
    gt_slice = gt_raw
    pred_slice = pred_bin

# ======================
# Contrast stretching (CRITICAL)
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
# ------------------
# Legend
# ------------------
legend_elements = [
    Patch(facecolor="red", edgecolor="red", label="Ground Truth"),
    Patch(facecolor="green", edgecolor="green", label="Prediction"),
]

plt.legend(
    handles=legend_elements,
    loc="lower center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, -0.05),
)

plt.savefig("visualize_predictions.png", dpi=300)
plt.show()

# import torch
# import numpy as np
# import nibabel as nib
# import matplotlib.pyplot as plt

# from monai.networks.nets import SwinUNETR
# from monai.inferers import sliding_window_inference
# from monai.transforms import (
#     Compose, LoadImaged, EnsureChannelFirstd,
#     Orientationd, Spacingd, NormalizeIntensityd, EnsureTyped
# )

# # ------------------
# # Config
# # ------------------
# DEVICE = "cuda"
# ROI_SIZE = (96, 96, 96)
# SW_BATCH_SIZE = 2

# MODEL_PATH = "best_swinunetr.pth"

# # pick ONE example (test set recommended)
# IMAGE_PATH = "MSLesSeg Dataset/test/P60/P60_FLAIR.nii.gz"
# LABEL_PATH = "MSLesSeg Dataset/test/P60/P60_MASK.nii.gz"

# # ------------------
# # Transforms (same as validation)
# # ------------------
# val_transform = Compose([
#     LoadImaged(keys=["image", "label"]),
#     EnsureChannelFirstd(keys=["image", "label"]),
#     Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
#     Spacingd(
#         keys=["image", "label"],
#         pixdim=(1, 1, 1),
#         mode=("bilinear", "nearest"),
#     ),
#     NormalizeIntensityd(keys="image", nonzero=True),
#     EnsureTyped(keys=["image", "label"]),
# ])

# # ------------------
# # Load data
# # ------------------
# data = val_transform({"image": IMAGE_PATH, "label": LABEL_PATH})

# image = data["image"].unsqueeze(0).to(DEVICE)  # [1,1,D,H,W]
# label_3d = data["label"].cpu().numpy()[0, 0]  # [D,H,W]

# # ------------------
# # Load model
# # ------------------
# model = SwinUNETR(
#     spatial_dims=3,
#     in_channels=1,
#     out_channels=1,
#     feature_size=48,
# )
# model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
# model.to(DEVICE)
# model.eval()

# # ------------------
# # Predict
# # ------------------
# with torch.no_grad():
#     logits = sliding_window_inference(
#         image, ROI_SIZE, SW_BATCH_SIZE, model
#     )
#     pred = torch.sigmoid(logits).cpu().numpy()[0, 0]

# pred_bin = (pred > 0.5).astype(np.uint8)

# # ------------------
# # Choose slice (middle or lesion-rich)
# # ------------------
# # Find slices that contain lesions
# # ------------------
# # Handle 2D vs 3D safely
# # ------------------
# # ------------------
# # Handle 3D vs 2D safely
# # ------------------
# # ------------------
# # Prepare numpy arrays
# # ------------------
# image_np = image.cpu().numpy()[0, 0]   # [D,H,W] or [H,W]
# label_np = label_3d                    # [D,H,W] or [H,W]
# pred_np = pred_bin                     # same shape as label

# # ------------------
# # Ensure 2D slices for plotting
# # ------------------
# if image_np.ndim == 3:
#     # choose a slice with lesions if possible
#     if label_np.ndim == 3:
#         lesion_slices = np.where(label_np.sum(axis=(1, 2)) > 0)[0]
#         slice_idx = (
#             lesion_slices[len(lesion_slices) // 2]
#             if len(lesion_slices) > 0
#             else image_np.shape[0] // 2
#         )
#     else:
#         slice_idx = image_np.shape[0] // 2

#     flair_slice = image_np[slice_idx]
#     gt_slice = label_np[slice_idx] if label_np.ndim == 3 else label_np
#     pred_slice = pred_np[slice_idx] if pred_np.ndim == 3 else pred_np

# else:
#     # already 2D
#     flair_slice = image_np
#     gt_slice = label_np
#     pred_slice = pred_np




# # ------------------
# # Plot
# # ------------------
# plt.figure(figsize=(12, 4))

# plt.subplot(1, 3, 1)
# plt.title("FLAIR")
# plt.imshow(flair_slice, cmap="gray")
# plt.axis("off")

# plt.subplot(1, 3, 2)
# plt.title("Ground Truth")
# plt.imshow(flair_slice, cmap="gray")
# plt.imshow(gt_slice, cmap="Reds", alpha=0.5)
# plt.axis("off")

# plt.subplot(1, 3, 3)
# plt.title("Prediction")
# plt.imshow(flair_slice, cmap="gray")
# plt.imshow(pred_slice, cmap="Greens", alpha=0.5)
# plt.axis("off")

# plt.tight_layout()
# plt.savefig("visualize_predictions.png", dpi=300)
# plt.show()
