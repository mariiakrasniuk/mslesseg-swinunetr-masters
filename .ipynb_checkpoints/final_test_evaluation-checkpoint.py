import argparse
import torch
from tqdm import tqdm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import Dataset
from torch.utils.data import DataLoader

from model import build_model
from build_datalist import build_test_list
from transforms import get_val_transforms


def get_test_loader(root):
    test_data = build_test_list(root)
    test_ds = Dataset(test_data, transform=get_val_transforms())
    return DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=0
    )


# ------------------
# Args
# ------------------
parser = argparse.ArgumentParser()
parser.add_argument("--variant", type=str, default="baseline",
                    choices=["baseline", "wavelet_a", "wavelet_a_plus", "wavelet_b", "wavelet_a_higher_level"],
                    help="Model variant to evaluate")
args = parser.parse_args()

VARIANT = args.variant

ROOT = "MSLesSeg_Dataset"
DEVICE = "cuda"
ROI_SIZE = (96, 96, 96)
SW_BATCH_SIZE = 2
MODEL_PATH = f"best_{VARIANT}.pth"

print(f"Variant: {VARIANT}")
print(f"Loading weights from: {MODEL_PATH}")

test_loader = get_test_loader(ROOT)

model = build_model(VARIANT).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

dice_metric = DiceMetric(include_background=False, reduction="mean")
loss_fn = DiceLoss(sigmoid=True)

test_loss = 0.0
steps = 0

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Final Test"):
        x = batch["image"].to(DEVICE)
        y = batch["label"].to(DEVICE)

        preds = sliding_window_inference(
            x, ROI_SIZE, SW_BATCH_SIZE, model
        )

        loss = loss_fn(preds, y)
        test_loss += loss.item()
        steps += 1

        preds = torch.sigmoid(preds)
        dice_metric(preds, y)

test_loss /= steps
test_dice = dice_metric.aggregate().item()

print(f"[{VARIANT}] FINAL TEST Dice: {test_dice:.4f}")
print(f"[{VARIANT}] FINAL TEST Loss: {test_loss:.4f}")
