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


def get_test_loader(root, multimodal: bool = False):
    if multimodal:
        from build_datalist import build_test_list_mm
        from transforms import get_val_transforms_mm
        test_data = build_test_list_mm(root)
        test_ds   = Dataset(test_data, transform=get_val_transforms_mm())
    else:
        test_data = build_test_list(root)
        test_ds   = Dataset(test_data, transform=get_val_transforms())
    return DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)


# ------------------
# Args
# ------------------
parser = argparse.ArgumentParser()
parser.add_argument("--variant", type=str, default="baseline",
                    choices=["baseline", "wavelet_a", "wavelet_ml"],
                    help="Model variant to evaluate")
parser.add_argument("--wavelet", type=str, default="haar",
                    choices=["haar", "db2", "sym4"],
                    help="Wavelet family — only used with --variant wavelet_ml")
parser.add_argument("--levels", type=int, default=1,
                    choices=[1, 2, 3],
                    help="Decomposition levels — only used with --variant wavelet_ml")
parser.add_argument("--multimodal", action="store_true",
                    help="Use FLAIR + T1 + T2 as input (3 channels)")
parser.add_argument("--use_v2", action="store_true",
                    help="Use SwinUNETR-V2")
parser.add_argument("--aug", type=str, default="none",
                    choices=["none", "image", "coeff", "both"],
                    help="Must match the --aug used during training (affects run name / checkpoint path)")
args = parser.parse_args()

VARIANT     = args.variant
WAVELET     = args.wavelet
LEVELS      = args.levels
MULTIMODAL  = args.multimodal
USE_V2      = args.use_v2
AUG         = args.aug
IN_CHANNELS = 3 if MULTIMODAL else 1
COEFF_AUG   = AUG in ("coeff", "both")

# Must match the naming logic in train.py
if VARIANT == "wavelet_ml":
    RUN_NAME = f"wavelet_ml_{WAVELET}_l{LEVELS}"
else:
    RUN_NAME = VARIANT
if MULTIMODAL:
    RUN_NAME += "_mm"
if USE_V2:
    RUN_NAME += "_v2"
if AUG != "none":
    RUN_NAME += f"_aug_{AUG}"

ROOT = "MSLesSeg_Dataset"
DEVICE = "cuda"
ROI_SIZE = (96, 96, 96)
SW_BATCH_SIZE = 2
MODEL_PATH = f"best_{RUN_NAME}.pth"

print(f"Variant    : {VARIANT}")
if VARIANT == "wavelet_ml":
    print(f"Wavelet    : {WAVELET}  |  Levels: {LEVELS}")
print(f"Multimodal : {MULTIMODAL}  |  SwinV2: {USE_V2}  |  in_channels: {IN_CHANNELS}")
print(f"Run name   : {RUN_NAME}")
print(f"Loading weights from: {MODEL_PATH}")

test_loader = get_test_loader(ROOT, multimodal=MULTIMODAL)

model = build_model(VARIANT, in_channels=IN_CHANNELS,
                    wavelet=WAVELET, levels=LEVELS, use_v2=USE_V2,
                    coeff_aug=COEFF_AUG).to(DEVICE)
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

print(f"[{RUN_NAME}] FINAL TEST Dice: {test_dice:.4f}")
print(f"[{RUN_NAME}] FINAL TEST Loss: {test_loss:.4f}")
