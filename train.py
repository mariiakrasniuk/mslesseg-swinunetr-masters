import argparse
import torch
import matplotlib.pyplot as plt
from torch.optim import AdamW
from tqdm import tqdm

from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import build_model
from dataloaders import get_loaders
from splits import TRAIN_PATIENTS, VAL_PATIENTS

# ------------------
# Args
# ------------------
parser = argparse.ArgumentParser()
parser.add_argument("--variant", type=str, default="baseline",
                    choices=["baseline", "wavelet_a", "wavelet_a_plus", "wavelet_b",
                             "wavelet_a_higher_level", "wavelet_ml", "wavelet_swt"],
                    help="Model variant to train")
parser.add_argument("--wavelet", type=str, default="haar",
                    choices=["haar", "db2", "sym4"],
                    help="Wavelet family — only used with --variant wavelet_ml")
parser.add_argument("--levels", type=int, default=1,
                    choices=[1, 2, 3],
                    help="Decomposition levels — only used with --variant wavelet_ml")
args = parser.parse_args()

VARIANT = args.variant
WAVELET = args.wavelet
LEVELS  = args.levels

# Run name encodes variant + wavelet family + level for wavelet_ml experiments.
# Legacy variants keep a flat name so existing checkpoints are not affected.
if VARIANT == "wavelet_ml":
    RUN_NAME = f"wavelet_ml_{WAVELET}_l{LEVELS}"
elif VARIANT == "wavelet_swt":
    RUN_NAME = f"wavelet_swt_{WAVELET}_l{LEVELS}"
else:
    RUN_NAME = VARIANT

# ------------------
# Config
# ------------------
ROOT = "MSLesSeg_Dataset"
DEVICE = "cuda"
EPOCHS = 70
LR = 1e-4
WEIGHT_DECAY = 1e-5
ROI_SIZE = (96, 96, 96)
SW_BATCH_SIZE = 2

# Early stopping
PATIENCE = 10
MIN_DELTA = 1e-4

# Run-aware output paths
BEST_MODEL_PATH    = f"best_{RUN_NAME}.pth"
HISTORY_PATH       = f"training_history_{RUN_NAME}.pth"
LOSS_CURVE_PATH    = f"loss_curves_{RUN_NAME}.png"
DICE_CURVE_PATH    = f"dice_curves_{RUN_NAME}.png"

print(f"Variant : {VARIANT}")
if VARIANT == "wavelet_ml":
    print(f"Wavelet : {WAVELET}  |  Levels: {LEVELS}")
print(f"Run name: {RUN_NAME}")
print(f"Best model will be saved to: {BEST_MODEL_PATH}")

# ------------------
# Data
# ------------------
train_loader, val_loader = get_loaders(
    ROOT, TRAIN_PATIENTS, VAL_PATIENTS
)

# ------------------
# Model
# ------------------
model = build_model(VARIANT, use_checkpoint=True,
                    wavelet=WAVELET, levels=LEVELS).to(DEVICE)

# ------------------
# Loss / Optim / Metrics
# ------------------
loss_fn = DiceCELoss(sigmoid=True, lambda_dice=1.0, lambda_ce=0.5)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

dice_metric = DiceMetric(include_background=False, reduction="mean")

# ------------------
# State
# ------------------
best_dice = 0.0
best_val_loss = float("inf")
early_stop_counter = 0

# ------------------
# History
# ------------------
train_loss_history = []
val_loss_history = []
train_dice_history = []
val_dice_history = []

# ------------------
# Training loop
# ------------------
for epoch in range(1, EPOCHS + 1):

    # ========= TRAIN =========
    model.train()
    dice_metric.reset()

    train_loss = 0.0
    steps = 0

    train_iter = iter(train_loader)
    for _ in tqdm(range(len(train_loader)), desc=f"Epoch {epoch} [train]"):
        try:
            batch = next(train_iter)
        except RuntimeError as e:
            # MONAI wraps the original CUDA error, so walk the full chain
            exc, chain = e, ""
            while exc is not None:
                chain += str(exc)
                exc = getattr(exc, "__cause__", None)
            if "INTERNAL ASSERT" in chain:
                print(f"\n[warn] skipping bad batch: {e}")
                continue
            raise

        # batch is list of dicts (multi-patch from RandCropByPosNegLabeld)
        for sample in batch:
            x = sample["image"].to(DEVICE)
            y = sample["label"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            steps += 1

            preds = torch.sigmoid(logits)
            dice_metric(preds, y)

    scheduler.step()
    train_loss /= steps
    train_dice = dice_metric.aggregate().item()

    train_loss_history.append(train_loss)
    train_dice_history.append(train_dice)

    # ========= VALIDATION =========
    model.eval()
    dice_metric.reset()

    val_loss = 0.0
    val_steps = 0

    with torch.no_grad():
        val_iter = iter(val_loader)
        for _ in tqdm(range(len(val_loader)), desc=f"Epoch {epoch} [val]"):
            try:
                batch = next(val_iter)
            except RuntimeError as e:
                exc, chain = e, ""
                while exc is not None:
                    chain += str(exc)
                    exc = getattr(exc, "__cause__", None)
                if "INTERNAL ASSERT" in chain:
                    print(f"\n[warn] skipping bad val batch: {e}")
                    continue
                raise

            x = batch["image"].to(DEVICE)
            y = batch["label"].to(DEVICE)

            preds = sliding_window_inference(
                x, ROI_SIZE, SW_BATCH_SIZE, model
            )

            loss = loss_fn(preds, y)
            val_loss += loss.item()
            val_steps += 1

            preds = torch.sigmoid(preds)
            dice_metric(preds, y)

    val_loss = val_loss / val_steps if val_steps > 0 else float("nan")
    val_dice = dice_metric.aggregate().item() if val_steps > 0 else 0.0

    val_loss_history.append(val_loss)
    val_dice_history.append(val_dice)

    # ========= LOG =========
    print(
        f"Epoch {epoch:03d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Train Dice: {train_dice:.4f} | "
        f"Val Dice: {val_dice:.4f}"
    )

    # ========= CHECKPOINT =========
    if val_dice > best_dice:
        best_dice = val_dice
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"New best model saved (Val Dice={best_dice:.4f})")

    # ========= EARLY STOPPING =========
    if val_loss < best_val_loss - MIN_DELTA:
        best_val_loss = val_loss
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        print(f"EarlyStopping {early_stop_counter}/{PATIENCE}")

    if early_stop_counter >= PATIENCE:
        print("Early stopping triggered")
        break

# ------------------
# Save history
# ------------------
torch.save(
    {
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
        "train_dice": train_dice_history,
        "val_dice": val_dice_history,
    },
    HISTORY_PATH,
)

# ------------------
# Plot: Loss
# ------------------
plt.figure(figsize=(8, 5))
plt.plot(train_loss_history, label="Train Loss")
plt.plot(val_loss_history, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Dice Loss")
plt.title(f"Training / Validation Loss [{RUN_NAME}]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(LOSS_CURVE_PATH, dpi=300)
plt.close()

# ------------------
# Plot: Dice
# ------------------
plt.figure(figsize=(8, 5))
plt.plot(train_dice_history, label="Train Dice")
plt.plot(val_dice_history, label="Val Dice")
plt.xlabel("Epoch")
plt.ylabel("Dice Score")
plt.title(f"Training / Validation Dice [{RUN_NAME}]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(DICE_CURVE_PATH, dpi=300)
plt.close()

print(f"Training history saved to {HISTORY_PATH}")
print(f"Loss curves saved to {LOSS_CURVE_PATH}")
print(f"Dice curves saved to {DICE_CURVE_PATH}")
