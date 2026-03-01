import argparse
import torch
import matplotlib.pyplot as plt
from torch.optim import AdamW
from tqdm import tqdm

from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

from dataloaders import get_loaders
from splits import TRAIN_PATIENTS, VAL_PATIENTS
from model import build_model

# ------------------
# Args
# ------------------
parser = argparse.ArgumentParser()
parser.add_argument("--variant",  default="baseline",
                    help="Model variant: baseline | wavelet_a | wavelet_b | "
                         "wavelet_ab | wavelet_ab_freq")
parser.add_argument("--run_name", default=None,
                    help="Checkpoint/history filename prefix. Defaults to variant.")
args = parser.parse_args()
RUN_NAME = args.run_name or args.variant

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

# ------------------
# Data
# ------------------
train_loader, val_loader = get_loaders(
    ROOT, TRAIN_PATIENTS, VAL_PATIENTS
)

# ------------------
# Model + Loss
# ------------------
model, loss_fn = build_model(args.variant)
model = model.to(DEVICE)

# ------------------
# Optim / Metrics
# ------------------
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

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

    for batch in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
        # batch is list of dicts (multi-patch)
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
        for batch in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
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

    val_loss /= val_steps
    val_dice = dice_metric.aggregate().item()

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
        torch.save(model.state_dict(), f"best_{RUN_NAME}.pth")
        print(f"New best model saved (Val Dice={best_dice:.4f})")

    # ========= EARLY STOPPING =========
    if val_loss < best_val_loss - MIN_DELTA:
        best_val_loss = val_loss
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        print(f"⏸ EarlyStopping {early_stop_counter}/{PATIENCE}")

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
    f"training_history_{RUN_NAME}.pth",
)

# ------------------
# Plot: Loss
# ------------------
plt.figure(figsize=(8, 5))
plt.plot(train_loss_history, label="Train Loss")
plt.plot(val_loss_history, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Dice Loss")
plt.title("Training / Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"loss_curves_{RUN_NAME}.png", dpi=300)
plt.close()

# ------------------
# Plot: Dice
# ------------------
plt.figure(figsize=(8, 5))
plt.plot(train_dice_history, label="Train Dice")
plt.plot(val_dice_history, label="Val Dice")
plt.xlabel("Epoch")
plt.ylabel("Dice Score")
plt.title("Training / Validation Dice")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"dice_curves_{RUN_NAME}.png", dpi=300)
plt.close()

print(f"Training history saved to training_history_{RUN_NAME}.pth")
print(f"Loss curves saved to loss_curves_{RUN_NAME}.png")
print(f"Dice curves saved to dice_curves_{RUN_NAME}.png")
