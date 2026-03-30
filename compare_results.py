"""
compare_results.py

For every ablation run:
  1. Reads training_history_*.pth  →  best val Dice, epochs trained
  2. Loads best_*.pth + runs sliding-window inference on the test set  →  test Dice

Outputs
-------
  - Formatted comparison table printed to stdout
  - results_comparison.csv
"""

import os
import csv
import math
import torch
from tqdm import tqdm
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.data import Dataset
from torch.utils.data import DataLoader

from model import build_model
from build_datalist import build_test_list
from transforms import get_val_transforms

# ---------------------------------------------------------------------------
# Experiment registry — must match train.py naming exactly
# ---------------------------------------------------------------------------
RUNS = [
    {"run_name": "baseline",           "variant": "baseline",   "wavelet": "haar", "levels": 1},
    {"run_name": "wavelet_ml_haar_l1", "variant": "wavelet_ml", "wavelet": "haar", "levels": 1},
    {"run_name": "wavelet_ml_haar_l2", "variant": "wavelet_ml", "wavelet": "haar", "levels": 2},
    {"run_name": "wavelet_ml_haar_l3", "variant": "wavelet_ml", "wavelet": "haar", "levels": 3},
    {"run_name": "wavelet_ml_db2_l1",  "variant": "wavelet_ml", "wavelet": "db2",  "levels": 1},
    {"run_name": "wavelet_ml_db2_l2",  "variant": "wavelet_ml", "wavelet": "db2",  "levels": 2},
    {"run_name": "wavelet_ml_db2_l3",  "variant": "wavelet_ml", "wavelet": "db2",  "levels": 3},
    {"run_name": "wavelet_ml_sym4_l1", "variant": "wavelet_ml", "wavelet": "sym4", "levels": 1},
    {"run_name": "wavelet_ml_sym4_l2", "variant": "wavelet_ml", "wavelet": "sym4", "levels": 2},
    {"run_name": "wavelet_ml_sym4_l3", "variant": "wavelet_ml", "wavelet": "sym4", "levels": 3},
]

ROOT         = "MSLesSeg_Dataset"
DEVICE       = "cuda"
ROI_SIZE     = (96, 96, 96)
SW_BATCH_SIZE = 2

# ---------------------------------------------------------------------------
# Build test loader once (reused for every model)
# ---------------------------------------------------------------------------
print("Loading test set...")
test_ds     = Dataset(build_test_list(ROOT), transform=get_val_transforms())
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
print(f"Test set: {len(test_loader)} volumes\n")

dice_metric = DiceMetric(include_background=False, reduction="mean")

# ---------------------------------------------------------------------------
# Evaluate each run
# ---------------------------------------------------------------------------
results = []

for run in RUNS:
    run_name     = run["run_name"]
    history_path = f"training_history_{run_name}.pth"
    model_path   = f"best_{run_name}.pth"

    # ---- val metrics from saved training history ----
    best_val_dice  = float("nan")
    final_val_dice = float("nan")
    epochs_trained = 0

    if os.path.exists(history_path):
        h = torch.load(history_path, map_location="cpu")
        vd = h["val_dice"]
        best_val_dice  = float(max(vd))
        final_val_dice = float(vd[-1])
        epochs_trained = len(vd)
    else:
        print(f"  [warn] missing history: {history_path}")

    # ---- test dice via inference ----
    test_dice = float("nan")

    if os.path.exists(model_path):
        model = build_model(
            run["variant"],
            wavelet=run["wavelet"],
            levels=run["levels"],
        ).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        dice_metric.reset()

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"  {run_name}", leave=False):
                x = batch["image"].to(DEVICE)
                y = batch["label"].to(DEVICE)
                preds = sliding_window_inference(x, ROI_SIZE, SW_BATCH_SIZE, model)
                preds = torch.sigmoid(preds)
                dice_metric(preds, y)

        test_dice = float(dice_metric.aggregate().item())
        del model
        torch.cuda.empty_cache()
    else:
        print(f"  [warn] missing model: {model_path}")

    results.append({
        "run_name":      run_name,
        "best_val_dice": best_val_dice,
        "final_val_dice": final_val_dice,
        "test_dice":     test_dice,
        "epochs":        epochs_trained,
    })

    def _fmt(v):
        return f"{v:.4f}" if not math.isnan(v) else "  N/A "

    print(f"  {run_name:<30}  val={_fmt(best_val_dice)}  test={_fmt(test_dice)}  ep={epochs_trained}")

# ---------------------------------------------------------------------------
# Formatted comparison table
# ---------------------------------------------------------------------------
SEP = "=" * 72
print(f"\n{SEP}")
print(f"{'Run':<30} {'Best Val':>9} {'Final Val':>10} {'Test Dice':>10} {'Epochs':>7}")
print("-" * 72)

best_test = max((r["test_dice"] for r in results if not math.isnan(r["test_dice"])), default=None)

for r in results:
    def _fmt(v):
        return f"{v:.4f}" if not math.isnan(v) else "  N/A"

    marker = " <-- best" if (best_test is not None
                             and not math.isnan(r["test_dice"])
                             and abs(r["test_dice"] - best_test) < 1e-6) else ""
    print(
        f"{r['run_name']:<30} "
        f"{_fmt(r['best_val_dice']):>9} "
        f"{_fmt(r['final_val_dice']):>10} "
        f"{_fmt(r['test_dice']):>10} "
        f"{r['epochs']:>7}"
        f"{marker}"
    )

print(SEP)

# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------
csv_path = "results_comparison.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["run_name", "best_val_dice", "final_val_dice", "test_dice", "epochs"],
    )
    writer.writeheader()
    writer.writerows(results)

print(f"\nSaved: {csv_path}")
