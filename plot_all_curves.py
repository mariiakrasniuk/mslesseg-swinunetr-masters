"""
plot_all_curves.py

Layout: 2 rows × 3 columns
  rows    → val loss (top) | val dice (bottom)
  columns → Haar | db2 | sym4

Each subplot shows three level curves (L1, L2, L3) for that wavelet family
plus the baseline drawn as a thin grey reference line in every panel.

Visual encoding inside each panel
----------------------------------
  Colour  → decomposition level  (blue=L1, orange=L2, green=L3)
  Baseline → dashed grey in every panel for easy comparison

Output: all_curves.png
"""

import os
import torch
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ---------------------------------------------------------------------------
# Run registry
# ---------------------------------------------------------------------------
WAVELETS = ["haar", "db2", "sym4"]
LEVELS   = [1, 2, 3]

LEVEL_COLORS = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}   # blue / orange / green
BASELINE_STYLE = dict(color="grey", linestyle="--", linewidth=1.2, alpha=0.7)

WAVELET_TITLES = {"haar": "Haar", "db2": "Daubechies-2 (db2)", "sym4": "Symlet-4 (sym4)"}

# ---------------------------------------------------------------------------
# Load all histories
# ---------------------------------------------------------------------------
def _history_path(run_name):
    return f"training_history_{run_name}.pth"

def _load(run_name):
    p = _history_path(run_name)
    if not os.path.exists(p):
        print(f"[skip] not found: {p}")
        return None
    return torch.load(p, map_location="cpu")

baseline_h = _load("baseline")

wavelet_data = {}   # wavelet -> {level -> history dict}
for w in WAVELETS:
    wavelet_data[w] = {}
    for l in LEVELS:
        run = f"wavelet_ml_{w}_l{l}"
        h = _load(run)
        if h is not None:
            wavelet_data[w][l] = h

# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharey="row")

METRICS = [
    ("val_loss", "Validation Loss",  "Dice Loss"),
    ("val_dice", "Validation Dice",  "Dice Score"),
]

for row_idx, (metric_key, row_label, ylabel) in enumerate(METRICS):
    for col_idx, wavelet in enumerate(WAVELETS):
        ax = axes[row_idx][col_idx]

        # --- baseline reference ---
        if baseline_h is not None and metric_key in baseline_h:
            ax.plot(baseline_h[metric_key], label="baseline", **BASELINE_STYLE)

        # --- one curve per level ---
        for level in LEVELS:
            h = wavelet_data[wavelet].get(level)
            if h is None or metric_key not in h:
                continue
            ax.plot(
                h[metric_key],
                color=LEVEL_COLORS[level],
                linewidth=1.8,
                label=f"L{level}",
            )

        # --- titles and labels ---
        if row_idx == 0:
            ax.set_title(WAVELET_TITLES[wavelet], fontsize=12, fontweight="bold", pad=8)
        if col_idx == 0:
            ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.grid(True, alpha=0.25)

        # --- row label on the right side of the last column ---
        if col_idx == 2:
            ax.annotate(
                row_label,
                xy=(1.02, 0.5), xycoords="axes fraction",
                rotation=270, va="center", ha="left", fontsize=10,
            )

# ---------------------------------------------------------------------------
# Shared legend below the figure
# ---------------------------------------------------------------------------
legend_handles = [
    mlines.Line2D([], [], **BASELINE_STYLE, label="baseline"),
    mlines.Line2D([], [], color=LEVEL_COLORS[1], linewidth=1.8, label="Level 1"),
    mlines.Line2D([], [], color=LEVEL_COLORS[2], linewidth=1.8, label="Level 2"),
    mlines.Line2D([], [], color=LEVEL_COLORS[3], linewidth=1.8, label="Level 3"),
]
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=4,
    fontsize=11,
    bbox_to_anchor=(0.5, -0.04),
    frameon=True,
)

fig.suptitle(
    "Wavelet Family × Decomposition Level — Validation Curves",
    fontsize=14, fontweight="bold", y=1.01,
)
plt.tight_layout(rect=[0, 0.06, 0.97, 1.0])
plt.savefig("all_curves.png", dpi=150, bbox_inches="tight")
print("Saved: all_curves.png")
plt.close()
