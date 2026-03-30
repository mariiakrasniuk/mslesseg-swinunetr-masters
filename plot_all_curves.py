"""
plot_all_curves.py

Overlays training history curves for all ablation runs on a single figure.
Produces one 2×2 panel: train loss / val loss / train dice / val dice.

Visual encoding
---------------
  Colour  → wavelet family  (black=baseline, blue=haar, red=db2, green=sym4)
  Style   → decomposition level  (solid=1, dashed=2, dotted=3)

Output: all_curves.png
"""

import os
import torch
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ---------------------------------------------------------------------------
# All runs in the same order as the experiment plan
# ---------------------------------------------------------------------------
RUNS = [
    "baseline",
    "wavelet_ml_haar_l1",
    "wavelet_ml_haar_l2",
    "wavelet_ml_haar_l3",
    "wavelet_ml_db2_l1",
    "wavelet_ml_db2_l2",
    "wavelet_ml_db2_l3",
    "wavelet_ml_sym4_l1",
    "wavelet_ml_sym4_l2",
    "wavelet_ml_sym4_l3",
]

COLORS = {
    "baseline": "black",
    "haar":     "#1f77b4",   # blue
    "db2":      "#d62728",   # red
    "sym4":     "#2ca02c",   # green
}
STYLES = {1: "-", 2: "--", 3: ":"}


def _style(run_name):
    """Return (color, linestyle, label) for a run name."""
    if run_name == "baseline":
        return COLORS["baseline"], "-", "baseline"
    # format: wavelet_ml_{wavelet}_l{level}
    parts  = run_name.split("_")   # ['wavelet', 'ml', wavelet, 'l{n}']
    wavelet = parts[2]
    level   = int(parts[3][1:])
    return COLORS[wavelet], STYLES[level], f"{wavelet} L{level}"


# ---------------------------------------------------------------------------
# Load histories
# ---------------------------------------------------------------------------
histories = {}
for run in RUNS:
    path = f"training_history_{run}.pth"
    if not os.path.exists(path):
        print(f"[skip] not found: {path}")
        continue
    histories[run] = torch.load(path, map_location="cpu")

if not histories:
    raise FileNotFoundError("No training_history_*.pth files found.")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
(ax_tl, ax_vl), (ax_td, ax_vd) = axes

panel_cfg = [
    (ax_tl, "train_loss",  "Train Loss",       "Dice Loss"),
    (ax_vl, "val_loss",    "Validation Loss",  "Dice Loss"),
    (ax_td, "train_dice",  "Train Dice",       "Dice Score"),
    (ax_vd, "val_dice",    "Validation Dice",  "Dice Score"),
]

for ax, key, title, ylabel in panel_cfg:
    for run, h in histories.items():
        if key not in h:
            continue
        color, style, label = _style(run)
        lw = 2.0 if run == "baseline" else 1.4
        ax.plot(h[key], color=color, linestyle=style, linewidth=lw, label=label)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)

# ---------------------------------------------------------------------------
# Shared legend (colour = family, style = level)
# ---------------------------------------------------------------------------
legend_handles = [
    mlines.Line2D([], [], color="black",        linestyle="-",  linewidth=2,   label="baseline"),
    mlines.Line2D([], [], color=COLORS["haar"], linestyle="-",  linewidth=1.4, label="Haar"),
    mlines.Line2D([], [], color=COLORS["db2"],  linestyle="-",  linewidth=1.4, label="db2"),
    mlines.Line2D([], [], color=COLORS["sym4"], linestyle="-",  linewidth=1.4, label="sym4"),
    mlines.Line2D([], [], color="gray",         linestyle="-",  linewidth=1.4, label="Level 1"),
    mlines.Line2D([], [], color="gray",         linestyle="--", linewidth=1.4, label="Level 2"),
    mlines.Line2D([], [], color="gray",         linestyle=":",  linewidth=1.4, label="Level 3"),
]
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=7,
    fontsize=10,
    bbox_to_anchor=(0.5, -0.03),
    frameon=True,
)

fig.suptitle(
    "Wavelet Family × Decomposition Level — All Runs",
    fontsize=14, fontweight="bold",
)
plt.tight_layout(rect=[0, 0.05, 1, 0.97])
plt.savefig("all_curves.png", dpi=150, bbox_inches="tight")
print("Saved: all_curves.png")
plt.close()
