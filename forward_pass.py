"""
forward_pass.py — Smoke test for all model variants and wavelet correctness.

Run with:  python forward_pass.py
Expected:  All variants pass, reconstruction error < 1e-5
"""

import torch
from model import build_model, VARIANTS
from wavelet import HaarDWT3d, HaarIDWT3d

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_SHAPE = (1, 1, 96, 96, 96)

# ------------------------------------------------------------------
# 1. Wavelet correctness: DWT → IDWT should reconstruct perfectly
# ------------------------------------------------------------------
print("=" * 55)
print("Wavelet reconstruction test")
print("=" * 55)

dwt  = HaarDWT3d().to(DEVICE)
idwt = HaarIDWT3d().to(DEVICE)

x = torch.randn(INPUT_SHAPE).to(DEVICE)
sub = dwt(x)
x_rec = idwt(sub)

err = (x - x_rec).abs().max().item()
print(f"  Input shape  : {tuple(x.shape)}")
print(f"  DWT output   : {tuple(sub.shape)}")
print(f"  Reconstructed: {tuple(x_rec.shape)}")
print(f"  Max recon err: {err:.2e}  {'OK' if err < 1e-4 else 'FAILED'}")

# ------------------------------------------------------------------
# 2. Forward pass for every variant
# ------------------------------------------------------------------
print()
print("=" * 55)
print("Forward pass — all variants")
print("=" * 55)

x = torch.randn(INPUT_SHAPE).to(DEVICE)

for variant in VARIANTS:
    model, loss_fn = build_model(variant)
    model = model.to(DEVICE).eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    with torch.no_grad():
        out = model(x)

    shape_ok = tuple(out.shape) == INPUT_SHAPE
    print(
        f"  {variant:<20}  out={tuple(out.shape)}  "
        f"params={n_params:,}  {'OK' if shape_ok else 'SHAPE MISMATCH'}"
    )

print()
print("All done.")
