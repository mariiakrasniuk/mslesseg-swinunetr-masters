# Variant A — Wavelet-Based Patch Embedding

## Overview

Variant A replaces the standard patch embedding in SwinUNETR with a
3D Haar wavelet decomposition followed by a learned 1×1×1 projection.
It is a single-file, drop-in modification with zero extra parameters
compared to the baseline.

---

## Where the Code Lives

| File | What it contains |
|---|---|
| `wavelet.py` | `HaarDWT3d`, `WaveletPatchEmbed` — the math |
| `model.py` lines 105–110 | The swap into SwinUNETR |
| `train.py` | `--variant wavelet_a` to select it |
| `final_test_evaluation.py` | `--variant wavelet_a` to evaluate |

---

## Full Data Flow

```
Input MRI volume
[B=1, C=1, D=96, H=96, W=96]
        |
        |  ── BASELINE patch embed (replaced) ──────────────────────────
        |  Conv3d(in=1, out=48, kernel=2, stride=2)  ← learned
        |  Output: [B, 48, 48, 48, 48]
        |  ────────────────────────────────────────────────────────────
        |
        |  ── VARIANT A patch embed ──────────────────────────────────
        ↓
  [ HaarDWT3d ]                          wavelet.py : HaarDWT3d
        |
        |  Applies 8 fixed 3D Haar filters via F.conv3d(stride=2, groups=1)
        |  Each filter is the outer product of three 1D filters:
        |    L = [+1/√2, +1/√2]   low-pass  (averages)
        |    H = [+1/√2, -1/√2]   high-pass (differences)
        |
        |  8 combinations: LLL LLH LHL LHH HLL HLH HHL HHH
        |
        ↓
  [B, 8, 48, 48, 48]   — 8 frequency sub-bands, spatial dims halved
        |
        |  Sub-band meaning:
        |    index 0  LLL  coarse anatomy (all low-pass)
        |    index 1  LLH  edges along Width
        |    index 2  LHL  edges along Height
        |    index 3  LHH  edges along Height + Width
        |    index 4  HLL  edges along Depth
        |    index 5  HLH  edges along Depth + Width
        |    index 6  HHL  edges along Depth + Height
        |    index 7  HHH  fine edges in all directions — lesion boundaries
        |
  [ Conv3d(8→48, kernel=1) ]             wavelet.py : WaveletPatchEmbed
        |
        |  1×1×1 convolution — learned projection that decides how much
        |  weight to give each frequency sub-band in the embedding.
        |
        ↓
  [B, 48, 48, 48, 48]                    same shape as baseline output
        |  ────────────────────────────────────────────────────────────
        |
        ↓
  [ SwinTransformer stages ]             unchanged from baseline
        |  window-based self-attention across 4 resolution levels
        ↓
  [ U-Net decoder ]                      unchanged from baseline
        |  skip connections + upsampling
        ↓
  Segmentation logits
  [B, 1, 96, 96, 96]
```

---

## Decomposition Level

**Single-level only.** The LLL sub-band is NOT further decomposed.

This is a deliberate constraint:

- SwinUNETR's original patch embed uses `stride=2` → output spatial dims = input / 2
- One level of Haar DWT also uses `stride=2` → same spatial dims
- The Swin transformer's window sizes are fixed and designed for this resolution
- Multi-level decomposition would halve the spatial dims again, breaking the architecture

One level of decomposition gives 8 sub-bands, which is sufficient to
separate coarse anatomy (LLL) from fine lesion boundaries (HHH).

---

## How the Swap Works (`model.py` lines 105–110)

```python
elif variant == "wavelet_a":
    model = _base_swinunetr(in_channels, out_channels, feature_size)
    # Post-construction patch embed swap — no subclassing needed
    model.swinViT.patch_embed = WaveletPatchEmbed(
        in_chans=in_channels, embed_dim=feature_size
    )
```

SwinUNETR is built normally first, then its `patch_embed` attribute is
replaced with `WaveletPatchEmbed`. Everything else in the model is
identical to the baseline. No subclassing, no architecture changes.

---

## Parameter Count

| Component | Baseline | Variant A |
|---|---|---|
| Patch embed | `Conv3d(1→48, k=2)` = 432 params | `Conv3d(8→48, k=1)` = 432 params |
| DWT filters | — | 0 (fixed buffers, not trained) |
| Total model | 62,186,659 | 62,186,659 |

**Identical parameter budget.** The wavelet filters are registered as
`nn.Module` buffers (not parameters), so they move to GPU with the model
but are never updated by the optimizer.

---

## Running Variant A

**Train:**
```bash
python train.py --variant wavelet_a
```

**Evaluate on test set:**
```bash
python final_test_evaluation.py --variant wavelet_a
```

**Plot training curves:**
```bash
python plot_eval_curves.py --run_name wavelet_a
```

**Smoke test (verify forward pass):**
```bash
python forward_pass.py
```

Outputs:
- `best_wavelet_a.pth` — best model weights (by val Dice)
- `training_history_wavelet_a.pth` — loss and Dice per epoch
- `loss_curves_wavelet_a.png`
- `dice_curves_wavelet_a.png`

---

## Why Wavelets for MS Lesion Segmentation

MS lesions are high-frequency anomalies in a low-frequency background:
- Healthy brain tissue → smooth, slowly varying → **low-frequency** (LLL band)
- Lesion boundaries → sharp contrast edges → **high-frequency** (HHH band)

The baseline Conv3d patch embed mixes all frequencies together before
the transformer processes them, and must learn to separate them from
data alone. `WaveletPatchEmbed` provides this separation explicitly and
mathematically, with no additional parameters.

The 1×1×1 projection then learns which frequency sub-bands are most
informative for each of the 48 embedding dimensions — effectively
learning a frequency-aware token representation.
