"""
loss.py — Loss functions for wavelet-augmented segmentation.

FrequencyWeightedDiceLoss combines standard Dice with a boundary-aware
MSE term. Boundaries are detected by applying DWT to the ground truth
mask and measuring high-frequency energy — no manual edge operators needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss

from wavelet import HaarDWT3d


class FrequencyWeightedDiceLoss(nn.Module):
    """
    Dice loss + frequency-weighted boundary loss.

    The boundary weight map is derived from the HF sub-bands of the
    ground truth mask via DWT, so it's cheap, parameter-free, and
    responds to where annotators drew sharp lesion boundaries.

    Args:
        lambda_freq: weight of the boundary MSE term relative to Dice.
                     0.0 reduces to plain DiceLoss. Default: 0.5
    """

    def __init__(self, lambda_freq: float = 0.5) -> None:
        super().__init__()
        self.lambda_freq = lambda_freq
        self.dice = DiceLoss(sigmoid=True)
        self.dwt = HaarDWT3d()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # --- Standard Dice component ---
        dice_loss = self.dice(logits, target)

        if self.lambda_freq == 0.0:
            return dice_loss

        # --- Boundary weight map from HF sub-bands of target ---
        with torch.no_grad():
            # target: [B, 1, D, H, W]
            sub = self.dwt(target)           # [B, 8, D//2, H//2, W//2]
            # Sub-bands 1..7 are high-frequency (index 0 is LLL)
            hf = sub[:, 1:, :, :, :]        # [B, 7, D//2, H//2, W//2]
            # Sum absolute HF energy per voxel → [B, 1, D//2, H//2, W//2]
            hf_energy = hf.abs().sum(dim=1, keepdim=True)
            # Upsample back to full resolution
            boundary_map = F.interpolate(
                hf_energy, size=target.shape[2:], mode="trilinear", align_corners=False
            )
            # Normalise to [0, 1]
            b_max = boundary_map.amax(dim=(2, 3, 4), keepdim=True).clamp(min=1e-8)
            boundary_map = boundary_map / b_max   # [B, 1, D, H, W]

        # --- Frequency-weighted MSE component ---
        probs = torch.sigmoid(logits)
        weighted_mse = ((probs - target) ** 2 * (1.0 + boundary_map)).mean()

        return dice_loss + self.lambda_freq * weighted_mse
