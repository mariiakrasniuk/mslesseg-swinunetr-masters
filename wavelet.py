import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarDWT3d(nn.Module):
    """
    Single-level 3D Haar Discrete Wavelet Transform.

    Decomposes a volume into 8 frequency sub-bands using fixed
    (non-trainable) Haar filters. Registered as buffers so they
    move to GPU with the model without contributing to parameters.

    Input:  [B, 1, D, H, W]
    Output: [B, 8, D/2, H/2, W/2]

    Sub-band order: LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH
    """

    def __init__(self):
        super().__init__()

        # 1D Haar filters (orthonormal)
        L = torch.tensor([1.0, 1.0]) / (2 ** 0.5)   # low-pass:  average
        H = torch.tensor([1.0, -1.0]) / (2 ** 0.5)  # high-pass: difference

        # Build all 8 outer-product 3D filters [8, 1, 2, 2, 2]
        filters = []
        for fd in (L, H):
            for fh in (L, H):
                for fw in (L, H):
                    # outer product: depth x height x width
                    f3d = fd[:, None, None] * fh[None, :, None] * fw[None, None, :]
                    filters.append(f3d)                   # [2, 2, 2]

        # Stack to [8, 1, 2, 2, 2]  (out_channels, in_channels/groups, kD, kH, kW)
        weight = torch.stack(filters, dim=0).unsqueeze(1)
        self.register_buffer("weight", weight)            # not a parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, D, H, W]
        return F.conv3d(x, self.weight, stride=2, padding=0)
        # output: [B, 8, D/2, H/2, W/2]


class WaveletPatchEmbed(nn.Module):
    """
    Drop-in replacement for SwinUNETR's patch_embed.

    Pipeline:
        HaarDWT3d  →  [B, 8, D/2, H/2, W/2]
        Conv3d(8 → embed_dim, kernel=1)  →  [B, embed_dim, D/2, H/2, W/2]

    Parameter count: 8 * embed_dim + embed_dim = 432 for embed_dim=48
    (identical to the baseline Conv3d(1→48, kernel=2, stride=2) = 432 params)
    """

    def __init__(self, in_chans: int = 1, embed_dim: int = 48):
        super().__init__()
        self.dwt = HaarDWT3d()
        # 1x1x1 learned projection: mixes frequency sub-bands into token dim
        self.proj = nn.Conv3d(8, embed_dim, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dwt(x)       # [B, 8, D/2, H/2, W/2]
        x = self.proj(x)      # [B, embed_dim, D/2, H/2, W/2]
        return x
