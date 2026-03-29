import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarDWT3d(nn.Module):
    """
    Single-level 3D Haar Discrete Wavelet Transform.

    Decomposes a volume into 8 frequency sub-bands using fixed orthonormal
    Haar filters. Registered as buffers — move to GPU without contributing
    to the parameter count.

    Sub-band order: LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH
      LLL = coarse approximation (smooth structure)
      *H* = high-pass axis → encodes edges / detail in that dimension

    Input:  [B, 1, D, H, W]
    Output: [B, 8, D/2, H/2, W/2]
    """

    def __init__(self):
        super().__init__()
        L = torch.tensor([1.0, 1.0]) / (2 ** 0.5)   # low-pass
        H = torch.tensor([1.0, -1.0]) / (2 ** 0.5)  # high-pass

        filters = []
        for fd in (L, H):
            for fh in (L, H):
                for fw in (L, H):
                    f3d = fd[:, None, None] * fh[None, :, None] * fw[None, None, :]
                    filters.append(f3d)

        weight = torch.stack(filters, dim=0).unsqueeze(1)  # [8, 1, 2, 2, 2]
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv3d(x, self.weight, stride=2, padding=0)


def _max_dwt_levels(dim: int) -> int:
    """
    Maximum number of DWT levels applicable to a spatial dimension.
    DWT requires an even input at each level, so we count how many times
    dim is divisible by 2.

    Examples: 96 → 5,  48 → 4,  24 → 3,  12 → 2,  6 → 1,  3 → 0
    """
    levels = 0
    while dim % 2 == 0:
        dim //= 2
        levels += 1
    return levels


class SubBandSE(nn.Module):
    """
    Squeeze-and-Excitation over the 8 DWT sub-band channels.

    Learns a per-sub-band importance weight via global average pooling and
    a two-layer bottleneck MLP (8 → 4 → 8) with sigmoid gating.

    Input/Output shape: [B, 8, D, H, W]  (unchanged)
    Parameters: 2 × (8 × 4) = 64  (no bias)
    """

    def __init__(self, channels: int = 8, reduction: int = 2):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        scale = self.squeeze(x).view(b, c)
        scale = self.excite(scale).view(b, c, 1, 1, 1)
        return x * scale


class WaveletPatchEmbed(nn.Module):
    """
    Variant A — single-level wavelet patch embedding.

    Replaces SwinUNETR's standard patch embedding with a single-level 3D
    Haar DWT followed by a learned 1×1×1 projection.

    Pipeline:
        [B, 1, D, H, W]
        → HaarDWT3d  → [B, 8, D/2, H/2, W/2]   (8 frequency sub-bands)
        → Conv3d(8→embed_dim, k=1)  → [B, embed_dim, D/2, H/2, W/2]

    Parameters: 8×embed_dim + embed_dim = 432 for embed_dim=48
    (identical to the baseline Conv3d(1→48, k=2, s=2) = 432 params)
    """

    def __init__(self, in_chans: int = 1, embed_dim: int = 48):
        super().__init__()
        self.dwt = HaarDWT3d()
        self.proj = nn.Conv3d(8, embed_dim, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dwt(x)   # [B, 8, D/2, H/2, W/2]
        return self.proj(x)


class HaarIDWT3d(nn.Module):
    """
    Single-level 3D Haar Inverse Discrete Wavelet Transform.

    Exact inverse of HaarDWT3d — Haar filters are orthonormal (W^T W = I),
    so IDWT = DWT^T = ConvTranspose3d with the same filter weights.

    Input:  [B, 8, D/2, H/2, W/2]
    Output: [B, 1, D,   H,   W  ]
    """

    def __init__(self):
        super().__init__()
        L = torch.tensor([1.0, 1.0]) / (2 ** 0.5)
        H = torch.tensor([1.0, -1.0]) / (2 ** 0.5)

        filters = []
        for fd in (L, H):
            for fh in (L, H):
                for fw in (L, H):
                    f3d = fd[:, None, None] * fh[None, :, None] * fw[None, None, :]
                    filters.append(f3d)

        weight = torch.stack(filters, dim=0).unsqueeze(1)  # [8, 1, 2, 2, 2]
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv_transpose3d(x, self.weight, stride=2)


class WaveletSkipRefinement(nn.Module):
    """
    Variant B — frequency-aware refinement of a decoder skip connection.

    Each feature channel is decomposed independently via the 3D Haar DWT
    into 8 frequency sub-bands. A single SubBandSE block recalibrates
    sub-band importance (e.g. suppressing HHH noise, amplifying edge bands).
    The signal is exactly reconstructed via the IDWT and added back as a
    residual.

    Pipeline per channel:
        [B, C, D, H, W]
        → reshape [B*C, 1, D, H, W]
        → HaarDWT3d   → [B*C, 8, D/2, H/2, W/2]
        → SubBandSE   → frequency recalibration
        → HaarIDWT3d  → [B*C, 1, D, H, W]
        → reshape [B, C, D, H, W]
        → + skip  (residual)

    Parameters: 64 (SE only — DWT/IDWT are fixed buffers)
    """

    def __init__(self):
        super().__init__()
        self.dwt = HaarDWT3d()
        self.se = SubBandSE(channels=8, reduction=2)
        self.idwt = HaarIDWT3d()

    def forward(self, skip: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = skip.shape
        x = skip.reshape(B * C, 1, D, H, W)
        x = self.dwt(x)
        x = self.se(x)
        x = self.idwt(x)
        x = x.reshape(B, C, D, H, W)
        return x + skip


class WaveletSkipDecoder(nn.Module):
    """
    Wraps a SwinUNETR UnetrUpBlock to apply WaveletSkipRefinement on the
    skip connection before it is concatenated with the upsampled features.

    Forward signature mirrors UnetrUpBlock: forward(inp, skip)
    """

    def __init__(self, decoder_block: nn.Module):
        super().__init__()
        self.decoder = decoder_block
        self.refine = WaveletSkipRefinement()

    def forward(self, inp: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.decoder(inp, self.refine(skip))


class WaveletPatchEmbedSE(nn.Module):
    """
    Variant A+ — single-level wavelet patch embedding with sub-band attention.

    Extends WaveletPatchEmbed by adding a SubBandSE block between the DWT
    and the projection, allowing the network to learn which of the 8
    frequency sub-bands are most informative before mixing them into tokens.

    Pipeline:
        [B, 1, D, H, W]
        → HaarDWT3d  → [B, 8, D/2, H/2, W/2]
        → SubBandSE  → per-sub-band recalibration
        → Conv3d(8→embed_dim, k=1)  → [B, embed_dim, D/2, H/2, W/2]

    Parameters: 64 (SE) + 432 (proj) = 496
    """

    def __init__(self, in_chans: int = 1, embed_dim: int = 48):
        super().__init__()
        self.dwt = HaarDWT3d()
        self.se = SubBandSE(channels=8, reduction=2)
        self.proj = nn.Conv3d(8, embed_dim, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dwt(x)   # [B, 8, D/2, H/2, W/2]
        x = self.se(x)    # recalibrate sub-band importance
        return self.proj(x)


class WaveletPatchEmbedML(nn.Module):
    """
    Variant A-HL — multi-level wavelet patch embedding with full LLL decomposition.

    Applies the 3D Haar DWT recursively to the LLL (coarse approximation)
    sub-band for `levels` iterations, fully decomposing the volume down to
    its base coefficient. For 96×96×96 patches, levels=5 reduces LLL to 3³.

    Sub-bands from every level are upsampled back to the level-1 spatial
    resolution (D/2 × H/2 × W/2) and recalibrated by a dedicated per-level
    SubBandSE block. All recalibrated bands are concatenated and projected
    to embed_dim, giving the Swin Transformer a multi-resolution frequency
    view at the tokenisation stage:

        level 1 — fine detail bands at D/2    (original token resolution)
        level 2 — mid-scale bands at D/4  → upsampled to D/2
        level 3 — coarse bands at D/8     → upsampled to D/2
        ...
        level L — base approximation at D/2ᴸ  → upsampled to D/2

    Pipeline:
        [B, 1, D, H, W]
        → DWT₁ → [B, 8, D/2, ...]   SE₁ recalibrates
        → DWT₂ → [B, 8, D/4, ...]   SE₂ recalibrates → upsample to D/2
        ...
        → DWT_L → [B, 8, D/2ᴸ, ...] SE_L recalibrates → upsample to D/2
        → concat → [B, 8L, D/2, H/2, W/2]
        → Conv3d(8L→embed_dim, k=1) → [B, embed_dim, D/2, H/2, W/2]

    Parameters (levels=5, embed_dim=48):
        SE:   5 × 64  = 320
        proj: 40×48+48 = 1968
        total: 2288
    """

    def __init__(self, in_chans: int = 1, embed_dim: int = 48, levels: int = 5):
        super().__init__()
        self.levels = levels
        self.dwt = HaarDWT3d()
        self.se_blocks = nn.ModuleList(
            [SubBandSE(channels=8, reduction=2) for _ in range(levels)]
        )
        self.proj = nn.Conv3d(8 * levels, embed_dim, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cap to what spatial dims actually support
        D, H, W = x.shape[2], x.shape[3], x.shape[4]
        actual_levels = min(
            self.levels,
            _max_dwt_levels(D),
            _max_dwt_levels(H),
            _max_dwt_levels(W),
        )

        target_size = None  # level-1 token resolution: D/2 × H/2 × W/2
        all_bands = []
        approx = x          # raw input [B, 1, D, H, W]

        for i in range(actual_levels):
            sub = self.dwt(approx)        # [B, 8, D/2^(i+1), ...]
            sub = self.se_blocks[i](sub)  # per-level SE recalibration

            if i == 0:
                target_size = sub.shape[2:]
                all_bands.append(sub)
            else:
                all_bands.append(
                    F.interpolate(sub, size=target_size,
                                  mode="trilinear", align_corners=False)
                )

            approx = sub[:, :1]           # LLL → input for next level

        x = torch.cat(all_bands, dim=1)   # [B, 8*actual_levels, D/2, H/2, W/2]
        return self.proj(x)
