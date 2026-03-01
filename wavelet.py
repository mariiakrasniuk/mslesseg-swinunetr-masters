"""
wavelet.py — Reusable 3D Haar wavelet decomposition primitives.

All modules work on arbitrary [B, C, D, H, W] tensors and have no
dependency beyond PyTorch. Fixed (non-learned) filters mean zero extra
trainable parameters for DWT/IDWT; only the projection in
WaveletPatchEmbed and the 8 scalars in WaveletSkipRefinement are learned.

Sub-band ordering (8 bands per input channel):
    index 0: LLL  (all low-pass  — coarse anatomy)
    index 1: LLH  (low D, low H, high W)
    index 2: LHL  (low D, high H, low W)
    index 3: LHH  (low D, high H, high W)
    index 4: HLL  (high D, low H, low W)
    index 5: HLH  (high D, low H, high W)
    index 6: HHL  (high D, high H, low W)
    index 7: HHH  (all high-pass — fine edges / lesion boundaries)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _haar_filters_3d() -> torch.Tensor:
    """
    Build the 8 fixed 3D Haar filters, each of shape [1, 1, 2, 2, 2].
    Returns a tensor of shape [8, 1, 2, 2, 2].
    """
    s = 1.0 / math.sqrt(2)
    L = torch.tensor([s,  s], dtype=torch.float32)
    H = torch.tensor([s, -s], dtype=torch.float32)

    filters = []
    for f0 in (L, H):
        for f1 in (L, H):
            for f2 in (L, H):
                # outer product of three 1D filters → [2, 2, 2]
                k = torch.einsum("i,j,k->ijk", f0, f1, f2)
                filters.append(k.unsqueeze(0).unsqueeze(0))  # [1,1,2,2,2]

    return torch.cat(filters, dim=0)  # [8, 1, 2, 2, 2]


class HaarDWT3d(nn.Module):
    """
    3D Discrete Haar Wavelet Transform.

    Input:  [B, C, D, H, W]
    Output: [B, 8*C, D//2, H//2, W//2]

    The 8 sub-bands per input channel are concatenated along the channel
    dimension. All weights are fixed (requires_grad=False).

    Spatial dimensions must be even; pad before calling if necessary.
    """

    def __init__(self) -> None:
        super().__init__()
        # shape [8, 1, 2, 2, 2] — one filter set, used with groups=C
        self.register_buffer("weight", _haar_filters_3d())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        # Repeat the 8 filters for each input channel → [8C, 1, 2, 2, 2]
        w = self.weight.repeat(C, 1, 1, 1, 1)
        # groups=C applies each set of 8 filters independently per channel
        out = F.conv3d(x, w, stride=2, groups=C)  # [B, 8C, D//2, H//2, W//2]
        return out


class HaarIDWT3d(nn.Module):
    """
    3D Inverse Discrete Haar Wavelet Transform.

    Input:  [B, 8*C, D//2, H//2, W//2]
    Output: [B, C, D, H, W]

    Perfect reconstruction: IDWT(DWT(x)) == x  (up to float32 precision).
    All weights are fixed (requires_grad=False).
    """

    def __init__(self) -> None:
        super().__init__()
        # For the inverse, the Haar filters are self-adjoint (L=L^T, H=H^T)
        # so the inverse uses the same filter bank as a transposed convolution.
        self.register_buffer("weight", _haar_filters_3d())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C8, D2, H2, W2 = x.shape
        assert C8 % 8 == 0, f"Expected channels divisible by 8, got {C8}"
        C = C8 // 8
        # Repeat filters → [8C, 1, 2, 2, 2]
        w = self.weight.repeat(C, 1, 1, 1, 1)
        out = F.conv_transpose3d(x, w, stride=2, groups=C)  # [B, C, D, H, W]
        return out


class WaveletPatchEmbed(nn.Module):
    """
    Wavelet-based patch embedding for Variant A.

    Replaces the standard strided-conv PatchEmbed inside SwinTransformer.
    Produces identical output shape: [B, embed_dim, D//2, H//2, W//2].

    The DWT decomposes the input into 8 frequency sub-bands (0 extra
    trainable params), then a 1×1×1 conv projects to embed_dim.

    New trainable parameters: in_chans * 8 * embed_dim  (e.g. 1*8*48 = 384)
    """

    def __init__(self, in_chans: int, embed_dim: int) -> None:
        super().__init__()
        self.dwt = HaarDWT3d()
        self.proj = nn.Conv3d(in_chans * 8, embed_dim, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dwt(x)       # [B, 8*in_chans, D//2, H//2, W//2]
        x = self.proj(x)      # [B, embed_dim,  D//2, H//2, W//2]
        return x


class WaveletSkipRefinement(nn.Module):
    """
    Frequency-selective refinement of a skip connection for Variant B.

    Applies DWT to decompose the feature map into 8 frequency sub-bands,
    re-weights them with 8 learned scalars (softmax-normalised), reconstructs
    with IDWT, and adds the result residually to the original.

    New trainable parameters: 8 scalars per instance.
    Input and output shape: [B, C, D, H, W]  (identical)
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.dwt = HaarDWT3d()
        self.idwt = HaarIDWT3d()
        # 8 per-subband scalar weights, initialised to uniform emphasis
        self.band_weights = nn.Parameter(torch.zeros(8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        sub = self.dwt(x)                          # [B, 8C, D//2, H//2, W//2]

        # Reshape to [B, 8, C, D//2, H//2, W//2] for per-band weighting
        sub = sub.view(B, 8, C, D // 2, H // 2, W // 2)
        w = torch.softmax(self.band_weights, dim=0)  # [8] — sums to 1, all positive
        sub = sub * w.view(1, 8, 1, 1, 1, 1)
        sub = sub.view(B, 8 * C, D // 2, H // 2, W // 2)

        refined = self.idwt(sub)                   # [B, C, D, H, W]
        return x + refined
