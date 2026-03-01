"""
model.py — Model factory for all ablation variants.

Usage:
    model, loss_fn = build_model("baseline")
    model, loss_fn = build_model("wavelet_a")
    model, loss_fn = build_model("wavelet_b")
    model, loss_fn = build_model("wavelet_ab")
    model, loss_fn = build_model("wavelet_ab_freq")

build_model returns:
    model   — nn.Module ready for .to(device)
    loss_fn — nn.Module loss function matching the variant
"""

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR
from monai.losses import DiceLoss

from wavelet import WaveletPatchEmbed, WaveletSkipRefinement
from loss import FrequencyWeightedDiceLoss

VARIANTS = ("baseline", "wavelet_a", "wavelet_b", "wavelet_ab", "wavelet_ab_freq")


class WaveSwinUNETR_B(SwinUNETR):
    """
    SwinUNETR + wavelet skip-connection refinement (Variant B).

    Adds a WaveletSkipRefinement module at each of the four encoder
    skip connections (enc0..enc3) before they are passed to the decoder.
    Only 8 × 4 = 32 extra trainable scalar parameters.
    """

    def __init__(self, feature_size: int = 48, **kwargs) -> None:
        super().__init__(feature_size=feature_size, **kwargs)
        fs = feature_size
        self.wsr0 = WaveletSkipRefinement(fs)       # enc0: [B, 48,  96, 96, 96]
        self.wsr1 = WaveletSkipRefinement(fs)       # enc1: [B, 48,  48, 48, 48]
        self.wsr2 = WaveletSkipRefinement(fs * 2)   # enc2: [B, 96,  24, 24, 24]
        self.wsr3 = WaveletSkipRefinement(fs * 4)   # enc3: [B, 192, 12, 12, 12]

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            self._check_input_size(x_in.shape[2:])

        hidden_states_out = self.swinViT(x_in, self.normalize)

        enc0 = self.wsr0(self.encoder1(x_in))
        enc1 = self.wsr1(self.encoder2(hidden_states_out[0]))
        enc2 = self.wsr2(self.encoder3(hidden_states_out[1]))
        enc3 = self.wsr3(self.encoder4(hidden_states_out[2]))
        dec4 = self.encoder10(hidden_states_out[4])

        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out  = self.decoder1(dec0, enc0)
        return self.out(out)


def _base_swinunetr(in_channels: int, out_channels: int, feature_size: int) -> SwinUNETR:
    return SwinUNETR(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        use_checkpoint=True,
    )


def build_model(
    variant: str,
    feature_size: int = 48,
    in_channels: int = 1,
    out_channels: int = 1,
):
    """
    Build model and matching loss function for the given variant name.

    Args:
        variant:      One of VARIANTS (see module-level constant).
        feature_size: SwinUNETR feature size. Default 48.
        in_channels:  Input modality channels. Default 1 (FLAIR only).
        out_channels: Segmentation classes. Default 1 (binary).

    Returns:
        (model, loss_fn): Both are nn.Module instances.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {VARIANTS}")

    # ── Loss ────────────────────────────────────────────────────────────────
    if variant == "wavelet_ab_freq":
        loss_fn = FrequencyWeightedDiceLoss(lambda_freq=0.5)
    else:
        loss_fn = DiceLoss(sigmoid=True)

    # ── Model ───────────────────────────────────────────────────────────────
    if variant == "baseline":
        model = _base_swinunetr(in_channels, out_channels, feature_size)

    elif variant == "wavelet_a":
        model = _base_swinunetr(in_channels, out_channels, feature_size)
        # Post-construction patch embed swap — no subclassing needed
        model.swinViT.patch_embed = WaveletPatchEmbed(
            in_chans=in_channels, embed_dim=feature_size
        )

    elif variant == "wavelet_b":
        model = WaveSwinUNETR_B(
            feature_size=feature_size,
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            use_checkpoint=True,
        )

    elif variant in ("wavelet_ab", "wavelet_ab_freq"):
        model = WaveSwinUNETR_B(
            feature_size=feature_size,
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            use_checkpoint=True,
        )
        # Also replace patch embed (Variant A on top of Variant B)
        model.swinViT.patch_embed = WaveletPatchEmbed(
            in_chans=in_channels, embed_dim=feature_size
        )

    return model, loss_fn
