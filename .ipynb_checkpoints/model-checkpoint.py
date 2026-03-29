import torch.nn as nn
from monai.networks.nets import SwinUNETR

from wavelet import (
    WaveletPatchEmbed,
    WaveletPatchEmbedSE,
    WaveletPatchEmbedML,
    WaveletSkipDecoder,
    _max_dwt_levels,  # used by wavelet_a_higher_level
)


def _base_swinunetr(in_channels: int, out_channels: int, feature_size: int,
                    use_checkpoint: bool) -> nn.Module:
    return SwinUNETR(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        use_checkpoint=use_checkpoint,
    )


def build_model(
    variant: str,
    in_channels: int = 1,
    out_channels: int = 1,
    feature_size: int = 48,
    use_checkpoint: bool = False,
    roi_size: int = 96,
) -> nn.Module:
    """
    Factory that returns the model for a given variant name.

    Variants
    --------
    baseline
        Standard MONAI SwinUNETR, unchanged.

    wavelet_a
        Patch embedding replaced by a single-level 3D Haar DWT followed by
        a 1x1x1 learned projection. No SE attention. Parameter count at
        patch_embed identical to baseline (432).

    wavelet_a_plus
        Same as wavelet_a with a SubBandSE block inserted between the DWT
        and the projection. Learns per-sub-band importance weights.
        Extra parameters: 64.

    wavelet_b
        Combines wavelet patch embedding (as in wavelet_a) with frequency-aware
        skip refinement on two mid-scale decoder stages. Patch embedding uses
        HaarDWT3d followed by a 1×1×1 projection (432 params, param-parity with
        baseline). Additionally, decoder2 (48³ skip) and decoder3 (24³ skip) are
        wrapped with WaveletSkipRefinement: each skip is decomposed channel-wise
        via DWT into 8 sub-bands, recalibrated by a SubBandSE block, and
        reconstructed via IDWT with a residual connection. decoder1 (96³ skip,
        full-res CNN encoder) is left unchanged to avoid excessive memory cost.
        Coarser decoders (decoder4–5, ≤12³) are also left unchanged.
        Extra parameters over baseline: 128 (2 × 64 SE; DWT/IDWT are fixed buffers).

    wavelet_a_higher_level
        Patch embedding replaced by WaveletPatchEmbedML. The DWT is applied
        recursively to the LLL sub-band for the maximum number of levels
        supported by the patch size (5 levels for roi_size=96, reducing LLL
        from 96^3 to 3^3). Sub-bands from all levels are upsampled to the
        level-1 token resolution, recalibrated by a per-level SE block,
        concatenated, and projected to embed_dim.
        Extra parameters: 1856.
    """
    if variant == "baseline":
        return _base_swinunetr(in_channels, out_channels, feature_size, use_checkpoint)

    elif variant == "wavelet_a":
        model = _base_swinunetr(in_channels, out_channels, feature_size, use_checkpoint)
        model.swinViT.patch_embed = WaveletPatchEmbed(
            in_chans=in_channels, embed_dim=feature_size
        )
        return model

    elif variant == "wavelet_a_plus":
        model = _base_swinunetr(in_channels, out_channels, feature_size, use_checkpoint)
        model.swinViT.patch_embed = WaveletPatchEmbedSE(
            in_chans=in_channels, embed_dim=feature_size
        )
        return model

    elif variant == "wavelet_b":
        model = _base_swinunetr(in_channels, out_channels, feature_size, use_checkpoint)
        model.swinViT.patch_embed = WaveletPatchEmbed(
            in_chans=in_channels, embed_dim=feature_size
        )
        # decoder1 skip is 96³ (full-res CNN encoder) — too large for channel-wise DWT.
        # Wrap decoder2 (48³ skip) and decoder3 (24³ skip): enough spatial resolution
        # for meaningful sub-band attention while remaining memory-efficient.
        for name in ("decoder3", "decoder2"):
            setattr(model, name, WaveletSkipDecoder(getattr(model, name)))
        return model

    elif variant == "wavelet_a_higher_level":
        model = _base_swinunetr(in_channels, out_channels, feature_size, use_checkpoint)
        levels = _max_dwt_levels(roi_size)  # 5 for roi_size=96
        model.swinViT.patch_embed = WaveletPatchEmbedML(
            in_chans=in_channels, embed_dim=feature_size, levels=levels
        )
        return model

    else:
        raise ValueError(
            f"Unknown variant '{variant}'. Choose from: "
            f"baseline, wavelet_a, wavelet_a_plus, wavelet_b, wavelet_a_higher_level"
        )
