import torch.nn as nn
from monai.networks.nets import SwinUNETR
from wavelet import WaveletPatchEmbed


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
) -> nn.Module:
    """
    Factory that returns the model for a given variant name.

    Variants
    --------
    baseline   : Standard SwinUNETR — unchanged from MONAI.
    wavelet_a  : SwinUNETR with its patch_embed replaced by
                 WaveletPatchEmbed (HaarDWT3d + 1x1x1 Conv projection).
                 Same parameter count as baseline.
    """
    if variant == "baseline":
        return _base_swinunetr(in_channels, out_channels, feature_size, use_checkpoint)

    elif variant == "wavelet_a":
        model = _base_swinunetr(in_channels, out_channels, feature_size, use_checkpoint)
        # Post-construction swap — no subclassing needed
        model.swinViT.patch_embed = WaveletPatchEmbed(
            in_chans=in_channels, embed_dim=feature_size
        )
        return model

    else:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: baseline, wavelet_a")
