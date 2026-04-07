import torch.nn as nn
from monai.networks.nets import SwinUNETR

from wavelet import (
    WaveletPatchEmbed,
    WaveletPatchEmbedML,
    SWTSkipInjector,
    _max_dwt_levels,  # used by WaveletPatchEmbedML forward
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


class SwinUNETRWithSWTSkips(SwinUNETR):
    """
    SwinUNETR with SWT frequency injection into encoder skip connections.

    For each of the four encoder skip connections (enc0–enc3), an
    SWTSkipInjector enriches the feature map with multi-frequency information
    derived from the original network input at the matching spatial scale.
    The injection is purely residual — the original skip features are
    preserved and frequency cues are added on top.

    Parameters
    ----------
    swt_wavelet : str
        Wavelet family used by all four SWTSkipInjectors ('haar', 'db2', 'sym4').
    feature_size : int
        SwinUNETR feature_size (default 48).  Skip channel counts are derived
        from this: enc0=feature_size, enc1=feature_size, enc2=feature_size*2,
        enc3=feature_size*4.
    """

    def __init__(self, *args, swt_wavelet: str = "haar", feature_size: int = 48,
                 **kwargs):
        super().__init__(*args, feature_size=feature_size, **kwargs)
        self.swt_skip0 = SWTSkipInjector(feature_size,     swt_wavelet)  # enc0: 48ch, full res
        self.swt_skip1 = SWTSkipInjector(feature_size,     swt_wavelet)  # enc1: 48ch, D/2
        self.swt_skip2 = SWTSkipInjector(feature_size * 2, swt_wavelet)  # enc2: 96ch, D/4
        self.swt_skip3 = SWTSkipInjector(feature_size * 4, swt_wavelet)  # enc3: 192ch, D/8

    def forward(self, x_in):
        hidden_states_out = self.swinViT(x_in, self.normalize)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])

        # Inject SWT frequency information into encoder skip connections
        enc0 = self.swt_skip0(x_in, enc0)
        enc1 = self.swt_skip1(x_in, enc1)
        enc2 = self.swt_skip2(x_in, enc2)
        enc3 = self.swt_skip3(x_in, enc3)

        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out  = self.decoder1(dec0, enc0)
        return self.out(out)


def build_model(
    variant: str,
    in_channels: int = 1,
    out_channels: int = 1,
    feature_size: int = 48,
    use_checkpoint: bool = False,
    roi_size: int = 96,
    wavelet: str = "haar",
    levels: int = 1,
    swt_wavelet: str = "haar",
) -> nn.Module:
    """
    Factory that returns the model for a given variant name.

    Variants
    --------
    baseline
        Standard MONAI SwinUNETR, unchanged.  wavelet/levels are ignored.

    wavelet_a
        Patch embedding replaced by a single-level 3D Haar DWT followed by
        a 1x1x1 learned projection. Parameter count identical to baseline (432).

    wavelet_ml  [ablation variant]
        Patch embedding replaced by WaveletPatchEmbedML parameterised by the
        `wavelet` and `levels` arguments. Used for the family × depth ablation.

        Supported wavelet families : 'haar', 'db2', 'sym4'
        Supported levels           : 1, 2, 3

        Parameter count (embed_dim=48):
            levels=1 :  64  (SE)  + 432 (proj) =  496
            levels=2 : 128  (SE)  + 816 (proj) =  944
            levels=3 : 192  (SE)  + 1200 (proj) = 1392

    wavelet_swt_skip
        SwinUNETR with SWT frequency injection into all four encoder skip
        connections.  Standard Conv3d patch embedding is kept unchanged.
        Controlled by `swt_wavelet`.

    wavelet_ml_swt
        Combines WaveletPatchEmbedML patch embedding (controlled by `wavelet`
        and `levels`) with SWT skip injection (controlled by `swt_wavelet`).
        Multi-frequency cues are present at both the tokenisation stage and in
        the skip connections.
    """
    if variant == "baseline":
        return _base_swinunetr(in_channels, out_channels, feature_size, use_checkpoint)

    elif variant == "wavelet_a":
        model = _base_swinunetr(in_channels, out_channels, feature_size, use_checkpoint)
        model.swinViT.patch_embed = WaveletPatchEmbed(
            in_chans=in_channels, embed_dim=feature_size
        )
        return model

    elif variant == "wavelet_ml":
        model = _base_swinunetr(in_channels, out_channels, feature_size, use_checkpoint)
        model.swinViT.patch_embed = WaveletPatchEmbedML(
            in_chans=in_channels, embed_dim=feature_size,
            levels=levels, wavelet=wavelet,
        )
        return model

    elif variant == "wavelet_swt_skip":
        model = SwinUNETRWithSWTSkips(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
            swt_wavelet=swt_wavelet,
        )
        return model

    elif variant == "wavelet_ml_swt":
        model = SwinUNETRWithSWTSkips(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
            swt_wavelet=swt_wavelet,
        )
        model.swinViT.patch_embed = WaveletPatchEmbedML(
            in_chans=in_channels, embed_dim=feature_size,
            levels=levels, wavelet=wavelet,
        )
        return model

    else:
        raise ValueError(
            f"Unknown variant '{variant}'. Choose from: "
            f"baseline, wavelet_a, wavelet_ml, wavelet_swt_skip, wavelet_ml_swt"
        )
