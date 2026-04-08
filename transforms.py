from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, NormalizeIntensityd, RandCropByPosNegLabeld,
    RandFlipd, EnsureTyped, ConcatItemsd, DeleteItemsd,
)

_MM = ["flair", "t1", "t2"]   # multimodal key names


# ------------------------------------------------------------------
# Single-modality (FLAIR only)
# ------------------------------------------------------------------

def get_train_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1, 1, 1),
                 mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1, neg=1, num_samples=4,
        ),
        RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
        EnsureTyped(keys=["image", "label"]),
    ])


def get_val_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1, 1, 1),
                 mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True),
        EnsureTyped(keys=["image", "label"]),
    ])


# ------------------------------------------------------------------
# Multi-modal (FLAIR + T1 + T2)
# Each modality is loaded and normalised independently, then
# concatenated along the channel dim into a single "image" tensor.
# ------------------------------------------------------------------

def get_train_transforms_mm():
    return Compose([
        LoadImaged(keys=_MM + ["label"]),
        EnsureChannelFirstd(keys=_MM + ["label"]),
        Orientationd(keys=_MM + ["label"], axcodes="RAS"),
        Spacingd(keys=_MM + ["label"], pixdim=(1, 1, 1),
                 mode=["bilinear", "bilinear", "bilinear", "nearest"]),
        NormalizeIntensityd(keys=_MM, nonzero=True),
        ConcatItemsd(keys=_MM, name="image", dim=0),  # [3, D, H, W]
        DeleteItemsd(keys=_MM),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1, neg=1, num_samples=4,
        ),
        RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
        EnsureTyped(keys=["image", "label"]),
    ])


def get_val_transforms_mm():
    return Compose([
        LoadImaged(keys=_MM + ["label"]),
        EnsureChannelFirstd(keys=_MM + ["label"]),
        Orientationd(keys=_MM + ["label"], axcodes="RAS"),
        Spacingd(keys=_MM + ["label"], pixdim=(1, 1, 1),
                 mode=["bilinear", "bilinear", "bilinear", "nearest"]),
        NormalizeIntensityd(keys=_MM, nonzero=True),
        ConcatItemsd(keys=_MM, name="image", dim=0),  # [3, D, H, W]
        DeleteItemsd(keys=_MM),
        EnsureTyped(keys=["image", "label"]),
    ])
