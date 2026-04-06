from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, NormalizeIntensityd, RandCropByPosNegLabeld,
    RandFlipd, EnsureTyped
)

def get_train_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1,1,1),
                 mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96,96,96),
            pos=1, neg=1, num_samples=4
        ),
        RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
        EnsureTyped(keys=["image", "label"]),
    ])


def get_val_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1,1,1),
                 mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True),
        EnsureTyped(keys=["image", "label"]),
    ])
