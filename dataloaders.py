from torch.utils.data import DataLoader
from monai.data import Dataset
from build_datalist import build_train_list
from transforms import get_train_transforms, get_val_transforms

def get_loaders(root, train_patients, val_patients):
    train_data = build_train_list(root, train_patients)
    val_data   = build_train_list(root, val_patients)

    train_ds = Dataset(train_data, transform=get_train_transforms())
    val_ds   = Dataset(val_data, transform=get_val_transforms())

    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=4
    )

    return train_loader, val_loader
