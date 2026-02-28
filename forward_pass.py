from dataloaders import get_loaders
from splits import TRAIN_PATIENTS, VAL_PATIENTS
from monai.networks.nets import SwinUNETR
import torch

ROOT = "MSLesSeg Dataset"

train_loader, _ = get_loaders(ROOT, TRAIN_PATIENTS, VAL_PATIENTS)

model = SwinUNETR(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    feature_size=48,
    use_checkpoint=True,
).cuda()

batch = next(iter(train_loader))

# batch is a list of dicts because num_samples > 1
sample = batch[0]

x = sample["image"].cuda()


y = model(x)
print(x.shape, y.shape)
