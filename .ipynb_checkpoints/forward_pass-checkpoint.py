import argparse
import torch
from dataloaders import get_loaders
from splits import TRAIN_PATIENTS, VAL_PATIENTS
from model import build_model

ROOT = "MSLesSeg_Dataset"

parser = argparse.ArgumentParser()
parser.add_argument("--variant", type=str, default="baseline",
                    choices=["baseline", "wavelet_a", "wavelet_a_plus", "wavelet_b", "wavelet_a_higher_level"],
                    help="Model variant to smoke test")
args = parser.parse_args()

train_loader, _ = get_loaders(ROOT, TRAIN_PATIENTS, VAL_PATIENTS)

model = build_model(args.variant, use_checkpoint=True).cuda()

batch = next(iter(train_loader))

# batch is a list of dicts because num_samples > 1
sample = batch[0]
x = sample["image"].cuda()

y = model(x)
print(f"[{args.variant}] input: {x.shape}  output: {y.shape}")
