import argparse
import torch
from dataloaders import get_loaders
from splits import TRAIN_PATIENTS, VAL_PATIENTS
from model import build_model

ROOT = "MSLesSeg_Dataset"

parser = argparse.ArgumentParser()
parser.add_argument("--variant", type=str, default="baseline",
                    choices=["baseline", "wavelet_a", "wavelet_a_plus", "wavelet_b",
                             "wavelet_a_higher_level", "wavelet_ml"],
                    help="Model variant to smoke test")
parser.add_argument("--wavelet", type=str, default="haar",
                    choices=["haar", "db2", "sym4"],
                    help="Wavelet family — only used with --variant wavelet_ml")
parser.add_argument("--levels", type=int, default=1,
                    choices=[1, 2, 3],
                    help="Decomposition levels — only used with --variant wavelet_ml")
args = parser.parse_args()

train_loader, _ = get_loaders(ROOT, TRAIN_PATIENTS, VAL_PATIENTS)

model = build_model(args.variant, use_checkpoint=True,
                    wavelet=args.wavelet, levels=args.levels).cuda()

batch = next(iter(train_loader))

# batch is a list of dicts because num_samples > 1
sample = batch[0]
x = sample["image"].cuda()

y = model(x)
tag = args.variant
if args.variant == "wavelet_ml":
    tag = f"wavelet_ml  wavelet={args.wavelet}  levels={args.levels}"
print(f"[{tag}] input: {x.shape}  output: {y.shape}")
