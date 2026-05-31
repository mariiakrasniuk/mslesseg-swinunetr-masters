import argparse
import torch
from dataloaders import get_loaders
from splits import TRAIN_PATIENTS, VAL_PATIENTS
from model import build_model

ROOT = "MSLesSeg_Dataset"

parser = argparse.ArgumentParser()
parser.add_argument("--variant", type=str, default="baseline",
                    choices=["baseline", "wavelet_a", "wavelet_ml"],
                    help="Model variant to smoke test")
parser.add_argument("--wavelet", type=str, default="haar",
                    choices=["haar", "db2", "sym4"],
                    help="Wavelet family — only used with --variant wavelet_ml")
parser.add_argument("--levels", type=int, default=1,
                    choices=[1, 2, 3],
                    help="Decomposition levels — only used with --variant wavelet_ml")
parser.add_argument("--multimodal", action="store_true",
                    help="Use FLAIR + T1 + T2 as input (3 channels)")
parser.add_argument("--use_v2", action="store_true",
                    help="Use SwinUNETR-V2")
parser.add_argument("--aug", type=str, default="none",
                    choices=["none", "image", "coeff", "both"],
                    help="Augmentation strategy to smoke-test")
args = parser.parse_args()

in_channels   = 3 if args.multimodal else 1
coeff_aug     = args.aug in ("coeff", "both")
intensity_aug = args.aug in ("image", "both")

train_loader, _ = get_loaders(ROOT, TRAIN_PATIENTS, VAL_PATIENTS,
                               multimodal=args.multimodal,
                               intensity_aug=intensity_aug)

model = build_model(args.variant, in_channels=in_channels, use_checkpoint=True,
                    wavelet=args.wavelet, levels=args.levels,
                    use_v2=args.use_v2, coeff_aug=coeff_aug).cuda()

batch = next(iter(train_loader))

# batch is a list of dicts because num_samples > 1
sample = batch[0]
x = sample["image"].cuda()

y = model(x)
tag = args.variant
if args.variant == "wavelet_ml":
    tag = f"wavelet_ml  wavelet={args.wavelet}  levels={args.levels}"
print(f"[{tag}] input: {x.shape}  output: {y.shape}")
