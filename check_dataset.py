import nibabel as nib
import numpy as np
from build_datalist import build_train_list

ROOT = "MSLesSeg_Dataset"
PATIENTS = ["P1", "P2", "P3"]  # test on a few first

data = build_train_list(ROOT, PATIENTS)

for d in data[:3]:
    img = nib.load(d["image"]).get_fdata()
    msk = nib.load(d["label"]).get_fdata()

    print(d["image"])
    print(" image shape:", img.shape)
    print(" mask shape:", msk.shape)
    print(" mask unique:", np.unique(msk))
    print("-" * 40)
