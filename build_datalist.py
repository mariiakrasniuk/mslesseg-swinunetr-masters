import os

def build_train_list(root, patients):
    data = []

    for p in patients:
        p_dir = os.path.join(root, "train", p)
        for t in os.listdir(p_dir):
            t_dir = os.path.join(p_dir, t)

            data.append({
                "image": os.path.join(t_dir, f"{p}_{t}_FLAIR.nii.gz"),
                "label": os.path.join(t_dir, f"{p}_{t}_MASK.nii.gz"),
                "patient": p,
                "timepoint": t
            })
    return data


def build_test_list(root):
    data = []
    test_root = os.path.join(root, "test")

    for p in sorted(os.listdir(test_root)):
        p_dir = os.path.join(test_root, p)
        data.append({
            "image": os.path.join(p_dir, f"{p}_FLAIR.nii.gz"),
            "label": os.path.join(p_dir, f"{p}_MASK.nii.gz"),
            "patient": p
        })
    return data
