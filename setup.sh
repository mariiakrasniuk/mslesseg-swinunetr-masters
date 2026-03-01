conda create -n ms_swinunetr python=3.10 -y
conda activate ms_swinunetr
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install monai nibabel numpy tqdm scikit-learn matplotlib einops
find "MSLesSeg_Dataset" -name ".DS_Store" -delete
find "MSLesSeg_Dataset" -name ".DS_Store"
python train.py 