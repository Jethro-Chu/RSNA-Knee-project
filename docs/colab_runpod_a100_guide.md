# Fast Training Guide for NVIDIA A100 / H100 (Google Colab, RunPod, Lambda Labs)

This guide provides one-click and command-line instructions to train the full 5-fold Multimodal Tri-Plane HMIL RSNA Knee model on an **NVIDIA A100 (40GB / 80GB)** or **H100** GPU in **under 15 minutes**.

---

## Option 1: Google Colab Pro / Pro+ (A100 GPU)

1. Open a new notebook on [Google Colab](https://colab.research.google.com).
2. Go to **Runtime $\rightarrow$ Change runtime type $\rightarrow$ Select GPU $\rightarrow$ A100 GPU**.
3. In the first code cell, paste and run the following automated script:

```python
# 1. Clone Repository & Install Dependencies
!git clone https://github.com/Jethro-Chu/RSNA-Knee-project.git
%cd RSNA-Knee-project
!pip install -q -r requirements.txt pydicom timm

# 2. Configure Kaggle API & Download Competition Data
# (Upload your kaggle.json or provide your username and key)
import os
os.environ['KAGGLE_USERNAME'] = "YOUR_KAGGLE_USERNAME"  # Replace with your Kaggle username
os.environ['KAGGLE_KEY'] = "YOUR_KAGGLE_KEY"            # Replace with your Kaggle API key

!mkdir -p data
!kaggle competitions download -c rsna-knee-abnormality-detection -p data/
!unzip -q data/rsna-knee-abnormality-detection.zip -d data/

# 3. Launch Fast 5-Fold Training on A100 with AMP (Takes ~12-15 minutes total!)
!python scripts/train_multimodal.py --fold -1 --epochs 10 --batch-size 32 --lr 3e-4 --image-size 224

# 4. Upload Checkpoints directly to Kaggle Dataset
!kaggle datasets version -p outputs/checkpoints -m "Updated 5-fold A100 trained checkpoints"
```

---

## Option 2: RunPod / Lambda Labs / Cloud VM (1-Click Shell Command)

On any Linux VM equipped with an A100/H100, open a terminal and execute:

```bash
# 1. Clone & Setup
git clone https://github.com/Jethro-Chu/RSNA-Knee-project.git
cd RSNA-Knee-project
pip install -r requirements.txt

# 2. Download Data via Kaggle CLI
kaggle competitions download -c rsna-knee-abnormality-detection -p data/
unzip -q data/rsna-knee-abnormality-detection.zip -d data/

# 3. Launch 5-Fold A100 Training
python scripts/train_multimodal.py \
    --fold -1 \
    --epochs 10 \
    --batch-size 32 \
    --lr 3e-4 \
    --image-size 224
```

---

## Expected Velocity on A100 vs T4:

| Hardware | Batch Size | Time per Epoch | Full 5-Fold Runtime |
| :--- | :---: | :---: | :---: |
| **Kaggle GPU T4 x2 (Current)** | 16 | $\approx 50 - 60$ seconds | **$\approx 25$ minutes** |
| **NVIDIA A100 80GB (Colab/RunPod)** | 32 | $\approx 15 - 20$ seconds | **$\approx 8 - 12$ minutes** |
| **NVIDIA H100 (Lambda Labs)** | 64 | $\approx 8 - 10$ seconds | **$\approx 5 - 7$ minutes** |

---

## How Trained Checkpoints Sync to Kaggle Inference:
Once training completes on your A100/H100 machine, the 5 `.pt` files (`model_fold_0_best.pt` through `model_fold_4_best.pt`) are automatically saved to `outputs/checkpoints/`.
You can upload them to the Kaggle dataset `chujethro/rsna-knee-checkpoints` with:
```bash
kaggle datasets version -p outputs/checkpoints -m "A100 5-fold trained weights"
```
The Kaggle inference kernel [`wenwen12/rsna-knee`](https://www.kaggle.com/code/wenwen12/rsna-knee) will immediately load all 5 models with `strict=True` and run offline T4 ensemble inference!
