"""
Inference engine for study-level Knee MRI multi-label abnormality prediction.
Processes raw test study DICOM files and generates model predictions.
"""

from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import pandas as pd
import torch

from rsna_knee.constants import ID_COLUMN, SUBMISSION_COLUMNS, TARGET_NAMES
from rsna_knee.data.dicom import load_dicom_slice, normalize_mri_series, sample_slices_2p5d, sort_dicom_files
from rsna_knee.models.model_2p5d import Knee2p5dModel


def predict_study_dicoms(
    study_dir: Path,
    model: Knee2p5dModel,
    device: torch.device,
    image_size: int = 256,
    slices_per_series: int = 16,
    channels: int = 3,
) -> np.ndarray:
    """
    Predicts abnormality probabilities for a single study folder containing DICOM files.
    Returns:
        1D array of 12 probabilities in range [0, 1].
    """
    model.eval()

    dicom_files = list(study_dir.glob("**/*.dcm")) if study_dir.exists() else []
    
    if len(dicom_files) == 0:
        # Graceful fallback: return default prevalence / neutral prior
        return np.full(len(TARGET_NAMES), 0.15, dtype=np.float32)

    try:
        sorted_files = sort_dicom_files(dicom_files)
        slices = []
        for file_path, _ in sorted_files:
            sl = load_dicom_slice(file_path)
            if sl is not None:
                slices.append(sl)

        if len(slices) == 0:
            return np.full(len(TARGET_NAMES), 0.15, dtype=np.float32)

        volume = np.stack(slices, axis=0)
        volume = normalize_mri_series(volume)
        sampled = sample_slices_2p5d(volume, target_slice_count=slices_per_series, channels=channels)

        # Resize if dimensions differ
        S, C, H, W = sampled.shape
        if H != image_size or W != image_size:
            import cv2
            resized = []
            for s in range(S):
                ch_stack = []
                for c in range(C):
                    res = cv2.resize(sampled[s, c], (image_size, image_size), interpolation=cv2.INTER_LINEAR)
                    ch_stack.append(res)
                resized.append(np.stack(ch_stack, axis=0))
            sampled = np.stack(resized, axis=0)

        tensor = torch.from_numpy(sampled).unsqueeze(0).float().to(device)  # (1, S, C, H, W)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        # Guarantee valid [0, 1] range and no NaNs
        probs = np.nan_to_num(probs, nan=0.15, posinf=1.0, neginf=0.0)
        probs = np.clip(probs, 0.0, 1.0)
        return probs.astype(np.float32)

    except Exception as e:
        # Failsafe fallback
        return np.full(len(TARGET_NAMES), 0.15, dtype=np.float32)


def generate_submission(
    test_df: pd.DataFrame,
    data_dir: Path,
    model: Knee2p5dModel,
    device: torch.device,
    output_path: Path,
) -> pd.DataFrame:
    """
    Generates full submission dataframe and writes submission.csv.
    """
    results = []

    for _, row in test_df.iterrows():
        study_id = str(row[ID_COLUMN])
        study_path = data_dir / "test_series" / study_id
        if not study_path.exists():
            study_path = data_dir / study_id

        probs = predict_study_dicoms(study_path, model, device)

        row_dict = {ID_COLUMN: study_id}
        for t_idx, target in enumerate(TARGET_NAMES):
            row_dict[target] = float(probs[t_idx])

        results.append(row_dict)

    sub_df = pd.DataFrame(results, columns=SUBMISSION_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(output_path, index=False)
    print(f"[+] Submission generated and saved to: {output_path}")
    return sub_df
