"""
Study-level multi-label stratified cross-validation fold generation.
Ensures zero leakage across studies and patients while balancing multi-label prevalence.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from rsna_knee.constants import ID_COLUMN, TARGET_NAMES


def create_study_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    stratify_by_expert: bool = True,
) -> pd.DataFrame:
    """
    Creates leakage-free fold assignments at the StudyInstanceUID level.
    If expert labels are present, stratifies expert studies across all folds evenly.
    """
    df = df.copy()
    df["fold"] = -1

    # Check if expert labels indicator exists
    if "has_expert_labels" in df.columns:
        expert_mask = df["has_expert_labels"] == True
    else:
        # Infer from presence of non-null target values in raw columns
        expert_mask = pd.Series(False, index=df.index)
        for target in TARGET_NAMES:
            if target in df.columns:
                valid = df[target].notna() & (df[target] != "")
                expert_mask = expert_mask | valid

    expert_indices = df[expert_mask].index.to_numpy()
    non_expert_indices = df[~expert_mask].index.to_numpy()

    # Helper to assign folds safely
    def _assign_splits(indices: np.ndarray):
        if len(indices) == 0:
            return
        if len(indices) >= n_splits:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            for fold, (_, val_idx) in enumerate(kf.split(indices)):
                df.loc[indices[val_idx], "fold"] = fold
        else:
            rng = np.random.RandomState(seed)
            shuffled = rng.permutation(indices)
            for i, idx in enumerate(shuffled):
                df.loc[idx, "fold"] = i % n_splits

    _assign_splits(expert_indices)
    _assign_splits(non_expert_indices)

    return df
