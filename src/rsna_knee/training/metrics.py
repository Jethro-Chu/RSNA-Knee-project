"""
Evaluation metrics for RSNA Knee Abnormality Detection.
Implements exact unweighted macro-average ROC-AUC across the 12 target findings.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from rsna_knee.constants import TARGET_NAMES


def compute_per_target_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Computes per-target ROC-AUC scores, safely handling targets with single class or missing labels.

    Args:
        y_true: Ground truth binary labels (N, num_targets)
        y_pred: Predicted probabilities (N, num_targets) in [0, 1]
        target_names: List of target column names
        mask: Boolean mask of valid evaluation entries (N, num_targets). If None, all non-NaNs used.

    Returns:
        Dict mapping target name to ROC-AUC score (or np.nan if undefined)
    """
    if target_names is None:
        target_names = TARGET_NAMES

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    assert y_true.shape == y_pred.shape, f"Shape mismatch: {y_true.shape} vs {y_pred.shape}"
    num_targets = y_true.shape[1]

    results: Dict[str, float] = {}

    for i in range(num_targets):
        col_name = target_names[i] if i < len(target_names) else f"Target_{i}"
        
        target_true = y_true[:, i]
        target_pred = y_pred[:, i]

        # Apply valid mask
        valid_idx = ~np.isnan(target_true) & ~np.isnan(target_pred)
        if mask is not None:
            valid_idx = valid_idx & mask[:, i]

        t_true_valid = target_true[valid_idx]
        t_pred_valid = target_pred[valid_idx]

        # Check if both positive and negative classes are present
        unique_classes = np.unique(t_true_valid)
        if len(unique_classes) < 2:
            # Undefined AUC for single class
            results[col_name] = np.nan
        else:
            try:
                auc = float(roc_auc_score(t_true_valid, t_pred_valid))
                results[col_name] = auc
            except Exception:
                results[col_name] = np.nan

    return results


def compute_macro_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None,
    mask: Optional[np.ndarray] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Computes unweighted macro-average ROC-AUC across all valid target classes.

    Returns:
        Tuple of (macro_auc_score, per_target_auc_dict)
    """
    per_target = compute_per_target_auc(y_true, y_pred, target_names, mask)
    valid_aucs = [v for v in per_target.values() if not np.isnan(v)]

    if len(valid_aucs) == 0:
        macro_auc = np.nan
    else:
        macro_auc = float(np.mean(valid_aucs))

    return macro_auc, per_target


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None,
    n_bootstraps: int = 500,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Computes bootstrap confidence interval for macro ROC-AUC.

    Returns:
        Tuple of (mean_macro_auc, lower_bound, upper_bound)
    """
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    bootstrap_scores = []

    for _ in range(n_bootstraps):
        boot_idx = rng.randint(0, n_samples, n_samples)
        score, _ = compute_macro_auc(y_true[boot_idx], y_pred[boot_idx], target_names)
        if not np.isnan(score):
            bootstrap_scores.append(score)

    if len(bootstrap_scores) == 0:
        return np.nan, np.nan, np.nan

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(bootstrap_scores, alpha * 100))
    upper = float(np.percentile(bootstrap_scores, (1.0 - alpha) * 100))
    mean_score = float(np.mean(bootstrap_scores))

    return mean_score, lower, upper
