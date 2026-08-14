"""
Ensembling, Model Blending, Rank Averaging, and Test-Time Augmentation (TTA) Engine.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn


def apply_test_time_augmentation_2p5d(
    tensor: torch.Tensor,
    flip_h: bool = False,
    intensity_scale: float = 1.0,
) -> torch.Tensor:
    """
    Applies test-time augmentation to (B, S, C, H, W) 2.5D input tensor.
    """
    out = tensor.clone()
    if flip_h:
        out = torch.flip(out, dims=[-1])
    if intensity_scale != 1.0:
        out = out * intensity_scale
    return out


def ensemble_predictions_weighted_average(
    model_predictions: List[np.ndarray],
    weights: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Computes weighted linear average of prediction matrices [(N, 12), ...].
    """
    if len(model_predictions) == 1:
        return model_predictions[0]

    if weights is None:
        weights = [1.0 / len(model_predictions)] * len(model_predictions)

    weights = np.array(weights, dtype=np.float32)
    weights = weights / np.sum(weights)

    ensemble_pred = np.zeros_like(model_predictions[0], dtype=np.float32)
    for pred, w in zip(model_predictions, weights):
        ensemble_pred += pred * w

    return np.clip(ensemble_pred, 0.0, 1.0)


def ensemble_predictions_rank_average(
    model_predictions: List[np.ndarray],
    weights: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Computes weighted rank average across multiple diverse model predictions.
    Particularly robust for ROC-AUC optimization where calibration curves differ across architectures.
    """
    if len(model_predictions) == 1:
        return model_predictions[0]

    N, num_targets = model_predictions[0].shape
    if weights is None:
        weights = [1.0 / len(model_predictions)] * len(model_predictions)

    weights = np.array(weights, dtype=np.float32)
    weights = weights / np.sum(weights)

    ranked_preds = []
    for pred in model_predictions:
        rank_matrix = np.zeros_like(pred, dtype=np.float32)
        for k in range(num_targets):
            # Rank values from 0 to 1
            col = pred[:, k]
            ranks = stats.rankdata(col) / len(col)
            rank_matrix[:, k] = ranks
        ranked_preds.append(rank_matrix)

    # Average ranks
    avg_ranks = np.zeros((N, num_targets), dtype=np.float32)
    for r, w in zip(ranked_preds, weights):
        avg_ranks += r * w

    return avg_ranks
