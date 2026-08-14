"""
Tests for RSNA Knee multi-label macro ROC-AUC evaluation metrics.
"""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from rsna_knee.constants import TARGET_NAMES
from rsna_knee.training.metrics import compute_bootstrap_ci, compute_macro_auc, compute_per_target_auc


def test_perfect_macro_auc():
    y_true = np.array([
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    ], dtype=np.float32)

    y_pred = y_true.copy()  # Perfect predictions

    macro_auc, per_target = compute_macro_auc(y_true, y_pred)
    assert macro_auc == 1.0
    for target in TARGET_NAMES:
        assert per_target[target] == 1.0


def test_random_predictions_macro_auc():
    np.random.seed(42)
    y_true = np.random.randint(0, 2, size=(100, 12)).astype(np.float32)
    y_pred = np.random.uniform(0.0, 1.0, size=(100, 12)).astype(np.float32)

    macro_auc, per_target = compute_macro_auc(y_true, y_pred)
    assert 0.35 <= macro_auc <= 0.65
    assert len(per_target) == 12


def test_single_class_handling():
    # If a target has all 0s (no positives), it should be NaN and excluded from macro average
    y_true = np.array([
        [0, 1],
        [0, 0],
        [0, 1],
        [0, 0],
    ], dtype=np.float32)

    y_pred = np.array([
        [0.1, 0.9],
        [0.2, 0.1],
        [0.3, 0.8],
        [0.4, 0.2],
    ], dtype=np.float32)

    target_names = ["SingleClassTarget", "BinaryTarget"]
    macro_auc, per_target = compute_macro_auc(y_true, y_pred, target_names=target_names)

    assert np.isnan(per_target["SingleClassTarget"])
    assert per_target["BinaryTarget"] == 1.0
    assert macro_auc == 1.0  # Unweighted mean of valid targets


def test_masked_loss_auc():
    y_true = np.array([
        [1, 1],
        [0, 0],
        [1, 0],
        [0, 1],
    ], dtype=np.float32)

    y_pred = np.array([
        [0.9, 0.9],
        [0.1, 0.1],
        [0.8, 0.2],
        [0.2, 0.8],
    ], dtype=np.float32)

    mask = np.array([
        [True, False],
        [True, True],
        [True, True],
        [True, True],
    ], dtype=bool)

    macro_auc, per_target = compute_macro_auc(y_true, y_pred, target_names=["T1", "T2"], mask=mask)
    assert not np.isnan(macro_auc)


def test_bootstrap_ci():
    np.random.seed(42)
    y_true = np.random.randint(0, 2, size=(50, 12)).astype(np.float32)
    y_pred = np.random.uniform(0.0, 1.0, size=(50, 12)).astype(np.float32)

    mean, lower, upper = compute_bootstrap_ci(y_true, y_pred, n_bootstraps=50)
    assert lower <= mean <= upper
