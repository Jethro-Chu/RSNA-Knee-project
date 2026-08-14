"""
Tests for Ensembling, Rank Averaging, and Test-Time Augmentation.
"""

import numpy as np
import pytest
import torch

from rsna_knee.inference.ensemble import (
    apply_test_time_augmentation_2p5d,
    ensemble_predictions_rank_average,
    ensemble_predictions_weighted_average,
)


def test_ensemble_weighted_average():
    pred1 = np.full((5, 12), 0.2, dtype=np.float32)
    pred2 = np.full((5, 12), 0.8, dtype=np.float32)

    blended = ensemble_predictions_weighted_average([pred1, pred2], weights=[0.5, 0.5])
    assert np.allclose(blended, 0.5)


def test_ensemble_rank_average():
    pred1 = np.array([[0.1, 0.2], [0.9, 0.8]], dtype=np.float32)
    pred2 = np.array([[0.2, 0.1], [0.8, 0.9]], dtype=np.float32)

    ranked = ensemble_predictions_rank_average([pred1, pred2])
    assert ranked.shape == (2, 2)
    assert ranked.min() >= 0.0
    assert ranked.max() <= 1.0


def test_test_time_augmentation():
    x = torch.randn(2, 8, 3, 32, 32)
    flipped = apply_test_time_augmentation_2p5d(x, flip_h=True)
    assert flipped.shape == x.shape
    assert not torch.equal(x, flipped)
