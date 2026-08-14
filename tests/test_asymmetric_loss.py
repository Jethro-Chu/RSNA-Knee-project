"""
Tests for Asymmetric Loss function and gradient behavior.
"""

import pytest
import torch
from rsna_knee.models.asymmetric_loss import AsymmetricLoss


def test_asymmetric_loss_forward():
    criterion = AsymmetricLoss()
    logits = torch.randn(4, 12, requires_grad=True)
    targets = torch.randint(0, 2, (4, 12)).float()
    mask = torch.ones(4, 12, dtype=torch.bool)
    weights = torch.ones(4, 12)

    loss = criterion(logits, targets, loss_mask=mask, weights=weights)
    assert loss.item() > 0.0
    assert not torch.isnan(loss)

    loss.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()


def test_asymmetric_loss_masked_supervision():
    criterion = AsymmetricLoss()
    logits = torch.randn(2, 12)
    targets = torch.zeros(2, 12)
    # Mask out all except first target
    mask = torch.zeros(2, 12, dtype=torch.bool)
    mask[:, 0] = True

    loss = criterion(logits, targets, loss_mask=mask)
    assert not torch.isnan(loss)
