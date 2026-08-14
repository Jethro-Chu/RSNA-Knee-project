"""
Tests for Multi-View Hierarchical MIL Architecture and Missing Plane Handling.
"""

import pytest
import torch
from rsna_knee.models.multi_view_hmil import MultiViewHMILModel


def test_multi_view_hmil_forward():
    model = MultiViewHMILModel(
        backbone_name="resnet34d",
        pretrained=False,
        num_targets=12,
        in_channels=3,
    )
    model.eval()

    B, S, C, H, W = 2, 8, 3, 64, 64
    inputs = {
        "sagittal": torch.randn(B, S, C, H, W),
        "coronal": torch.randn(B, S, C, H, W),
        "axial": torch.randn(B, S, C, H, W),
    }

    with torch.no_grad():
        logits = model(inputs)

    assert logits.shape == (B, 12)
    assert not torch.isnan(logits).any()


def test_multi_view_hmil_missing_plane():
    model = MultiViewHMILModel(
        backbone_name="resnet34d",
        pretrained=False,
        num_targets=12,
        in_channels=3,
    )
    model.eval()

    B, S, C, H, W = 2, 8, 3, 64, 64
    # Missing axial plane
    inputs = {
        "sagittal": torch.randn(B, S, C, H, W),
        "coronal": torch.randn(B, S, C, H, W),
    }

    with torch.no_grad():
        logits = model(inputs)

    assert logits.shape == (B, 12)
    assert not torch.isnan(logits).any()
