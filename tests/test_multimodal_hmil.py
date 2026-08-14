"""
Tests for Multimodal HMIL Architecture (Vision + Metadata Fusion).
"""

import pytest
import torch
from rsna_knee.models.multimodal_hmil import MultimodalHMILModel


def test_multimodal_hmil_forward():
    model = MultimodalHMILModel(
        backbone_name="resnet34d",
        pretrained=False,
        num_targets=12,
        in_channels=3,
        meta_dim=16,
    )
    model.eval()

    B, S, C, H, W = 2, 8, 3, 64, 64
    inputs = {
        "sagittal": torch.randn(B, S, C, H, W),
        "coronal": torch.randn(B, S, C, H, W),
        "axial": torch.randn(B, S, C, H, W),
    }
    meta_feats = torch.randn(B, 16)

    with torch.no_grad():
        logits = model(inputs, meta_features=meta_feats)

    assert logits.shape == (B, 12)
    assert not torch.isnan(logits).any()


def test_multimodal_hmil_missing_meta():
    model = MultimodalHMILModel(
        backbone_name="resnet34d",
        pretrained=False,
        num_targets=12,
        in_channels=3,
        meta_dim=16,
    )
    model.eval()

    B, S, C, H, W = 2, 8, 3, 64, 64
    inputs = {
        "sagittal": torch.randn(B, S, C, H, W),
        "coronal": torch.randn(B, S, C, H, W),
        "axial": torch.randn(B, S, C, H, W),
    }

    with torch.no_grad():
        logits = model(inputs, meta_features=None)

    assert logits.shape == (B, 12)
    assert not torch.isnan(logits).any()
