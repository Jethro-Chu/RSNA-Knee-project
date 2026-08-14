"""
Tests for DICOM Metadata Feature Extraction.
"""

import pytest
from rsna_knee.data.metadata_features import extract_study_metadata_features


def test_extract_metadata_empty():
    feats = extract_study_metadata_features([])
    assert feats.shape == (16,)
    assert feats.dtype.kind == "f"
    assert not pytest.approx(feats[0]) == 999.0
