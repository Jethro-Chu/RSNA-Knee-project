"""
Tests for DICOM series plane classification (Sagittal, Coronal, Axial).
"""

import pytest
from rsna_knee.data.series import determine_plane_from_orientation


def test_determine_sagittal_plane():
    # Sagittal slice: normal along X-axis
    iop = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]
    plane = determine_plane_from_orientation(iop)
    assert plane == "sagittal"


def test_determine_coronal_plane():
    # Coronal slice: normal along Y-axis
    iop = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    plane = determine_plane_from_orientation(iop)
    assert plane == "coronal"


def test_determine_axial_plane():
    # Axial slice: normal along Z-axis
    iop = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    plane = determine_plane_from_orientation(iop)
    assert plane == "axial"


def test_unknown_plane():
    plane = determine_plane_from_orientation([0.0, 0.0])
    assert plane == "unknown"
