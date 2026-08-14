"""
Unit tests for DICOM parsing, geometric normal projection, intensity normalization, and 2.5D slice sampling.
"""

import numpy as np
import pytest
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset

from rsna_knee.data.dicom import calculate_slice_position_along_normal, normalize_mri_series, sample_slices_2p5d


def create_synthetic_dicom(iop, ipp, instance_num=1):
    ds = Dataset()
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.ImageOrientationPatient = iop
    ds.ImagePositionPatient = ipp
    ds.InstanceNumber = instance_num
    ds.PhotometricInterpretation = "MONOCHROME2"
    return ds


def test_calculate_slice_position_along_normal():
    # Axial slice: row = [1, 0, 0], col = [0, 1, 0] -> normal = [0, 0, 1] (Z-axis)
    iop = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ipp1 = [0.0, 0.0, 10.5]
    ipp2 = [0.0, 0.0, 25.0]

    ds1 = create_synthetic_dicom(iop, ipp1)
    ds2 = create_synthetic_dicom(iop, ipp2)

    pos1 = calculate_slice_position_along_normal(ds1)
    pos2 = calculate_slice_position_along_normal(ds2)

    assert pos1 == pytest.approx(10.5, abs=1e-4)
    assert pos2 == pytest.approx(25.0, abs=1e-4)
    assert pos2 > pos1


def test_normalize_mri_series():
    volume = np.array([[[0, 100], [500, 1000]]], dtype=np.float32)
    norm = normalize_mri_series(volume)
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0


def test_sample_slices_2p5d():
    volume = np.random.randn(20, 64, 64).astype(np.float32)
    sampled = sample_slices_2p5d(volume, target_slice_count=8, channels=3)

    assert sampled.shape == (8, 3, 64, 64)
    assert sampled.dtype == np.float32
