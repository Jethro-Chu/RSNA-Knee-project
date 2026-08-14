"""
Tests for submission generation and validation logic.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from rsna_knee.constants import ID_COLUMN, SUBMISSION_COLUMNS, TARGET_NAMES
from rsna_knee.inference.submission import validate_submission_file


def test_submission_validation_success(tmp_path):
    sub_file = tmp_path / "submission.csv"
    sample_file = tmp_path / "sample_submission.csv"

    # Create dummy sample
    sample_data = {ID_COLUMN: ["1.2.840.1", "1.2.840.2", "1.2.840.3"]}
    for t in TARGET_NAMES:
        sample_data[t] = [0.5, 0.5, 0.5]
    sample_df = pd.DataFrame(sample_data)
    sample_df.to_csv(sample_file, index=False)

    # Create valid submission
    sub_data = {ID_COLUMN: ["1.2.840.1", "1.2.840.2", "1.2.840.3"]}
    for t in TARGET_NAMES:
        sub_data[t] = [0.12, 0.85, 0.04]
    sub_df = pd.DataFrame(sub_data)
    sub_df.to_csv(sub_file, index=False)

    assert validate_submission_file(sub_file, sample_file) == True


def test_submission_validation_nan_failure(tmp_path):
    sub_file = tmp_path / "submission_nan.csv"
    sample_file = tmp_path / "sample_submission.csv"

    sample_data = {ID_COLUMN: ["1.2.840.1"]}
    for t in TARGET_NAMES:
        sample_data[t] = [0.5]
    pd.DataFrame(sample_data).to_csv(sample_file, index=False)

    sub_data = {ID_COLUMN: ["1.2.840.1"]}
    for t in TARGET_NAMES:
        sub_data[t] = [np.nan]
    pd.DataFrame(sub_data).to_csv(sub_file, index=False)

    assert validate_submission_file(sub_file, sample_file) == False
