"""
Submission generation and validation logic for RSNA Knee Abnormality Detection.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from rsna_knee.constants import ID_COLUMN, SUBMISSION_COLUMNS, TARGET_NAMES


def validate_submission_file(submission_path: Path, sample_path: Optional[Path] = None) -> bool:
    """
    Validates a generated submission file against the official format and constraints.
    """
    print(f"[*] Validating submission file: {submission_path}")

    if not submission_path.exists():
        print(f"[!] ERROR: Submission file does not exist: {submission_path}")
        return False

    sample_df = None
    if sample_path is not None and sample_path.exists():
        sample_df = pd.read_csv(sample_path)

    try:
        sub_df = pd.read_csv(submission_path)
    except Exception as e:
        print(f"[!] ERROR: Failed to parse submission CSV: {e}")
        return False

    # Check Columns
    expected_cols = SUBMISSION_COLUMNS
    actual_cols = list(sub_df.columns)

    if actual_cols != expected_cols:
        print(f"[!] ERROR: Column mismatch!")
        print(f"    Expected: {expected_cols}")
        print(f"    Actual:   {actual_cols}")
        return False

    # Check ID column and rows if sample provided
    if sample_df is not None:
        if len(sub_df) != len(sample_df):
            print(f"[!] ERROR: Row count mismatch! Submission has {len(sub_df)} rows, sample has {len(sample_df)} rows.")
            return False

        if not (sub_df[ID_COLUMN].values == sample_df[ID_COLUMN].values).all():
            print(f"[!] ERROR: StudyInstanceUID values or ordering do not match sample_submission.csv!")
            return False

    # Check for NaNs, Infs, and Value Bounds
    for target in TARGET_NAMES:
        vals = sub_df[target].values
        
        if not np.issubdtype(vals.dtype, np.number):
            print(f"[!] ERROR: Target '{target}' contains non-numeric values.")
            return False

        if np.isnan(vals).any():
            nan_count = int(np.isnan(vals).sum())
            print(f"[!] ERROR: Target '{target}' contains {nan_count} NaN values.")
            return False

        if np.isinf(vals).any():
            inf_count = int(np.isinf(vals).sum())
            print(f"[!] ERROR: Target '{target}' contains {inf_count} Infinite values.")
            return False

        min_val, max_val = float(np.min(vals)), float(np.max(vals))
        if min_val < 0.0 or max_val > 1.0:
            print(f"[!] ERROR: Target '{target}' values out of bounds [0, 1]: min={min_val}, max={max_val}")
            return False

    print("[+] SUCCESS: Submission passed all validation checks!")
    print(f"    - Total studies: {len(sub_df)}")
    print(f"    - Total target columns: {len(TARGET_NAMES)}")
    return True
