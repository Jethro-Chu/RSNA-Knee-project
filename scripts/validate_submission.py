#!/usr/bin/env python3
"""
Submission validation script for RSNA Knee Abnormality Detection.
Validates row counts, study ID matching, target columns, values in [0, 1], and lack of NaNs/infs.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from rsna_knee.constants import ID_COLUMN, SUBMISSION_COLUMNS, TARGET_NAMES


def validate_submission_file(submission_path: Path, sample_path: Path) -> bool:
    """
    Validates a generated submission file against the official sample_submission.csv.
    """
    print(f"[*] Validating submission file: {submission_path}")
    print(f"[*] Comparing against sample file: {sample_path}")

    if not submission_path.exists():
        print(f"[!] ERROR: Submission file does not exist: {submission_path}")
        return False

    if not sample_path.exists():
        print(f"[!] WARNING: Sample file not found at {sample_path}. Performing standalone format checks.")
        sample_df = None
    else:
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
        # Check if same set but different order
        if set(actual_cols) == set(expected_cols):
            print("    [!] Hint: Columns are present but in wrong order.")
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
        
        # Check numeric type
        if not np.issubdtype(vals.dtype, np.number):
            print(f"[!] ERROR: Target '{target}' contains non-numeric values.")
            return False

        # Check NaNs
        if np.isnan(vals).any():
            nan_count = int(np.isnan(vals).sum())
            print(f"[!] ERROR: Target '{target}' contains {nan_count} NaN values.")
            return False

        # Check Infs
        if np.isinf(vals).any():
            inf_count = int(np.isinf(vals).sum())
            print(f"[!] ERROR: Target '{target}' contains {inf_count} Infinite values.")
            return False

        # Check [0, 1] range
        min_val, max_val = float(np.min(vals)), float(np.max(vals))
        if min_val < 0.0 or max_val > 1.0:
            print(f"[!] ERROR: Target '{target}' values out of bounds [0, 1]: min={min_val}, max={max_val}")
            return False

        # Check variance (warn if constant)
        if min_val == max_val:
            print(f"    [!] Warning: Target '{target}' has constant probability: {min_val:.4f}")

    print("[+] SUCCESS: Submission passed all validation checks!")
    print(f"    - Total studies: {len(sub_df)}")
    print(f"    - Total target columns: {len(TARGET_NAMES)}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate RSNA Knee submission CSV")
    parser.add_argument("--submission", type=str, required=True, help="Path to submission.csv")
    parser.add_argument("--sample", type=str, default="data/metadata/sample_submission.csv", help="Path to sample_submission.csv")
    args = parser.parse_args()

    success = validate_submission_file(Path(args.submission), Path(args.sample))
    sys.exit(0 if success else 1)
