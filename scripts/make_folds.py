#!/usr/bin/env python3
"""
CLI script to partition studies into leakage-free stratified GroupKFold cross-validation splits.
"""

import argparse
from pathlib import Path
import pandas as pd

from rsna_knee.paths import get_data_dir, get_metadata_path
from rsna_knee.training.folds import create_study_folds


def main():
    parser = argparse.ArgumentParser(description="Create study-level stratified cross-validation folds")
    parser.add_argument("--input", type=str, default=None, help="Input metadata / pseudo-labels path")
    parser.add_argument("--output", type=str, default=None, help="Output folds CSV path")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else get_data_dir() / "pseudo_labels" / "pseudo_labels_v1.parquet"
    output_path = Path(args.output) if args.output else get_data_dir() / "metadata" / "folds.csv"

    if not input_path.exists():
        input_path = get_metadata_path("train.csv")

    print(f"[*] Reading dataset from: {input_path}")
    if str(input_path).endswith(".parquet"):
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    folds_df = create_study_folds(df, n_splits=args.n_splits, seed=args.seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    folds_df.to_csv(output_path, index=False)
    print(f"[+] Successfully partitioned {len(folds_df)} studies across {args.n_splits} folds.")
    print(f"[+] Saved folds table to: {output_path}")


if __name__ == "__main__":
    main()
