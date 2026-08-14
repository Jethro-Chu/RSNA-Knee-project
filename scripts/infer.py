#!/usr/bin/env python3
"""
CLI script to run end-to-end inference and produce verified submission.csv.
"""

import argparse
from pathlib import Path
import pandas as pd
import torch

from rsna_knee.constants import ID_COLUMN, SUBMISSION_COLUMNS, TARGET_NAMES
from rsna_knee.inference.predict import generate_submission
from rsna_knee.models.model_2p5d import Knee2p5dModel
from rsna_knee.paths import get_data_dir, get_output_dir
from rsna_knee.inference.submission import validate_submission_file


def main():
    parser = argparse.ArgumentParser(description="Run RSNA Knee inference on test set")
    parser.add_argument("--weights", type=str, default=None, help="Path to checkpoint .pt")
    parser.add_argument("--output", type=str, default=None, help="Path to output submission.csv")
    parser.add_argument("--test-csv", type=str, default=None, help="Path to test.csv")
    args = parser.parse_args()

    data_dir = get_data_dir()
    test_csv = Path(args.test_csv) if args.test_csv else data_dir / "test.csv"
    output_path = Path(args.output) if args.output else get_output_dir() / "submissions" / "submission.csv"

    if not test_csv.exists():
        sample_path = data_dir / "metadata" / "sample_submission.csv"
        if not sample_path.exists():
            sample_path = data_dir / "sample_submission.csv"
        if sample_path.exists():
            test_df = pd.read_csv(sample_path)[[ID_COLUMN]]
        else:
            print("[!] Warning: test.csv not found. Creating dummy test template.")
            test_df = pd.DataFrame({ID_COLUMN: ["test_study_001", "test_study_002"]})
    else:
        test_df = pd.read_csv(test_csv)

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[*] Inference on device: {device} | Test studies: {len(test_df)}")

    model = Knee2p5dModel(
        backbone_name="resnet34d",
        pretrained=False,
        num_targets=12,
        in_channels=3,
    )

    if args.weights and Path(args.weights).exists():
        print(f"[*] Loading checkpoint weights: {args.weights}")
        ckpt = torch.load(args.weights, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
    else:
        print("[!] No checkpoint weights provided. Running baseline Model 0 inference.")

    model.to(device)
    sub_df = generate_submission(
        test_df=test_df,
        data_dir=data_dir,
        model=model,
        device=device,
        output_path=output_path,
    )

    # Validate output
    sample_sub = data_dir / "metadata" / "sample_submission.csv"
    if not sample_sub.exists():
        sample_sub = data_dir / "sample_submission.csv"
    validate_submission_file(output_path, sample_sub)


if __name__ == "__main__":
    main()
