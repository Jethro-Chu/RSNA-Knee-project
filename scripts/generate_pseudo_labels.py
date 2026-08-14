#!/usr/bin/env python3
"""
CLI script to generate structured, versioned pseudo-labels from train.csv radiology reports.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from rsna_knee.paths import get_data_dir, get_metadata_path, get_output_dir
from rsna_knee.reports.extractor import ReportAbnormalityExtractor
from rsna_knee.reports.weak_supervision import generate_pseudo_labels_dataframe


def main():
    parser = argparse.ArgumentParser(description="Generate weak pseudo-labels from reports")
    parser.add_argument("--input", type=str, default=None, help="Path to train.csv")
    parser.add_argument("--output", type=str, default=None, help="Output parquet/csv path")
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else get_metadata_path("train.csv")
    output_path = Path(args.output) if args.output else get_data_dir() / "pseudo_labels" / "pseudo_labels_v1.parquet"

    print(f"[*] Reading training metadata from: {input_path}")
    if not input_path.exists():
        print(f"[!] Warning: {input_path} not found. Creating dummy template for testing.")
        from rsna_knee.constants import ID_COLUMN, TARGET_NAMES
        dummy_data = {
            ID_COLUMN: ["study_001", "study_002", "study_003"],
            "Report": [
                "IMPRESSION: Complete tear of the ACL and bone contusion.",
                "FINDINGS: Intact menisci, no joint effusion or fracture.",
                "BEFUND: VKB-Ruptur mit Innenbandläsion.",
            ]
        }
        for t in TARGET_NAMES:
            dummy_data[t] = [1.0 if t in ["ACL", "Contusion"] else 0.0, np.nan, np.nan]
        train_df = pd.DataFrame(dummy_data)
    else:
        train_df = pd.read_csv(input_path)

    extractor = ReportAbnormalityExtractor()
    pseudo_df = generate_pseudo_labels_dataframe(train_df, extractor=extractor)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if str(output_path).endswith(".parquet"):
        pseudo_df.to_parquet(output_path, index=False)
    else:
        pseudo_df.to_csv(output_path, index=False)

    print(f"[+] Successfully extracted pseudo-labels for {len(pseudo_df)} studies.")
    print(f"[+] Saved to: {output_path}")


if __name__ == "__main__":
    main()
