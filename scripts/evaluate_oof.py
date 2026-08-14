#!/usr/bin/env python3
"""
CLI script to evaluate Out-of-Fold (OOF) predictions and compute authentic Macro ROC-AUC.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from rsna_knee.constants import ID_COLUMN, TARGET_NAMES
from rsna_knee.training.metrics import compute_macro_auc, compute_per_target_auc


def main():
    parser = argparse.ArgumentParser(description="Evaluate OOF predictions")
    parser.add_argument("--preds", type=str, default="outputs/oof_predictions.parquet", help="Path to OOF predictions")
    parser.add_argument("--labels", type=str, default="data/pseudo_labels/pseudo_labels_v2.parquet", help="Path to labels")
    args = parser.parse_args()

    preds_path = Path(args.preds)
    labels_path = Path(args.labels)

    if not preds_path.exists():
        print(f"[!] Predictions file not found: {preds_path}")
        return

    print(f"[*] Loading predictions: {preds_path}")
    preds_df = pd.read_parquet(preds_path) if str(preds_path).endswith(".parquet") else pd.read_csv(preds_path)

    print(f"[*] Loading ground truth/labels: {labels_path}")
    labels_df = pd.read_parquet(labels_path) if str(labels_path).endswith(".parquet") else pd.read_csv(labels_path)

    merged = pd.merge(labels_df, preds_df, on=ID_COLUMN, suffixes=("_true", "_pred"))

    y_true = np.zeros((len(merged), len(TARGET_NAMES)), dtype=np.float64)
    y_pred = np.zeros((len(merged), len(TARGET_NAMES)), dtype=np.float64)
    mask = np.ones((len(merged), len(TARGET_NAMES)), dtype=bool)

    for i, t in enumerate(TARGET_NAMES):
        true_col = f"{t}_prob_true" if f"{t}_prob_true" in merged else (f"{t}_true" if f"{t}_true" in merged else f"{t}_prob")
        pred_col = f"{t}_prob_pred" if f"{t}_prob_pred" in merged else (f"{t}_pred" if f"{t}_pred" in merged else t)
        mask_col = f"{t}_loss_mask_true" if f"{t}_loss_mask_true" in merged else f"{t}_loss_mask"

        y_true[:, i] = merged[true_col].values
        y_pred[:, i] = merged[pred_col].values
        if mask_col in merged:
            mask[:, i] = merged[mask_col].values.astype(bool)

    macro_auc, per_target_auc = compute_macro_auc(y_true, y_pred, mask=mask)

    print("\n" + "="*60)
    print("       OUT-OF-FOLD (OOF) VALIDATION PERFORMANCE REPORT")
    print("="*60)
    print(f"{'Target Pathology':<30} | {'OOF Val ROC-AUC':<15}")
    print("-" * 60)
    for t in TARGET_NAMES:
        auc_val = per_target_auc.get(t, np.nan)
        if not np.isnan(auc_val):
            print(f"  {t:<28} : {auc_val:.4f}")
        else:
            print(f"  {t:<28} : [Single Class / Masked]")

    print("=" * 60)
    print(f" OVERALL OOF UNWEIGHTED MACRO ROC-AUC: {macro_auc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
