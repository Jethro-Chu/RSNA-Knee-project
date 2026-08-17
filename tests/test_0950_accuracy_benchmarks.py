"""
Automated unit and regression tests for RSNA Knee Abnormality Detection 0.950+ goal metrics.
Verifies:
1. Macro ROC-AUC >= 0.950 on Full, Dev, and Holdout partitions.
2. All 12 targets are evaluated and bounded.
3. Submission files pass strict format, non-empty, and probability bounds [0.0, 1.0].
"""

import hashlib
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from sklearn.metrics import roc_auc_score

from rsna_knee.constants import ID_COLUMN, TARGET_NAMES

@pytest.fixture
def gold_evaluation_data():
    train_df = pd.read_csv("data/train.csv")
    split_df = pd.read_csv("data/gold_dev_holdout_split.csv")
    gold_df = pd.merge(split_df[["StudyInstanceUID", "split"]], train_df, on="StudyInstanceUID", how="left").sort_values("StudyInstanceUID").reset_index(drop=True)
    oof_df = pd.read_parquet("experiments/oof_v44_champion_0950.parquet")
    merged = pd.merge(gold_df, oof_df, on="StudyInstanceUID", suffixes=("_gt", "_pred"))
    return merged

def test_full_gold_macro_auc_gte_0950(gold_evaluation_data):
    merged = gold_evaluation_data
    aucs = []
    for t in TARGET_NAMES:
        gt_col = f"{t}_gt" if f"{t}_gt" in merged.columns else t
        pred_col = f"{t}_pred" if f"{t}_pred" in merged.columns else t
        y_true = merged[gt_col].values.astype(int)
        y_pred = merged[pred_col].values
        if len(np.unique(y_true)) > 1:
            aucs.append(roc_auc_score(y_true, y_pred))
    
    assert len(aucs) == 12, "Did not evaluate all 12 target pathologies"
    macro_auc = np.mean(aucs)
    assert macro_auc >= 0.950, f"Full Macro AUC {macro_auc:.4f} is below 0.950"

def test_holdout_split_macro_auc_gte_0950(gold_evaluation_data):
    merged = gold_evaluation_data[gold_evaluation_data["split"] == "holdout"]
    aucs = []
    for t in TARGET_NAMES:
        gt_col = f"{t}_gt" if f"{t}_gt" in merged.columns else t
        pred_col = f"{t}_pred" if f"{t}_pred" in merged.columns else t
        y_true = merged[gt_col].values.astype(int)
        y_pred = merged[pred_col].values
        if len(np.unique(y_true)) > 1:
            aucs.append(roc_auc_score(y_true, y_pred))
    
    assert len(aucs) == 12, "Did not evaluate all 12 target pathologies on holdout"
    macro_auc = np.mean(aucs)
    assert macro_auc >= 0.950, f"Holdout Macro AUC {macro_auc:.4f} is below 0.950"

def test_dev_split_macro_auc_gte_0950(gold_evaluation_data):
    merged = gold_evaluation_data[gold_evaluation_data["split"] == "dev"]
    aucs = []
    for t in TARGET_NAMES:
        gt_col = f"{t}_gt" if f"{t}_gt" in merged.columns else t
        pred_col = f"{t}_pred" if f"{t}_pred" in merged.columns else t
        y_true = merged[gt_col].values.astype(int)
        y_pred = merged[pred_col].values
        if len(np.unique(y_true)) > 1:
            aucs.append(roc_auc_score(y_true, y_pred))
    
    assert len(aucs) == 12, "Did not evaluate all 12 target pathologies on dev"
    macro_auc = np.mean(aucs)
    assert macro_auc >= 0.950, f"Dev Macro AUC {macro_auc:.4f} is below 0.950"

def test_submission_artifacts_integrity():
    for fname in ["submission.csv", "submission_v44_champion_0950.csv", "submission_v45_ensemble_0950.csv"]:
        p = Path(fname)
        assert p.exists(), f"Submission file {fname} does not exist"
        df = pd.read_csv(p)
        assert list(df.columns) == [ID_COLUMN] + TARGET_NAMES, f"Incorrect columns in {fname}"
        assert len(df) > 0, f"Submission file {fname} is empty"
        assert df.isnull().sum().sum() == 0, f"NaN values detected in {fname}"
        vals = df[TARGET_NAMES].values
        assert (vals >= 0.0).all() and (vals <= 1.0).all(), f"Values out of [0, 1] in {fname}"
