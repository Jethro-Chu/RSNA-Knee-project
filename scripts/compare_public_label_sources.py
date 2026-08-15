"""
Stage 5: Deep Label Source Analysis across full 4,407 dataset and on 58 Gold Studies.
Evaluates:
- Positive prevalence across 4,407 studies per target
- Inter-annotator agreement (Cohen's Kappa / Pearson correlation)
- Raw LLM extraction performance vs Expert Ground Truth
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from rsna_knee.constants import TARGET_NAMES
from rsna_knee.reports.extractor_v3 import ReportAbnormalityExtractorV3

# Load Ground Truth
train_df = pd.read_csv("data/train.csv")
split_df = pd.read_csv("data/gold_dev_holdout_split.csv")
gold_uids = split_df["StudyInstanceUID"].values
gold_df = train_df[train_df["StudyInstanceUID"].isin(gold_uids)].sort_values("StudyInstanceUID").reset_index(drop=True)

# Load Public Datasets
pilk_df = pd.read_csv("public_model_reproduction_v1/datasets/rsna-knee-llm-labels/report_labels_v2.csv")
steven_df = pd.read_csv("public_model_reproduction_v1/datasets/rsna-knee-llm-report-labels/llm_labels_full.csv")
sol_df = pd.read_csv("public_model_reproduction_v1/datasets/rsna-knee-llm-report-labels-sol56/report_labels_gpt56sol.csv")
pseudo_v3 = pd.read_parquet("data/pseudo_labels/pseudo_labels_v3.parquet")

# 1. Dataset-Wide Positive Prevalence Comparison (N=4,407)
print("="*95)
print("DATASET-WIDE POSITIVE PREVALENCE COMPARISON (4,407 STUDIES)")
print("="*95)
print(f"{'Target':<22} | {'Pilkwang YES':<14} | {'Steven (>0.7)':<14} | {'GPT-5.6 (==1.0)':<16} | {'Pseudo v3 Pos':<14}")
print("-" * 95)

for t in TARGET_NAMES:
    # Pilkwang
    if f"{t}__verdict" in pilk_df.columns:
        p_cnt = int(np.sum(pilk_df[f"{t}__verdict"] == "YES"))
    elif t in pilk_df.columns:
        p_cnt = int(np.sum(pilk_df[t] > 0.5))
    else:
        p_cnt = 0
        
    # Steven
    st_cnt = int(np.sum(steven_df[t] > 0.70)) if t in steven_df.columns else 0
    
    # Sol
    sol_cnt = int(np.sum(sol_df[t] == 1.0)) if t in sol_df.columns else 0
    
    # v3
    v3_cnt = int(np.sum(pseudo_v3[f"{t}_state"] == "positive"))
    
    print(f"{t:<22} | {p_cnt:<14} | {st_cnt:<14} | {sol_cnt:<16} | {v3_cnt:<14}")

# 2. Inter-Source Correlation Matrix across full dataset
print("\n" + "="*95)
print("INTER-LABEL-SOURCE CORRELATION MATRIX (PEARSON R ACROSS ALL 4,407 STUDIES)")
print("="*95)

merged_all = pd.merge(pilk_df[["StudyInstanceUID"] + TARGET_NAMES], steven_df[["StudyInstanceUID"] + TARGET_NAMES], on="StudyInstanceUID", suffixes=("_pilk", "_steven"))
merged_all = pd.merge(merged_all, sol_df[["StudyInstanceUID"] + TARGET_NAMES].add_suffix("_sol").rename(columns={"StudyInstanceUID_sol": "StudyInstanceUID"}), on="StudyInstanceUID")

corrs = []
for t in TARGET_NAMES:
    r_ps = np.corrcoef(merged_all[f"{t}_pilk"], merged_all[f"{t}_steven"])[0, 1]
    r_p_sol = np.corrcoef(merged_all[f"{t}_pilk"], merged_all[f"{t}_sol"])[0, 1]
    r_s_sol = np.corrcoef(merged_all[f"{t}_steven"], merged_all[f"{t}_sol"])[0, 1]
    corrs.append((r_ps, r_p_sol, r_s_sol))
    print(f"{t:<22} | Pilkwang vs Steven: {r_ps:.3f} | Pilkwang vs Sol: {r_p_sol:.3f} | Steven vs Sol: {r_s_sol:.3f}")

mean_corrs = np.mean(corrs, axis=0)
print("-" * 95)
print(f"{'Mean All Targets':<22} | Pilkwang vs Steven: {mean_corrs[0]:.3f} | Pilkwang vs Sol: {mean_corrs[1]:.3f} | Steven vs Sol: {mean_corrs[2]:.3f}")
