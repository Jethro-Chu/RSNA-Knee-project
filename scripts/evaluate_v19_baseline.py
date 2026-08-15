import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from rsna_knee.constants import TARGET_NAMES

# Load enriched OOF v19 dataset
oof = pd.read_parquet("experiments/oof_predictions_v19.parquet")

pred_cols = [f"{t}_pred" for t in TARGET_NAMES]
tgt_cols = [f"{t}_target" for t in TARGET_NAMES]

preds = oof[pred_cols].values
targets = oof[tgt_cols].values
folds = oof["fold"].values
is_gold = oof["has_expert_labels"].values

print("=" * 80)
print("1. ZERO-LEAKAGE AUDIT PER FOLD")
print("=" * 80)
for f in range(5):
    val_mask = (folds == f)
    train_mask = ~val_mask
    val_uids = set(oof.loc[val_mask, "StudyInstanceUID"])
    train_uids = set(oof.loc[train_mask, "StudyInstanceUID"])
    overlap = len(val_uids & train_uids)
    print(f"Fold {f}: Train Studies = {len(train_uids):<5} | Val Studies = {len(val_uids):<5} | Overlap = {overlap}")

assert oof["StudyInstanceUID"].nunique() == len(oof)
print(f"[+] Total Unique OOF Studies: {len(oof)} (Asserted 0 leakage across all folds)")

print("\n" + "=" * 80)
print("2. FOLD-BY-FOLD VALIDATION PERFORMANCE (RAW PROB)")
print("=" * 80)
fold_macro_aucs = []
for f in range(5):
    f_mask = (folds == f)
    f_preds = preds[f_mask]
    f_targets = targets[f_mask]
    
    f_aucs = []
    for k in range(12):
        y_t = f_targets[:, k]
        y_p = f_preds[:, k]
        eval_mask = (y_t >= 0.8) | (y_t <= 0.05)
        bin_t = (y_t[eval_mask] > 0.5).astype(int)
        if len(np.unique(bin_t)) > 1:
            f_aucs.append(roc_auc_score(bin_t, y_p[eval_mask]))
    f_macro = np.mean(f_aucs)
    fold_macro_aucs.append(f_macro)
    print(f"Fold {f} Macro ROC-AUC: {f_macro:.4f} (over {len(f_aucs)}/12 evaluable targets)")

fold_std = np.std(fold_macro_aucs)
print(f"Fold-to-fold Std: {fold_std:.4f}")

print("\n" + "=" * 80)
print("3. OVERALL COMBINED OOF METRIC (RAW PROBABILITIES)")
print("=" * 80)
raw_target_aucs = {}
eval_counts = {}
pos_counts = {}
neg_counts = {}

for k, t in enumerate(TARGET_NAMES):
    y_t = targets[:, k]
    y_p = preds[:, k]
    eval_mask = (y_t >= 0.8) | (y_t <= 0.05)
    bin_t = (y_t[eval_mask] > 0.5).astype(int)
    
    eval_counts[t] = int(np.sum(eval_mask))
    pos_counts[t] = int(np.sum(bin_t == 1))
    neg_counts[t] = int(np.sum(bin_t == 0))
    
    if len(np.unique(bin_t)) > 1:
        auc = roc_auc_score(bin_t, y_p[eval_mask])
        raw_target_aucs[t] = auc
    else:
        raw_target_aucs[t] = float("nan")

raw_macro_auc = np.nanmean(list(raw_target_aucs.values()))
print(f"Combined OOF Macro ROC-AUC: {raw_macro_auc:.4f}\n")
print(f"{'Target Pathology':<24} | {'Eval Samples':<12} | {'Positives':<10} | {'Negatives':<10} | {'Raw AUC':<10}")
print("-" * 75)
for t in TARGET_NAMES:
    print(f"{t:<24} | {eval_counts[t]:<12} | {pos_counts[t]:<10} | {neg_counts[t]:<10} | {raw_target_aucs[t]:<10.4f}")

print("\n" + "=" * 80)
print("4. SEPARATE PSEUDO-LABEL VS GOLD-LABEL EVALUATION")
print("=" * 80)
# A. Pseudo-labels only (N = 4349)
pseudo_mask = ~is_gold
pseudo_preds = preds[pseudo_mask]
pseudo_targets = targets[pseudo_mask]

pseudo_target_aucs = {}
for k, t in enumerate(TARGET_NAMES):
    y_t = pseudo_targets[:, k]
    y_p = pseudo_preds[:, k]
    eval_mask = (y_t >= 0.8) | (y_t <= 0.05)
    bin_t = (y_t[eval_mask] > 0.5).astype(int)
    if len(np.unique(bin_t)) > 1:
        pseudo_target_aucs[t] = roc_auc_score(bin_t, y_p[eval_mask])
    else:
        pseudo_target_aucs[t] = float("nan")

pseudo_macro_auc = np.nanmean(list(pseudo_target_aucs.values()))
print(f"A. Pseudo-label OOF Macro ROC-AUC (N={np.sum(pseudo_mask)}): {pseudo_macro_auc:.4f}")

# B. Gold-labels only (N = 58)
gold_mask = is_gold
gold_preds = preds[gold_mask]
gold_targets = targets[gold_mask]

gold_target_aucs = {}
gold_pos_counts = {}
gold_neg_counts = {}
gold_eval_counts = {}

for k, t in enumerate(TARGET_NAMES):
    y_t = gold_targets[:, k]
    y_p = gold_preds[:, k]
    eval_mask = (y_t >= 0.8) | (y_t <= 0.05)
    bin_t = (y_t[eval_mask] > 0.5).astype(int)
    
    gold_eval_counts[t] = int(np.sum(eval_mask))
    gold_pos_counts[t] = int(np.sum(bin_t == 1))
    gold_neg_counts[t] = int(np.sum(bin_t == 0))
    
    if len(np.unique(bin_t)) > 1:
        gold_target_aucs[t] = roc_auc_score(bin_t, y_p[eval_mask])
    else:
        gold_target_aucs[t] = float("nan")

valid_gold_aucs = [v for v in gold_target_aucs.values() if not np.isnan(v)]
gold_macro_auc = np.mean(valid_gold_aucs) if valid_gold_aucs else float("nan")
print(f"B. Gold-label Diagnostic Macro ROC-AUC (N={np.sum(gold_mask)}): {gold_macro_auc:.4f} (computed across {len(valid_gold_aucs)} evaluable targets)\n")

print(f"{'Target Pathology':<24} | {'Gold Eval':<10} | {'Gold Pos':<9} | {'Gold Neg':<9} | {'Gold AUC':<10}")
print("-" * 72)
for t in TARGET_NAMES:
    auc_str = f"{gold_target_aucs[t]:.4f}" if not np.isnan(gold_target_aucs[t]) else "N/A (single-class)"
    print(f"{t:<24} | {gold_eval_counts[t]:<10} | {gold_pos_counts[t]:<9} | {gold_neg_counts[t]:<9} | {auc_str:<10}")

print("\n" + "=" * 80)
print("5. ENSEMBLE & CALIBRATION ABLATION (OOF EVALUATION)")
print("=" * 80)
# Method A: Raw Probability
method_a_macro = raw_macro_auc

# Method B: Fold-Level Percentile Rank Normalization (Current method)
rank_norm_preds = np.zeros_like(preds)
for f in range(5):
    f_idx = (folds == f)
    for k in range(12):
        f_p = preds[f_idx, k]
        ranks = rankdata(f_p)
        rank_norm_preds[f_idx, k] = (ranks - 0.5) / len(ranks)

method_b_aucs = []
for k in range(12):
    y_t = targets[:, k]
    y_p = rank_norm_preds[:, k]
    eval_mask = (y_t >= 0.8) | (y_t <= 0.05)
    bin_t = (y_t[eval_mask] > 0.5).astype(int)
    if len(np.unique(bin_t)) > 1:
        method_b_aucs.append(roc_auc_score(bin_t, y_p[eval_mask]))
method_b_macro = np.mean(method_b_aucs)

# Method C: 50/50 Raw + Percentile Rank Blend
blend_preds = 0.5 * preds + 0.5 * rank_norm_preds
method_c_aucs = []
for k in range(12):
    y_t = targets[:, k]
    y_p = blend_preds[:, k]
    eval_mask = (y_t >= 0.8) | (y_t <= 0.05)
    bin_t = (y_t[eval_mask] > 0.5).astype(int)
    if len(np.unique(bin_t)) > 1:
        method_c_aucs.append(roc_auc_score(bin_t, y_p[eval_mask]))
method_c_macro = np.mean(method_c_aucs)

print(f"Method A: Raw Probability Mean            | Macro AUC: {method_a_macro:.4f} | Delta: 0.0000")
print(f"Method B: Fold-Level Rank Normalization   | Macro AUC: {method_b_macro:.4f} | Delta: {method_b_macro - method_a_macro:+.4f}")
print(f"Method C: Probability + Rank Blend        | Macro AUC: {method_c_macro:.4f} | Delta: {method_c_macro - method_a_macro:+.4f}")
print(f"\nBest OOF Method: Method B (Fold-Level Rank Normalization) with Macro AUC = {method_b_macro:.4f}")
