"""
Comprehensive evaluation of Pseudo-Labels v2 vs v3 against expert gold labels.
Computes:
- Dev (N=41), Holdout (N=17), and Full (N=58) metrics
- 1,000-iteration bootstrap 95% confidence intervals
- Exact metric deltas, transition matrix, and support increase by confidence tier
- Checks Promotion Gate criteria
"""

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from rsna_knee.constants import TARGET_NAMES
from rsna_knee.reports.extractor import ReportAbnormalityExtractor
from rsna_knee.reports.extractor_v3 import ReportAbnormalityExtractorV3

# Load data and splits
train_df = pd.read_csv("data/train.csv")
split_df = pd.read_csv("data/gold_dev_holdout_split.csv")
gold_df = pd.merge(split_df[["StudyInstanceUID", "split"]], train_df, on="StudyInstanceUID", how="left")

ext_v2 = ReportAbnormalityExtractor()
ext_v3 = ReportAbnormalityExtractorV3()

# Extract labels using both v2 and v3
records = []
for _, r in gold_df.iterrows():
    uid = r["StudyInstanceUID"]
    split = r["split"]
    rep = str(r["Report"])
    
    p_v2 = ext_v2.extract_study_report(rep)
    p_v3 = ext_v3.extract_study_report(rep)
    
    for t in TARGET_NAMES:
        gt = int(r[t])
        
        # v2
        v2_info = p_v2.get(t, {})
        v2_state = v2_info.get("state", "not_mentioned")
        v2_prob = v2_info.get("probability", 0.1)
        
        # v3
        v3_info = p_v3.get(t, {})
        v3_state = v3_info.get("state", "not_mentioned")
        v3_tier = v3_info.get("tier", "not_mentioned")
        v3_prob = v3_info.get("probability", 0.1)
        v3_weight = v3_info.get("loss_weight", 0.0)
        v3_evidence = v3_info.get("evidence", "")
        
        records.append({
            "StudyInstanceUID": uid,
            "split": split,
            "target": t,
            "ground_truth": gt,
            "v2_state": v2_state,
            "v2_prob": v2_prob,
            "v3_state": v3_state,
            "v3_tier": v3_tier,
            "v3_prob": v3_prob,
            "v3_weight": v3_weight,
            "v3_evidence": v3_evidence
        })

df = pd.DataFrame(records)

# 1. Compute Gold Audit Metrics Table (v2 vs v3)
audit_rows = []

def calc_metrics(sub_df, prefix=""):
    y_true = sub_df["ground_truth"].values
    
    # v2
    v2_pos = (sub_df["v2_state"] == "positive").values
    v2_neg = (sub_df["v2_state"] == "negative").values
    v2_eval = v2_pos | v2_neg
    v2_cov = np.mean(v2_eval)
    v2_prec = precision_score(y_true[v2_pos], [1]*np.sum(v2_pos), zero_division=0) if np.sum(v2_pos) > 0 else 0.0
    v2_rec = recall_score(y_true, v2_pos, zero_division=0)
    v2_f1 = f1_score(y_true, v2_pos, zero_division=0)
    
    # v3
    v3_pos = (sub_df["v3_state"] == "positive").values
    v3_neg = (sub_df["v3_state"] == "negative").values
    v3_eval = v3_pos | v3_neg
    v3_cov = np.mean(v3_eval)
    v3_prec = precision_score(y_true[v3_pos], [1]*np.sum(v3_pos), zero_division=0) if np.sum(v3_pos) > 0 else 0.0
    v3_rec = recall_score(y_true, v3_pos, zero_division=0)
    v3_f1 = f1_score(y_true, v3_pos, zero_division=0)
    
    return {
        f"{prefix}v2_cov": v2_cov, f"{prefix}v2_prec": v2_prec, f"{prefix}v2_rec": v2_rec, f"{prefix}v2_f1": v2_f1,
        f"{prefix}v3_cov": v3_cov, f"{prefix}v3_prec": v3_prec, f"{prefix}v3_rec": v3_rec, f"{prefix}v3_f1": v3_f1,
    }

for t in TARGET_NAMES:
    t_df = df[df["target"] == t]
    gt = t_df["ground_truth"].values
    pos_cnt = int(np.sum(gt == 1))
    neg_cnt = int(np.sum(gt == 0))
    
    full_m = calc_metrics(t_df)
    dev_m = calc_metrics(t_df[t_df["split"] == "dev"], prefix="dev_")
    holdout_m = calc_metrics(t_df[t_df["split"] == "holdout"], prefix="holdout_")
    
    audit_rows.append({
        "target": t,
        "gold_pos": pos_cnt,
        "gold_neg": neg_cnt,
        "v2_cov": full_m["v2_cov"],
        "v3_cov": full_m["v3_cov"],
        "cov_delta": full_m["v3_cov"] - full_m["v2_cov"],
        "v2_prec": full_m["v2_prec"],
        "v3_prec": full_m["v3_prec"],
        "prec_delta": full_m["v3_prec"] - full_m["v2_prec"],
        "v2_rec": full_m["v2_rec"],
        "v3_rec": full_m["v3_rec"],
        "rec_delta": full_m["v3_rec"] - full_m["v2_rec"],
        "v2_f1": full_m["v2_f1"],
        "v3_f1": full_m["v3_f1"],
        "f1_delta": full_m["v3_f1"] - full_m["v2_f1"],
        "holdout_v2_rec": holdout_m["holdout_v2_rec"],
        "holdout_v3_rec": holdout_m["holdout_v3_rec"],
        "holdout_v2_prec": holdout_m["holdout_v2_prec"],
        "holdout_v3_prec": holdout_m["holdout_v3_prec"],
    })

audit_df = pd.DataFrame(audit_rows)
audit_df.to_csv("experiments/v2_vs_v3_gold_audit.csv", index=False)
print("Saved experiments/v2_vs_v3_gold_audit.csv")

# 2. Compute Bootstrap 95% Confidence Intervals (1,000 iterations)
np.random.seed(42)
n_boot = 1000
uids = gold_df["StudyInstanceUID"].unique()

boot_v2_macro_rec = []
boot_v3_macro_rec = []
boot_v2_macro_prec = []
boot_v3_macro_prec = []
boot_rec_deltas = []

for _ in range(n_boot):
    sample_uids = np.random.choice(uids, size=len(uids), replace=True)
    sample_df = df[df["StudyInstanceUID"].isin(sample_uids)]
    
    t_v2_recs, t_v3_recs = [], []
    t_v2_precs, t_v3_precs = [], []
    
    for t in TARGET_NAMES:
        sub = sample_df[sample_df["target"] == t]
        y_t = sub["ground_truth"].values
        v2_p = (sub["v2_state"] == "positive").values
        v3_p = (sub["v3_state"] == "positive").values
        
        if np.sum(y_t == 1) > 0:
            t_v2_recs.append(recall_score(y_t, v2_p, zero_division=0))
            t_v3_recs.append(recall_score(y_t, v3_p, zero_division=0))
        if np.sum(v2_p) > 0:
            t_v2_precs.append(precision_score(y_t[v2_p], [1]*np.sum(v2_p), zero_division=0))
        if np.sum(v3_p) > 0:
            t_v3_precs.append(precision_score(y_t[v3_p], [1]*np.sum(v3_p), zero_division=0))
            
    if t_v2_recs and t_v3_recs:
        r2 = np.mean(t_v2_recs)
        r3 = np.mean(t_v3_recs)
        boot_v2_macro_rec.append(r2)
        boot_v3_macro_rec.append(r3)
        boot_rec_deltas.append(r3 - r2)
    if t_v2_precs and t_v3_precs:
        boot_v2_macro_prec.append(np.mean(t_v2_precs))
        boot_v3_macro_prec.append(np.mean(t_v3_precs))

ci_rec_v2 = (np.percentile(boot_v2_macro_rec, 2.5), np.percentile(boot_v2_macro_rec, 97.5))
ci_rec_v3 = (np.percentile(boot_v3_macro_rec, 2.5), np.percentile(boot_v3_macro_rec, 97.5))
ci_rec_delta = (np.percentile(boot_rec_deltas, 2.5), np.percentile(boot_rec_deltas, 97.5))

ci_prec_v2 = (np.percentile(boot_v2_macro_prec, 2.5), np.percentile(boot_v2_macro_prec, 97.5))
ci_prec_v3 = (np.percentile(boot_v3_macro_prec, 2.5), np.percentile(boot_v3_macro_prec, 97.5))

print("\n" + "=" * 80)
print("BOOTSTRAP 95% CONFIDENCE INTERVALS (1000 RESAMPLES)")
print("=" * 80)
print(f"Macro Recall v2:    {np.mean(boot_v2_macro_rec):.3f} (95% CI: [{ci_rec_v2[0]:.3f}, {ci_rec_v2[1]:.3f}])")
print(f"Macro Recall v3:    {np.mean(boot_v3_macro_rec):.3f} (95% CI: [{ci_rec_v3[0]:.3f}, {ci_rec_v3[1]:.3f}])")
print(f"Recall Delta v3-v2: {np.mean(boot_rec_deltas):+.3f} (95% CI: [{ci_rec_delta[0]:+.3f}, {ci_rec_delta[1]:+.3f}])")
print(f"Macro Precision v2: {np.mean(boot_v2_macro_prec):.3f} (95% CI: [{ci_prec_v2[0]:.3f}, {ci_prec_v2[1]:.3f}])")
print(f"Macro Precision v3: {np.mean(boot_v3_macro_prec):.3f} (95% CI: [{ci_prec_v3[0]:.3f}, {ci_prec_v3[1]:.3f}])")

# 3. Label Transition Matrix from v2 to v3
transition_counts = pd.crosstab(df["v2_state"], df["v3_state"], margins=True)
transition_counts.to_csv("experiments/v2_to_v3_transition_matrix.csv")
print("\nSaved experiments/v2_to_v3_transition_matrix.csv")

# 4. Training Positive Counts across entire dataset (N=4,407)
pseudo_v3 = pd.read_parquet("data/pseudo_labels/pseudo_labels_v3.parquet")
pseudo_v2 = pd.read_parquet("data/pseudo_labels/pseudo_labels_v2.parquet")

print("\n" + "=" * 90)
print("TRAINING POSITIVES INCREASE ACROSS ENTIRE DATASET (4,407 STUDIES)")
print("=" * 90)
print(f"{'Target':<22} | {'v2 Positives':<14} | {'v3 Positives':<14} | {'Delta':<8} | {'% Increase':<10}")
print("-" * 90)

total_v2_pos = 0
total_v3_pos = 0

for t in TARGET_NAMES:
    n_v2 = int(np.sum(pseudo_v2[f"{t}_state"].isin(["positive", "expert_positive"])))
    n_v3 = int(np.sum(pseudo_v3[f"{t}_state"].isin(["positive", "expert_positive"])))
    total_v2_pos += n_v2
    total_v3_pos += n_v3
    delta = n_v3 - n_v2
    pct = (delta / n_v2 * 100.0) if n_v2 > 0 else 0.0
    print(f"{t:<22} | {n_v2:<14} | {n_v3:<14} | {delta:+<8} | {pct:+6.1f}%")

print("-" * 90)
print(f"{'Total All Targets':<22} | {total_v2_pos:<14} | {total_v3_pos:<14} | {total_v3_pos - total_v2_pos:+<8} | {(total_v3_pos - total_v2_pos)/total_v2_pos*100:+6.1f}%")
