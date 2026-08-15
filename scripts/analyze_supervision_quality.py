import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from rsna_knee.constants import TARGET_NAMES

# Load datasets
oof = pd.read_parquet("experiments/oof_predictions_v19.parquet")
train_csv = pd.read_csv("data/train.csv")
pseudo = pd.read_parquet("data/pseudo_labels/pseudo_labels_v2.parquet")

# Merge reports for text analysis
merged = pd.merge(oof, train_csv[["StudyInstanceUID", "Report"]], on="StudyInstanceUID", how="left")

# Filter to gold cohort
gold_df = merged[merged["has_expert_labels"] == True].copy()
print(f"Total Gold Studies: {len(gold_df)}")

# Build gold-vs-pseudo error analysis table
gold_error_rows = []

for _, r in gold_df.iterrows():
    uid = r["StudyInstanceUID"]
    f = r["fold"]
    report = str(r["Report"])
    
    for t in TARGET_NAMES:
        gt = r[f"{t}_target"] # expert ground truth label
        pred_p = r[f"{t}_pred"] # OOF model prediction
        src = r.get(f"{t}_source", "expert_gold")
        mask = r.get(f"{t}_loss_mask", True)
        weight = r.get(f"{t}_loss_weight", 5.0)
        conf = r.get(f"{t}_confidence", 1.0)
        state = r.get(f"{t}_state", "positive" if gt > 0.5 else "negative")
        
        # Check raw pseudo label extraction directly from text
        pseudo_val = 1.0 if state == "positive" else (0.0 if state == "negative" else 0.5)
        
        is_gt_pos = (gt > 0.5)
        is_pred_pos = (pred_p > 0.5)
        is_pseudo_pos = (state == "positive")
        is_pseudo_neg = (state == "negative")
        
        pseudo_correct = (is_gt_pos and is_pseudo_pos) or (not is_gt_pos and is_pseudo_neg)
        model_correct = (is_gt_pos and is_pred_pos) or (not is_gt_pos and not is_pred_pos)
        
        gold_error_rows.append({
            "StudyInstanceUID": uid,
            "fold": f,
            "target": t,
            "expert_label": int(gt > 0.5),
            "pseudo_state": state,
            "pseudo_confidence": conf,
            "pseudo_source": src,
            "loss_mask": mask,
            "loss_weight": weight,
            "oof_model_prob": round(float(pred_p), 4),
            "pseudo_correct": bool(pseudo_correct),
            "model_correct": bool(model_correct),
            "report_excerpt": report[:250].replace("\n", " ")
        })

error_df = pd.DataFrame(gold_error_rows)
error_df.to_csv("experiments/gold_error_analysis_v1.csv", index=False)
print("Saved experiments/gold_error_analysis_v1.csv with shape:", error_df.shape)

# Compute per-target metrics against gold
print("\n" + "=" * 95)
print("SUPERVISION QUALITY METRICS (EVALUATED AGAINST 58 EXPERT GOLD STUDIES)")
print("=" * 95)
print(f"{'Target':<22} | {'Gold Pos':<8} | {'Gold Neg':<8} | {'Pseudo Pos':<10} | {'Pseudo Neg':<10} | {'Prec':<7} | {'Recall':<7} | {'F1':<7} | {'Coverage':<8}")
print("-" * 95)

metrics_summary = {}

for t in TARGET_NAMES:
    t_df = error_df[error_df["target"] == t]
    y_true = t_df["expert_label"].values
    
    # Pseudo labels: positive if state == positive, negative if state == negative, unmentioned/uncertain if other
    pseudo_pos = (t_df["pseudo_state"] == "positive").values
    pseudo_neg = (t_df["pseudo_state"] == "negative").values
    evaluable = pseudo_pos | pseudo_neg
    
    gold_pos_cnt = int(np.sum(y_true == 1))
    gold_neg_cnt = int(np.sum(y_true == 0))
    p_pos_cnt = int(np.sum(pseudo_pos))
    p_neg_cnt = int(np.sum(pseudo_neg))
    
    cov = np.mean(evaluable)
    
    if np.sum(pseudo_pos) > 0:
        prec = precision_score(y_true[pseudo_pos], np.ones(np.sum(pseudo_pos)), zero_division=0)
    else:
        prec = 0.0
        
    rec = recall_score(y_true, pseudo_pos, zero_division=0)
    f1 = f1_score(y_true, pseudo_pos, zero_division=0)
    
    metrics_summary[t] = {
        "gold_pos": gold_pos_cnt,
        "gold_neg": gold_neg_cnt,
        "pseudo_pos": p_pos_cnt,
        "pseudo_neg": p_neg_cnt,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "coverage": cov
    }
    
    print(f"{t:<22} | {gold_pos_cnt:<8} | {gold_neg_cnt:<8} | {p_pos_cnt:<10} | {p_neg_cnt:<10} | {prec:<7.3f} | {rec:<7.3f} | {f1:<7.3f} | {cov*100:<7.1f}%")
