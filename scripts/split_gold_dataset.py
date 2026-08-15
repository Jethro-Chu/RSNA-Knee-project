import pandas as pd
import numpy as np
from rsna_knee.constants import TARGET_NAMES

# Load gold studies
train_df = pd.read_csv("data/train.csv")
pseudo_v2 = pd.read_parquet("data/pseudo_labels/pseudo_labels_v2.parquet")
gold_uids = sorted(list(set(pseudo_v2.loc[pseudo_v2["has_expert_labels"] == True, "StudyInstanceUID"])))

gold_df = train_df[train_df["StudyInstanceUID"].isin(gold_uids)].sort_values("StudyInstanceUID").reset_index(drop=True)
print(f"Total Gold Studies: {len(gold_df)}")

# Deterministic multilabel stratification (70% Dev, 30% Holdout)
np.random.seed(42)

# Calculate label sum / rarity score per study to balance across splits
targets_matrix = gold_df[TARGET_NAMES].values # (58, 12)
total_per_target = targets_matrix.sum(axis=0)

# Iterative greedy stratified assignment
n_total = len(gold_df)
n_holdout_target = int(np.round(0.30 * n_total)) # 17 or 18
n_dev_target = n_total - n_holdout_target # 40 or 41

dev_indices = []
holdout_indices = []

# Sort studies by rarity of positive labels
study_rarity = []
for i in range(n_total):
    pos_targets = np.where(targets_matrix[i] == 1)[0]
    if len(pos_targets) > 0:
        rarity = np.sum(1.0 / total_per_target[pos_targets])
    else:
        rarity = 0.0
    study_rarity.append((rarity, i))

# Sort descending by rarity with fixed tiebreaker
study_rarity.sort(key=lambda x: (-x[0], x[1]))

dev_counts = np.zeros(12)
holdout_counts = np.zeros(12)

for rarity, idx in study_rarity:
    y = targets_matrix[idx]
    
    # Check current capacity
    if len(holdout_indices) >= n_holdout_target:
        dev_indices.append(idx)
        dev_counts += y
    elif len(dev_indices) >= n_dev_target:
        holdout_indices.append(idx)
        holdout_counts += y
    else:
        # Assign to split with lower relative representation for the study's positive targets
        dev_score = np.sum(dev_counts * y) / (len(dev_indices) + 1)
        holdout_score = np.sum(holdout_counts * y) / (len(holdout_indices) + 1)
        
        if holdout_score < dev_score:
            holdout_indices.append(idx)
            holdout_counts += y
        else:
            dev_indices.append(idx)
            dev_counts += y

gold_df["split"] = "dev"
gold_df.loc[holdout_indices, "split"] = "holdout"

print(f"Dev Studies: {len(dev_indices)} ({len(dev_indices)/n_total*100:.1f}%)")
print(f"Holdout Studies: {len(holdout_indices)} ({len(holdout_indices)/n_total*100:.1f}%)")

print("\n" + "=" * 65)
print(f"{'Target':<22} | {'Total Pos':<10} | {'Dev Pos':<10} | {'Holdout Pos':<12}")
print("-" * 65)
for k, t in enumerate(TARGET_NAMES):
    tot = int(total_per_target[k])
    d_p = int(dev_counts[k])
    h_p = int(holdout_counts[k])
    print(f"{t:<22} | {tot:<10} | {d_p:<10} | {h_p:<12}")

# Save split mapping
split_df = gold_df[["StudyInstanceUID", "split"] + TARGET_NAMES]
split_df.to_csv("data/gold_dev_holdout_split.csv", index=False)
print("\nSaved split to data/gold_dev_holdout_split.csv")
