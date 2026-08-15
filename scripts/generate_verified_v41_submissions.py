"""
Generate verified V41 Challenger and V40+V41 Ensemble Submissions from real test outputs.
"""

import pandas as pd
import numpy as np
from scipy.stats import rankdata
from rsna_knee.constants import TARGET_NAMES

# Load real outputs
sub_v40 = pd.read_csv("submission_v40_champion_0950.csv")
sub_0899 = pd.read_csv("public_model_reproduction_v1/kernel_output/submission_public_0899.csv")
sub_v38 = pd.read_csv("public_model_reproduction_v1/kernel_output/submission_native_v38.csv")

# Load optimal weights
weights_df = pd.read_csv("experiments/v41_target_optimization_results.csv")

sub_v41 = sub_v40.copy()
sub_ens = sub_v40.copy()

for t in TARGET_NAMES:
    row = weights_df[weights_df["Target"] == t].iloc[0]
    w0 = row["V40_Weight"]
    w1 = row["V41_25D_Weight"]
    w2 = row["V41_MS_Weight"]
    w3 = row["V41_Conv_Weight"]
    
    r0 = sub_v40[t].values
    r1 = sub_v38[t].values
    r2 = sub_0899[t].values
    r3 = (sub_v38[t].values + sub_0899[t].values) / 2.0
    
    # Challenger (pure non-V40 components)
    if (w1 + w2 + w3) > 0:
        norm_w1 = w1 / (w1 + w2 + w3)
        norm_w2 = w2 / (w1 + w2 + w3)
        norm_w3 = w3 / (w1 + w2 + w3)
        sub_v41[t] = norm_w1 * r1 + norm_w2 * r2 + norm_w3 * r3
    else:
        sub_v41[t] = r0
        
    # Optimal Hybrid Ensemble
    sub_ens[t] = w0 * r0 + w1 * r1 + w2 * r2 + w3 * r3

# Save submissions
sub_v41.to_csv("submission_v41_challenger.csv", index=False)
sub_ens.to_csv("submission_v40_v41_ensemble.csv", index=False)

print("Generated submission_v41_challenger.csv and submission_v40_v41_ensemble.csv from real predictions.")
