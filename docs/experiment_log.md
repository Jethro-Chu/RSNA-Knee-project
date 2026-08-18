# RSNA Knee Abnormality Detection: Experiment Log

**Read this first**: the "Macro AUC (Expert Val)" column is computed against the internal 58-study holdout only. It has never tracked the real Kaggle leaderboard score well — the actual public LB scores for these same model families topped out at **0.917** (see `docs/experiment_log.md` Kaggle Submissions table below), far below the 0.9995 internal figure for EXP-13. Use the Kaggle Submissions table as ground truth; treat "Expert Val" as an internal diagnostic that appears to overfit the 58-sample holdout.

## Experiment Tracking Table (internal offline validation — not leaderboard-verified)

| Exp ID | Date | Model Family | Architecture / Loss | Views / Slices | Macro AUC (Expert Val, $N=58$) | Inference (s/study) | Notes / Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP-00** | 2026-08-13 | Model 0 (Validator) | Constant Prevalence Prior | 1 Series / 16 Slices | Baseline | <0.01s | Verified 13-col schema, bounds [0, 1] |
| **EXP-01** | 2026-08-13 | 2.5D Single-Plane MIL | ResNet34d + Target-Specific Attn | 1 Series / 16 Slices | Baseline Ready | ~0.12s | 2.5D slice triplets with target attention |
| **EXP-02** | 2026-08-13 | Multi-View HMIL | ResNet34d + Tri-Plane Attention Fusion | Sag + Cor + Ax (36 Slices) | Advanced Multi-View | ~0.25s | Target-specific learned cross-plane fusion |
| **EXP-03** | 2026-08-13 | HMIL + Asymmetric Loss | ResNet34d + ASL ($\gamma_-=4.0, \gamma_+=0.5$) | Sag + Cor + Ax | Multi-View ASL | ~0.25s | Downweights easy negative normal anatomy |
| **EXP-04** | 2026-08-13 | HMIL Ensemble + TTA | Fold & Backbone Rank Average + Flip TTA | Sag + Cor + Ax | Full Competitive Ensemble | ~0.45s | Robust rank blending + horizontal flip TTA |
| **EXP-10** | 2026-08-14 | Phase 10 Baseline | 5-Fold HMIL ResNet Stem | Sag + Cor + Ax | 0.9090 | ~0.20s | Official baseline reproduction |
| **EXP-11** | 2026-08-15 | V41 Challenger | 2.5D Tri-Plane Multi-Head | Sag + Cor + Ax | 0.9250 | ~0.22s | Regularized rank ensembling |
| **EXP-12** | 2026-08-15 | V42 Consensus | Multi-Tier Clinical Soft Supervision | Sag + Cor + Ax | 0.9478 | ~0.25s | Consensus soft labels + Swin/ConvNeXt |
| **EXP-13** | 2026-08-16 | V44/V45 Champion | Multi-Expert Target-Specific HMIL | Sag + Cor + Ax | 0.9995 | ~0.28s | Internal holdout only — **not confirmed on leaderboard**, see below |

## Kaggle Submissions (ground truth — pulled via `kaggle competitions submissions`, 2026-08-17)

| Date | Description | Public LB Macro AUC |
| :--- | :--- | :---: |
| 2026-08-14 | Notebook RSNA_knee model \| Version 9 | 0.499 |
| 2026-08-15 | 5-Fold HMIL Ensemble with Percentile Rank Calibration | 0.562 |
| 2026-08-15 | RSNA Knee \| Version 6 | 0.911 |
| 2026-08-15 | V41 Ensemble Challenger | 0.911 |
| 2026-08-15 | V41 Challenger \| V40 Champion + Target-Specific Surgical Ensemble | 0.905 |
| 2026-08-16 | V43 Challenger \| High-Precision Surgical Calibration Engine | 0.910 |
| 2026-08-17 | Notebook RSNA Knee \| Version 10 | 0.910 |
| 2026-08-17 | Notebook RSNA Knee \| Version 13 | 0.911 |
| 2026-08-17 | **Version 14: V48 Frontier (E10 Alpha 0.60 + Diverse RadImageNet) — best** | **0.917** |

