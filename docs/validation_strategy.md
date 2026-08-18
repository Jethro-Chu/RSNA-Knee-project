# RSNA Knee Abnormality Detection: Validation Strategy

## 1. Core Principles
- **No Study / Patient Leakage**: Slices, series, and multi-view sequences from the same `StudyInstanceUID` must remain strictly within the same fold.
- **Stratified GroupKFold**: 5-fold cross-validation multi-label stratified across expert-annotated cases to ensure consistent positive prevalence across all folds.

---

## 2. Dual Validation Views

### View A: Gold-Standard Expert Validation (internal diagnostic only)
- Evaluated strictly on the ~58 ground-truth annotated cases.
- Computes unweighted macro-average ROC-AUC across all 12 target classes.
- **Caution**: in practice this view has *not* tracked the real Kaggle public leaderboard score — internal runs have scored up to 0.9995 here while the same models scored only 0.917 on Kaggle's actual public leaderboard (see `docs/experiment_log.md`). With only 58 samples, and given that calibration/ensembling decisions have been tuned by looking at this same set, treat it as overfit-prone rather than as a reliable leaderboard proxy. The Kaggle public leaderboard score is the only trustworthy performance number.

### View B: Pseudo-Label Validation (Secondary Signal)
- Evaluated on high-confidence pseudo-labeled studies extracted from report text.
- Used to monitor generalization across large study counts, but never treated as equivalent to human ground truth.

---

## 3. Metric Computation & Edge Case Rules
- If a fold lacks positive examples for a rare class, that class is marked as `NaN` and the macro average is calculated across the remaining valid target classes.
- Confidence intervals are computed via 500-iteration bootstrap resampling ($95\%$ CI).
