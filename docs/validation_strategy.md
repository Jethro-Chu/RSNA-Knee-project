# RSNA Knee Abnormality Detection: Validation Strategy

## 1. Core Principles
- **No Study / Patient Leakage**: Slices, series, and multi-view sequences from the same `StudyInstanceUID` must remain strictly within the same fold.
- **Stratified GroupKFold**: 5-fold cross-validation multi-label stratified across expert-annotated cases to ensure consistent positive prevalence across all folds.

---

## 2. Dual Validation Views

### View A: Gold-Standard Expert Validation (Primary Truth)
- Evaluated strictly on the ~58 ground-truth annotated cases.
- Computes unweighted macro-average ROC-AUC across all 12 target classes.
- Serves as the primary validation proxy for competition leaderboard ranking.

### View B: Pseudo-Label Validation (Secondary Signal)
- Evaluated on high-confidence pseudo-labeled studies extracted from report text.
- Used to monitor generalization across large study counts, but never treated as equivalent to human ground truth.

---

## 3. Metric Computation & Edge Case Rules
- If a fold lacks positive examples for a rare class, that class is marked as `NaN` and the macro average is calculated across the remaining valid target classes.
- Confidence intervals are computed via 500-iteration bootstrap resampling ($95\%$ CI).
