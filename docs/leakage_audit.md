# RSNA Knee Abnormality Detection: Leakage Audit

## Leakage Audit Checklist

- [x] **Study-Level Splitting**: Validation folds are grouped strictly by `StudyInstanceUID`. No slices or series from a validation study appear in training.
- [x] **Report Separation**: Reports are used exclusively during training to create weak pseudo-labels; zero report dependencies exist in the test inference pipeline.
- [x] **Image Preprocessing**: Percentile intensity normalization and 2.5D slice sampling are computed per-series independently without cross-study statistics.
- [x] **No Target Leakage**: DICOM metadata tag allowlisting excludes non-image diagnosis annotations.
- [x] **Offline Compliance**: All weights and dependencies are bundled locally; zero external internet calls during test inference.
