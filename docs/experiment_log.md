# RSNA Knee Abnormality Detection: Experiment Log

## Experiment Tracking Table

| Exp ID | Date | Model Family | Architecture / Loss | Views / Slices | Macro AUC (Expert Val) | Inference (s/study) | Notes / Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP-00** | 2026-08-13 | Model 0 (Validator) | Constant Prevalence Prior | 1 Series / 16 Slices | Baseline | <0.01s | Verified 13-col schema, bounds [0, 1] |
| **EXP-01** | 2026-08-13 | 2.5D Single-Plane MIL | ResNet34d + Target-Specific Attn | 1 Series / 16 Slices | Baseline Ready | ~0.12s | 2.5D slice triplets with target attention |
| **EXP-02** | 2026-08-13 | Multi-View HMIL | ResNet34d + Tri-Plane Attention Fusion | Sag + Cor + Ax (36 Slices) | Advanced Multi-View | ~0.25s | Target-specific learned cross-plane fusion |
| **EXP-03** | 2026-08-13 | HMIL + Asymmetric Loss | ResNet34d + ASL ($\gamma_-=4.0, \gamma_+=0.5$) | Sag + Cor + Ax | Multi-View ASL | ~0.25s | Downweights easy negative normal anatomy |
| **EXP-04** | 2026-08-13 | HMIL Ensemble + TTA | Fold & Backbone Rank Average + Flip TTA | Sag + Cor + Ax | Full Competitive Ensemble | ~0.45s | Robust rank blending + horizontal flip TTA |
