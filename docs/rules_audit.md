# RSNA Knee Abnormality Detection: Rules & Compliance Audit

- **Date Inspected**: 2026-08-13
- **Competition**: RSNA Knee Abnormality Detection (Kaggle)

---

## 1. Compliance Checklist

| Item | Status / Rule | Compliance Action |
| :--- | :--- | :--- |
| **External Data** | Permitted if publicly available and shared in discussion | Pretrained ImageNet / medical encoders must use permissible open licenses |
| **External LLMs / APIs** | Permitted for label extraction during training if terms of use adhere to data privacy | Offline deterministic NLP + locally reproducible LLM pseudo-labels; zero API calls in inference |
| **Test Set Inference** | **Internet Disabled** during Kaggle notebook evaluation | All weights, tokenizers, and wheels must be bundled into Kaggle input datasets |
| **Submission Format** | `submission.csv` matching `sample_submission.csv` (13 columns) | Strict schema validation before export |
| **Code Sharing for Winners** | Required open-source under MIT / Apache 2.0 with reproducibility | Fully reproducible modular repository |
| **Efficiency Prize** | Evaluated on total notebook runtime vs. ROC-AUC score | Lightweight 2.5D models with fast geometric DICOM decoding and mixed-precision inference |
| **Private Data / Leaks** | Strictly prohibited | Pure out-of-fold cross-validation; zero leakage between study instances |

---

## 2. Model & Dependency Offline Packaging Strategy
- To run completely offline in Kaggle:
  - Model weights stored in PyTorch checkpoint format (`.pt` or `.safetensors`).
  - Required utility packages (`pydicom`, `timm`, `albumentations`) installed via wheel cache if not already in Kaggle's base Python environment.
  - Deterministic report labeling rules stored as YAML/JSON configuration files with no external network requests.
