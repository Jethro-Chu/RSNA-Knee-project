# RSNA Knee Abnormality Detection — Project Context & Work Summary

---

## 1. Executive Summary & Competition Objective

- **Competition**: RSNA Knee Abnormality Detection ([Kaggle](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection))
- **Objective**: Develop an automated machine learning system that ingests complete knee MRI examinations (multi-series DICOM volumes) and predicts calibrated probabilities for **12 official target abnormalities**.
- **Evaluation Metric**: Unweighted **Macro-Average ROC-AUC** across the 12 targets:
  1. `ACL` (Anterior Cruciate Ligament Tear)
  2. `MCL` (Medial Collateral Ligament Tear)
  3. `Medial Meniscus` (Medial Meniscal Tear)
  4. `Lateral Meniscus` (Lateral Meniscal Tear)
  5. `Medial OA` (Medial Compartment Osteoarthritis)
  6. `Lateral OA` (Lateral Compartment Osteoarthritis)
  7. `PF OA` (Patellofemoral Osteoarthritis)
  8. `Effusion` (Joint Effusion)
  9. `Synovitis` (Synovial Thickening/Inflammation)
  10. `Baker's` (Baker's / Popliteal Cyst)
  11. `Contusion` (Bone Contusion / Edema)
  12. `Fracture` (Bone Fracture)

- **Official Output Format**: Exact 13-column `submission.csv` (`StudyInstanceUID` + 12 targets).

---

## 2. Core Problem Dynamics & Data Reality

| Property | Reality in Competition Dataset | Engineering Solution Implemented |
| :--- | :--- | :--- |
| **Weak Labeling** | Only **58 studies** have gold-standard expert labels; **4,349 studies** contain free-text multilingual radiology reports. | Built a deterministic **Multilingual Clinical NLP Extractor** (English, Spanish, French, German) with negation scoping to generate soft pseudo-labels and loss masks. |
| **Multi-Plane MRI** | Pathologies are strictly view-dependent (e.g. MCL on Coronal, Effusion on Axial, ACL on Sagittal). | Engineered a **Tri-Plane Multimodal HMIL Network** simultaneously ingesting Sagittal, Coronal, and Axial series. |
| **Slice Disorder** | Raw DICOM filenames are arbitrary and do not guarantee spatial sequence order. | Implemented **3D Slice Normal Vector Projection** ($\mathbf{n} = \frac{\mathbf{r} \times \mathbf{c}}{\|\mathbf{r} \times \mathbf{c}\|}$) to order slices along physical anatomy. |
| **Class Imbalance** | Normal knee structures heavily outnumber rare pathologies (fractures, high-grade tears). | Implemented **Asymmetric Loss (ASL)** ($\gamma_- = 4.0, \gamma_+ = 0.5$) to down-weight easy negative gradients. |
| **Scanner Variability** | 1.5T vs 3.0T field strengths and manufacturers create contrast variance. | Built a **16-dimensional Tabular Metadata Branch** fusing patient sex, scanner field strength, manufacturer, and geometry priors. |

---

## 3. Work Accomplished & Architectural Milestones

### A. Multilingual Clinical NLP Pseudo-Labeling Engine
- **Files**: [`src/rsna_knee/reports/extractor.py`](src/rsna_knee/reports/extractor.py), [`configs/target_ontology.yaml`](configs/target_ontology.yaml), [`scripts/generate_pseudo_labels.py`](scripts/generate_pseudo_labels.py).
- **Functionality**: Distinguishes 4 semantic states across reports:
  - `positive`: $p = 0.95$, loss mask = 1.0, confidence = 0.90
  - `negative`: $p = 0.05$, loss mask = 1.0, confidence = 0.90
  - `uncertain`: $p = 0.50$, loss mask = 0.0
  - `not_mentioned`: $p = 0.10$, loss mask = 0.0
- **Executed**: Processed all 4,407 training reports into `data/pseudo_labels/pseudo_labels_v1.parquet`.

### B. Physically-Correct Geometric DICOM Engine
- **Files**: [`src/rsna_knee/data/dicom.py`](src/rsna_knee/data/dicom.py), [`src/rsna_knee/data/series.py`](src/rsna_knee/data/series.py).
- **Functionality**: Extracts direction cosines (`ImageOrientationPatient`) and position (`ImagePositionPatient`), projects onto plane normal, and groups into Sagittal, Coronal, Axial series. Normalizes intensities using 0.5%–99.5% percentile windowing and builds 2.5D slice stacks $(z-1, z, z+1)$.

### C. Multimodal Tri-Plane HMIL Model Architecture
- **Files**: [`src/rsna_knee/models/multimodal_hmil.py`](src/rsna_knee/models/multimodal_hmil.py), [`src/rsna_knee/models/pooling.py`](src/rsna_knee/models/pooling.py), [`src/rsna_knee/models/asymmetric_loss.py`](src/rsna_knee/models/asymmetric_loss.py).
- **Vision Branch**: Ingests Sagittal, Coronal, and Axial slice stacks $\rightarrow$ 2D slice encoders $\rightarrow$ 12 Target-Specific Slice Attention Heads per plane $\rightarrow$ Learned Cross-Plane Gated Fusion.
- **Tabular Metadata Branch**: Ingests 16-dim DICOM feature vector $\rightarrow$ MLP projection $\rightarrow$ Gated Multimodal Fusion Head.

### D. Leakage-Free Validation Strategy & Full Test Suite
- **Files**: [`src/rsna_knee/training/folds.py`](src/rsna_knee/training/folds.py), [`scripts/make_folds.py`](scripts/make_folds.py), [`tests/`](tests/).
- **Folds**: Study-level Stratified 5-Fold GroupKFold (`data/metadata/folds.csv`) preventing scanner or subject leakage.
- **Unit Tests**: **29 automated unit tests** (metrics, DICOM sorting, multi-view forward pass, metadata extraction, NLP, submission schema) all passing with 100% success.

---

## 4. Kaggle Infrastructure & Deployment Log

- **Kaggle Account**: `chujethro`
- **Team Shared Notebook**: [`wenwen12/rsna-knee`](https://www.kaggle.com/code/wenwen12/rsna-knee)
- **Personal Notebook**: [`chujethro/rsna-knee`](https://www.kaggle.com/code/chujethro/rsna-knee)
- **GPU Training Notebook**: [`chujethro/rsna-knee-training`](https://www.kaggle.com/code/chujethro/rsna-knee-training)
- **Checkpoints Dataset**: [`chujethro/rsna-knee-checkpoints`](https://www.kaggle.com/datasets/chujethro/rsna-knee-checkpoints)

### Deployment Versions Summary:
- **Version 1–3**: Pulled and audited original 1-cell starter template.
- **Version 4**: Deployed full Multi-View HMIL offline inference pipeline. (Identified and fixed Kaggle Papermill kernelspec metadata requirement).
- **Version 5**: Fixed kernelspec metadata; ran to completion in 23 seconds.
- **Version 6**: Added Multimodal vision + tabular metadata branch and calibrated clinical prevalence priors.
- **Version 7**: Attached private Kaggle dataset (`chujethro/rsna-knee-checkpoints`) and integrated weight loading for `model_fold_4_best.pt`.
- **Version 8**: Added automated validation performance report printing in the Kaggle runtime log.

---

## 5. GitHub Repository & Collaboration

- **GitHub URL**: [https://github.com/Jethro-Chu/RSNA-Knee-project](https://github.com/Jethro-Chu/RSNA-Knee-project)
- **Branch**: `main`
- **Collaborator Invited**: `@ofeng1` (Write access granted via GitHub API).
- **Documentation**: Comprehensive [`README.md`](README.md) with ASCII architecture diagrams, benchmark tables, and CLI instructions.

---

## 6. Analysis of Initial Submission (Score: 0.499) & Next Step

### Root Cause of 0.499:
- In ROC-AUC evaluation, **0.499 / 0.500** represents a constant prediction or untrained model baseline.
- `model_fold_4_best.pt` was an architectural initialization checkpoint saved before executing full multi-epoch gradient descent on the 500 GB DICOM volume data.

### Concrete Next Step to Reach 0.90+ Leaderboard:
1. Run multi-epoch training on Kaggle GPU T4 x2 using `01_kaggle_train.ipynb` / `train.py` across all 4,407 studies.
2. Optimize with `AsymmetricLoss` ($\gamma_-=4.0, \gamma_+=0.5$) using cosine learning rate annealing.
3. Save the trained 5-fold checkpoints, attach to `rsna-knee.ipynb`, and submit to produce genuine, high-ranking discriminating predictions.
