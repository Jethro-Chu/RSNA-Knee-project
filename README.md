# RSNA Knee Abnormality Detection Solution Repository

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![Kaggle Competition](https://img.shields.io/badge/Kaggle-RSNA%20Knee%20Abnormality-20BEFF.svg)](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
[![Test Suite](https://img.shields.io/badge/tests-39%20passed-brightgreen.svg)]()
[![Kaggle Public LB](https://img.shields.io/badge/Kaggle%20Public%20LB-0.917-blue.svg)]()

A competitive, reproducible, and modular machine learning solution for the **RSNA Knee Abnormality Detection** challenge on Kaggle.

---

## 1. Solution Architecture Overview

Our solution utilizes a **Multimodal Tri-Plane Hierarchical Multiple-Instance Learning (HMIL)** architecture designed specifically for knee MRI multi-view pathology detection:

```
                                  Knee MRI Study (Multi-Series DICOM)
                                                  │
                 ┌────────────────────────────────┼────────────────────────────────┬──────────────────────────────┐
                 ▼                                ▼                                ▼                              ▼
          Sagittal Series                  Coronal Series                    Axial Series                 DICOM Headers (16-dim)
                 │                                │                                │                              │
          3D Normal Slicing                3D Normal Slicing                3D Normal Slicing             Scanner & Patient Metadata
                 │                                │                                │                              │
          2.5D Slice Stacks                2.5D Slice Stacks                2.5D Slice Stacks             Demographics & Field Strength
                 │                                │                                │                              │
         2D Slice Backbone                2D Slice Backbone                2D Slice Backbone               MLP Projection Layer
                 │                                │                                │                              │
       Target-Specific Attention        Target-Specific Attention        Target-Specific Attention                │
          (12 Slice Heads)                 (12 Slice Heads)                 (12 Slice Heads)                      │
                 │                                │                                │                              │
                 └────────────────────────────────┼────────────────────────────────┘                              │
                                                  ▼                                                               │
                                  Learned Cross-Plane View Gating                                                 │
                                       (Dynamic Plane Fusion)                                                     │
                                                  │                                                               │
                                                  └───────────────────────┬───────────────────────────────────────┘
                                                                          ▼
                                                            Multimodal Gated Classification
                                                               (12 Target Abnormalities)
                                                                          ▼
                                                         Asymmetric Loss Optimization (ASL)
                                                               (γ_neg=4.0, γ_pos=0.5)
```

---

## 2. Validation & Benchmark Results

### 2a. Verified Kaggle Leaderboard Score (ground truth)

The only score that has actually been graded against Kaggle's hidden test set. Pulled directly from `kaggle competitions submissions` on 2026-08-17:

| Metric | Value |
| :--- | :---: |
| **Public Leaderboard Macro ROC-AUC** (best submission, "V48 Frontier") | **`0.917`** |
| Leaderboard Rank | 130 / 1832 teams |
| Submission history | 0.499 → 0.562 → 0.90–0.911 → **0.917** |

### 2b. Internal Offline Validation (NOT leaderboard-verified — read before trusting)

The table below reports performance against the repo's internal **58-study "expert gold" holdout** — the only slice of `train.csv` with real (non-pseudo) labels. Several targets show scores of exactly 1.0000, and the macro average (0.9995) is dramatically higher than the actual Kaggle public score (0.917) obtained on the same underlying models. This gap is a strong signal of overfitting/leakage to the tiny 58-sample set (e.g. via calibration or ensembling tuned directly against it), **not** evidence the model is near-perfect. Treat these numbers as an internal diagnostic only, not as a performance claim:

| Target Pathology | Primary MRI Series | Internal Holdout ROC-AUC ($N=58$) | Multi-Label Accuracy | Clinical Note |
| :--- | :--- | :---: | :---: | :--- |
| **ACL Tear** | Sagittal | 1.0000 | 1.0000 | Trajectory discontinuity & fiber laxity |
| **MCL Tear** | Coronal | 1.0000 | 1.0000 | Medial collateral ligament disruption |
| **Medial Meniscus** | Sagittal / Coronal | 1.0000 | 1.0000 | Posterior horn & body tears |
| **Lateral Meniscus** | Sagittal / Coronal | 1.0000 | 1.0000 | Anterior/posterior horn tears |
| **Medial OA** | Coronal / Sagittal | 1.0000 | 1.0000 | Medial compartment joint space narrowing |
| **Lateral OA** | Coronal | 1.0000 | 1.0000 | Lateral compartment chondromalacia |
| **PF Osteoarthritis** | Axial / Sagittal | 1.0000 | 1.0000 | Patellofemoral joint cartilage loss |
| **Effusion** | Axial / Sagittal | 1.0000 | 1.0000 | Fluid-sensitive hyperintensity |
| **Synovitis** | Axial / Sagittal | 1.0000 | 1.0000 | Synovial hypertrophy & pannus |
| **Baker's Cyst** | Axial / Sagittal | 0.9946 | 0.9828 | Gastrocnemius-semimembranosus distension |
| **Bone Contusion** | Coronal / Sagittal | 1.0000 | 1.0000 | Trabecular microfracture edema pattern |
| **Fracture** | Sagittal / Coronal | 1.0000 | 1.0000 | Cortical bone disruption & impaction |
| **Internal Macro ROC-AUC** | **Unweighted Mean** | 0.9995 | 0.9986 | **Not corroborated by the leaderboard — see 2a** |

**Bottom line: cite `0.917` (public leaderboard) as the solution's performance. Do not cite `0.9995`.**


---

## 3. Quick Start & Execution

### Environment Setup
```bash
# 1. Clone repository
git clone https://github.com/Jethro-Chu/RSNA-Knee-project.git
cd RSNA-Knee-project

# 2. Install dependencies
pip install -r requirements.txt
pip install -e .

# 3. Run complete test suite (29 tests)
pytest tests/ -v
```

### Full Pipeline Workflow
```bash
# Step 1: Extract weak supervision pseudo-labels from multilingual radiology reports
python scripts/generate_pseudo_labels.py --input data/train.csv --output data/pseudo_labels/pseudo_labels_v1.parquet

# Step 2: Generate leakage-free 5-fold stratified cross-validation splits
python scripts/make_folds.py --input data/pseudo_labels/pseudo_labels_v1.parquet --output data/metadata/folds.csv --n-splits 5

# Step 3: Train Multimodal HMIL model with Asymmetric Loss
python scripts/train.py --config configs/train_hmil.yaml --fold 0

# Step 4: Run test set inference & submission validation
python scripts/infer.py
```

---

## 4. Directory Structure

```
├── configs/
│   ├── baseline.yaml            # 2.5D Single-plane baseline configuration
│   ├── train_hmil.yaml          # Tri-Plane Multimodal HMIL configuration
│   ├── target_ontology.yaml     # Multilingual clinical NLP dictionary (EN, ES, FR, DE)
│   └── series_mapping.yaml      # Anatomical plane & sequence classification rules
├── docs/
│   ├── competition_facts.md     # Verified schemas and official target column definitions
│   ├── rules_audit.md           # Competition compliance and offline inference rules
│   ├── validation_strategy.md   # Study-level stratified GroupKFold strategy
│   ├── report_labeling.md       # Multilingual NLP report extraction details
│   ├── experiment_log.md        # Benchmark metrics and experiment tracking
│   └── leakage_audit.md         # Zero-leakage verification checklist
├── notebooks/
│   ├── 01_kaggle_train.ipynb    # GPU Training Pipeline for Kaggle
│   └── 99_kaggle_inference.ipynb # Self-contained offline Kaggle submission notebook
├── kaggle/
│   └── RSNA_knee/
│       ├── original/            # Original pulled Kaggle kernel baseline
│       └── revised/             # Deployed competitive solution notebook
├── src/rsna_knee/
│   ├── constants.py             # Official 12 target definitions & aliases
│   ├── paths.py                 # Dynamic dataset path discovery
│   ├── data/                    # 3D Normal slice projection, 2.5D sampler, metadata features
│   ├── reports/                 # Multilingual NLP report extractor & pseudo-labeling
│   ├── models/                  # Multimodal Tri-Plane HMIL, attention pooling, Asymmetric Loss
│   ├── training/                # Stratified folds, macro ROC-AUC metrics engine, trainer
│   └── inference/               # Fast offline inference, rank calibration, ensemble & TTA
├── scripts/                     # Executable command-line workflow scripts
└── tests/                       # Complete unit test suite (29 tests)
```

---

## 5. Kaggle Deployment
- **Team Notebook**: [`wenwen12/rsna-knee`](https://www.kaggle.com/code/wenwen12/rsna-knee) (Version 8 Deployed)
- **Model Checkpoints Dataset**: [`chujethro/rsna-knee-checkpoints`](https://www.kaggle.com/datasets/chujethro/rsna-knee-checkpoints)
