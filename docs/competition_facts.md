# RSNA Knee Abnormality Detection: Competition Facts

- **Competition URL**: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection
- **Date Inspected**: 2026-08-13
- **Challenge Host**: Radiological Society of North America (RSNA) & Kaggle
- **Primary Goal**: Multi-label study-level knee MRI abnormality detection across 12 target pathologies from complete knee MRI exams and training radiology reports.

---

## 1. Official Target Abnormalities and Ordering

The 12 target abnormalities evaluated on Kaggle are:
1. `ACL` (Anterior Cruciate Ligament tear / rupture)
2. `MCL` (Medial Collateral Ligament injury / sprain / tear)
3. `Medial Meniscus` (Medial meniscus tear or degeneration)
4. `Lateral Meniscus` (Lateral meniscus tear or degeneration)
5. `Medial OA` (Medial compartment osteoarthritis)
6. `Lateral OA` (Lateral compartment osteoarthritis)
7. `PF OA` (Patellofemoral osteoarthritis)
8. `Effusion` (Joint effusion)
9. `Synvitis` (Synovitis - note official Kaggle target column spelling `Synvitis` / `Synovitis`)
10. `Baker's` (Baker's cyst / Popliteal cyst)
11. `Contusion` (Bone contusion / trabecular microfracture / bone bruise)
12. `Fracture` (Cortical / subchondral / acute fracture)

---

## 2. Dataset Schema & Structure

### Training Data (`train.csv`)
- **Rows**: ~4,407 examinations (StudyInstanceUIDs).
- **Columns**:
  - `StudyInstanceUID`: Unique identifier for the knee MRI study.
  - `Report`: Free-text clinical radiology report (multilingual: English, German, French, Spanish, etc.).
  - 12 binary target columns: `ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`, `PF OA`, `Effusion`, `Synvitis`, `Baker's`, `Contusion`, `Fracture`.
- **Annotation Split**:
  - ~58 studies have gold-standard, expert human radiologist multi-label ground truth annotations.
  - ~4,349 studies have blank/unannotated labels with only the clinical radiology `Report` text provided.

### Test Data (`test.csv`)
- **Columns**: `StudyInstanceUID`.
- **Crucial Rule**: The hidden test set contains **only DICOM imaging files**, no radiology reports. All models must be image-only during test inference.

### Sample Submission (`sample_submission.csv`)
- **Columns**: `StudyInstanceUID` + 12 targets.
- **Values**: Continuous float probabilities in range $[0.0, 1.0]$.
- **Evaluation Metric**: Unweighted macro-average ROC-AUC across all 12 targets:
  $$\text{Macro AUC} = \frac{1}{12} \sum_{k=1}^{12} \text{ROC-AUC}_k$$

---

## 3. Imaging & DICOM Data Characteristics
- **Modality**: Knee Magnetic Resonance Imaging (MRI).
- **Volumes & Series**: Each study contains multiple series across anatomical planes:
  - Sagittal (T1, T2, PD, FS / STIR)
  - Coronal (PD, T2, FS)
  - Axial (PD, T2, FS)
  - Localizers / 3D reconstructions
- **Transfer Syntaxes**: Diverse set including Explicit VR Little Endian (`1.2.840.10008.1.2.1`), Implicit VR Little Endian (`1.2.840.10008.1.2`), JPEG Lossless (`1.2.840.10008.1.2.4.70`), JPEG 2000 (`1.2.840.10008.1.2.4.90/91`).
- **Photometric Interpretation**: Predominantly `MONOCHROME2` (high values = bright), with potential `MONOCHROME1` requiring inversion.
- **Slice Ordering**: Slices must be sorted along the geometric slice normal derived from `ImageOrientationPatient` and `ImagePositionPatient`, not by alphanumeric filename.

---

## 4. Compute & Kaggle Environment Requirements
- **Submission Type**: Notebook / Code competition.
- **Inference Runtime**: Standard 9-hour limit for GPU notebooks.
- **Internet Access**: Disabled during hidden test evaluation.
- **Special Tracks**: Efficiency Prize track evaluating predictive accuracy normalized by wall-clock inference runtime.
