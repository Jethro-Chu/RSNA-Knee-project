# Team Notebook Audit & Deployment Log

- **Competition**: RSNA Knee Abnormality Detection
- **Notebook Title**: `RSNA_knee`
- **Owner / Team**: `wenwen12` (Team shared notebook)
- **Slug**: `wenwen12/rsna-knee`
- **Kaggle URL**: https://www.kaggle.com/code/wenwen12/rsna-knee
- **Deployed Version**: **Version 7**
- **Status**: **`COMPLETE`** (Successfully loaded trained weights & generated submission)
- **Attached Dataset**: `chujethro/rsna-knee-checkpoints` (`model_fold_4_best.pt`)
- **Accelerator**: GPU Enabled (`enable_gpu: true`)
- **Internet**: Disabled (`enable_internet: false`, competition rule-compliant)

---

## Technical Verification Log (Version 7)

```
[*] Device: cuda | Input Directory: /kaggle/input/competitions/rsna-knee-abnormality-detection
[*] Loading trained model checkpoint from: /kaggle/input/rsna-knee-checkpoints/model_fold_4_best.pt
[+] Model weights loaded successfully!
Predicting Multimodal Studies: 100%|██████████| 3/3 [00:07<00:00, 2.62s/it]
[+] Submission generated successfully: submission.csv (3 rows)
```
