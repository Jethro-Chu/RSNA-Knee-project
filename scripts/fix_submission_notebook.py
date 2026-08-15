import json
from pathlib import Path

# Load raw cell lines directly from a list
cell_4_lines = [
    "# 4. Full 5-Fold Test Inference with Deduplicated Ensemble & Rank-Percentile Calibration\n",
    "if TEST_CSV.exists():\n",
    "    test_df = pd.read_csv(TEST_CSV)\n",
    "elif SAMPLE_SUB.exists():\n",
    "    test_df = pd.read_csv(SAMPLE_SUB)[[ID_COLUMN]]\n",
    "else:\n",
    "    test_df = pd.DataFrame({ID_COLUMN: ['sample_001', 'sample_002', 'sample_003']})\n",
    "\n",
    "print(f'[*] Test Set Studies to Predict: {len(test_df)}')\n",
    "ensemble_models = load_trained_ensemble(device)\n",
    "if len(ensemble_models) != 5:\n",
    "    print(f'[!] Warning: Expected 5 ensemble models, but loaded {len(ensemble_models)}')\n",
    "\n",
    "raw_study_predictions = []\n",
    "study_ids = []\n",
    "\n",
    "for _, r in tqdm(test_df.iterrows(), total=len(test_df), desc='5-Fold Ensemble Inference'):\n",
    "    study_id = str(r[ID_COLUMN])\n",
    "    study_ids.append(study_id)\n",
    "    study_dir = TEST_SERIES_DIR / study_id\n",
    "    dicom_files = list(study_dir.glob('**/*.dcm')) if study_dir.exists() else []\n",
    "    \n",
    "    meta_vec = extract_dicom_metadata_features(dicom_files)\n",
    "    meta_tensor = torch.from_numpy(meta_vec).unsqueeze(0).float().to(device)\n",
    "    \n",
    "    plane_files = sort_study_dicoms_by_plane(dicom_files)\n",
    "    plane_tensors = {}\n",
    "    for p in ['sagittal', 'coronal', 'axial']:\n",
    "        f_list = plane_files[p] if len(plane_files[p]) > 0 else plane_files['unknown']\n",
    "        if len(f_list) > 0:\n",
    "            slices = []\n",
    "            for fp, _ in f_list:\n",
    "                s_arr = load_and_resize_slice(fp, target_size=224)\n",
    "                if s_arr is not None:\n",
    "                    slices.append(s_arr)\n",
    "            if slices:\n",
    "                s2p5 = sample_slices_2p5d_from_list(slices, target_slice_count=12, channels=3, image_size=224)\n",
    "                plane_tensors[p] = torch.from_numpy(s2p5).unsqueeze(0).float().to(device)\n",
    "            else:\n",
    "                plane_tensors[p] = torch.zeros((1, 12, 3, 224, 224), device=device)\n",
    "        else:\n",
    "            plane_tensors[p] = torch.zeros((1, 12, 3, 224, 224), device=device)\n",
    "    \n",
    "    # Ensemble forward pass over all 5 verified fold checkpoints\n",
    "    fold_probs = []\n",
    "    with torch.no_grad():\n",
    "        for model in ensemble_models:\n",
    "            p = torch.sigmoid(model(plane_tensors['sagittal'], plane_tensors['coronal'], plane_tensors['axial'], metadata=meta_tensor)).squeeze(0).cpu().numpy()\n",
    "            fold_probs.append(p)\n",
    "    \n",
    "    # Average across fold models\n",
    "    mean_study_prob = np.mean(fold_probs, axis=0)\n",
    "    raw_study_predictions.append(mean_study_prob)\n",
    "\n",
    "pred_matrix = np.vstack(raw_study_predictions)\n",
    "\n",
    "# Rank-Percentile Calibration (for test sets with N >= 5)\n",
    "if len(pred_matrix) >= 5:\n",
    "    calibrated_matrix = np.zeros_like(pred_matrix)\n",
    "    for k in range(len(TARGET_NAMES)):\n",
    "        ranks = rankdata(pred_matrix[:, k])\n",
    "        calibrated_matrix[:, k] = (ranks - 0.5) / len(ranks)\n",
    "    final_probs = 0.5 * pred_matrix + 0.5 * calibrated_matrix\n",
    "else:\n",
    "    final_probs = pred_matrix\n",
    "\n",
    "# Construct submission dataframe\n",
    "sub_df = pd.DataFrame(final_probs, columns=TARGET_NAMES)\n",
    "sub_df.insert(0, ID_COLUMN, study_ids)\n",
    "sub_df.to_csv(OUTPUT_CSV, index=False)\n",
    "\n",
    "# 5. Mandatory Submission Sanity Gate\n",
    "print('=' * 60)\n",
    "print('             SUBMISSION SANITY & INTEGRITY GATE')\n",
    "print('=' * 60)\n",
    "print(f'Shape: {sub_df.shape}')\n",
    "print('First 3 Predictions:')\n",
    "print(sub_df.head(3))\n",
    "print('Prediction Column Statistics:')\n",
    "print(sub_df[TARGET_NAMES].describe().T[['mean', 'std', 'min', 'max']])\n",
    "print('Unique values per column:')\n",
    "print(sub_df[TARGET_NAMES].nunique())\n",
    "\n",
    "# Assertions\n",
    "assert sub_df[ID_COLUMN].nunique() == len(sub_df), 'Duplicate StudyInstanceUIDs found!'\n",
    "assert not sub_df.isnull().any().any(), 'NaN values detected in submission!'\n",
    "assert (sub_df[TARGET_NAMES].values >= 0.0).all() and (sub_df[TARGET_NAMES].values <= 1.0).all(), 'Probabilities outside [0, 1]!'\n",
    "assert list(sub_df.columns) == [ID_COLUMN] + TARGET_NAMES, 'Incorrect submission columns!'\n",
    "print('[+] ALL SANITY & INTEGRITY GATES PASSED SUCCESSFULLY!')\n",
    "print(f'[+] Output written to: {OUTPUT_CSV.resolve()}')\n"
]

for nb_path in ['kaggle/RSNA_knee/revised/rsna-knee.ipynb', 'notebooks/02_kaggle_submission.ipynb']:
    with open(nb_path) as f:
        nb = json.load(f)
    
    nb['cells'][4]['source'] = cell_4_lines
    
    with open(nb_path, 'w') as f:
        json.dump(nb, f, indent=1)
    
    print(f'Cleanly updated {nb_path}')

# Test compilation of all cells
with open('kaggle/RSNA_knee/revised/rsna-knee.ipynb') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        code = ''.join(c['source'])
        compile(code, f'cell_{i}', 'exec')
        n_lines = len(c["source"])
        print(f"  [+] Cell {i} compiled cleanly ({n_lines} lines).")
