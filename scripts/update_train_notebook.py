import json
from pathlib import Path

nb_path = Path("notebooks/01_kaggle_train.ipynb")
with open(nb_path, "r") as f:
    nb = json.load(f)

# Cell 5 (the training loop cell)
train_cell_source = """# 5. High-Velocity Multi-GPU Training Loop with AMP
train_df = pd.read_csv(TRAIN_CSV)
print(f"[*] Loaded training set: {len(train_df)} studies")

# Extract report pseudo labels
records = []
for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Extracting NLP Supervision"):
    rec = {ID_COLUMN: row[ID_COLUMN]}
    has_expert = False
    for t in TARGET_NAMES:
        if t in row and not pd.isna(row[t]):
            rec[f"{t}_prob"] = float(row[t])
            rec[f"{t}_loss_mask"] = True
            rec[f"{t}_loss_weight"] = 5.0
            has_expert = True
    if not has_expert:
        extracted = extract_study_report(row.get("Report", ""))
        for t in TARGET_NAMES:
            prob, mask, weight = extracted[t]
            rec[f"{t}_prob"] = prob
            rec[f"{t}_loss_mask"] = mask
            rec[f"{t}_loss_weight"] = weight
    records.append(rec)

processed_df = pd.DataFrame(records)

# Stratified 5-Fold Split
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
strat_col = (processed_df["ACL_prob"] > 0.5).astype(int) + (processed_df["Effusion_prob"] > 0.5).astype(int)*2
processed_df["fold"] = -1
for f, (tr_idx, val_idx) in enumerate(skf.split(processed_df, strat_col)):
    processed_df.loc[val_idx, "fold"] = f

EPOCHS = 3
BATCH_SIZE = 16 if num_gpus >= 2 else 8
NUM_WORKERS = 2
FOLDS_TO_TRAIN = [4]  # Folds to actively train (Folds 0, 1, 2, 3 pre-trained)

def find_checkpoint(fold_idx):
    search_dirs = [
        Path("/kaggle/input/rsna-knee-checkpoints"),
        Path("/kaggle/input/rsna-knee-checkpoints/checkpoints"),
        CHECKPOINT_DIR,
        Path("/kaggle/working"),
        Path("/kaggle/working/checkpoints"),
        Path("checkpoints"),
    ]
    for d in search_dirs:
        p = d / f"model_fold_{fold_idx}_best.pt"
        if p.exists() and p.stat().st_size > 1000000:
            return p
    return None

oof_predictions = np.zeros((len(processed_df), len(TARGET_NAMES)), dtype=np.float32)
oof_targets = np.zeros((len(processed_df), len(TARGET_NAMES)), dtype=np.float32)
oof_masks = np.zeros((len(processed_df), len(TARGET_NAMES)), dtype=bool)

for fold in range(5):
    print(f"\\n{'='*60}\\n         PROCESSING 5-FOLD PIPELINE: FOLD {fold}/5\\n{'='*60}")
    val_sub = processed_df[processed_df["fold"] == fold].copy()
    val_indices = val_sub.index.values
    val_ds = TriPlaneDataset(val_sub, KAGGLE_INPUT, slices_per_plane=8, is_training=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    raw_model = MultimodalHMILModel(num_targets=12)
    ckpt_found = find_checkpoint(fold)

    if fold not in FOLDS_TO_TRAIN and ckpt_found is not None:
        print(f"[*] Found existing pre-trained checkpoint for Fold {fold}: {ckpt_found}")
        ckpt = torch.load(ckpt_found, map_location=device)
        raw_model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model = raw_model.to(device)
        model.eval()

        # Also ensure copy exists in working dir
        if Path("/kaggle/working").exists():
            torch.save({"model_state_dict": raw_model.state_dict(), "fold": fold, "val_macro_auc": ckpt.get("val_macro_auc", 0.0)}, Path(f"/kaggle/working/model_fold_{fold}_best.pt"))
            if (Path("/kaggle/working") / "checkpoints").exists():
                torch.save({"model_state_dict": raw_model.state_dict(), "fold": fold, "val_macro_auc": ckpt.get("val_macro_auc", 0.0)}, Path(f"/kaggle/working/checkpoints/model_fold_{fold}_best.pt"))

        # Evaluate Val Split
        p_list, t_list, m_list = [], [], []
        with torch.no_grad():
            for b in tqdm(val_loader, desc=f"Fold {fold} [OOF Eval]", leave=False):
                sag, cor, ax = b["sagittal"].to(device, non_blocking=True), b["coronal"].to(device, non_blocking=True), b["axial"].to(device, non_blocking=True)
                meta, tgt = b["metadata"].to(device, non_blocking=True), b["targets"].to(device, non_blocking=True)
                mask = b["loss_mask"].to(device, non_blocking=True)
                with autocast(enabled=(device.type == "cuda")):
                    logits = model(sag, cor, ax, metadata=meta)
                p_list.append(torch.sigmoid(logits).cpu().numpy())
                t_list.append(tgt.cpu().numpy())
                m_list.append(mask.cpu().numpy())
        y_p = np.vstack(p_list)
        y_t = np.vstack(t_list)
        y_m = np.vstack(m_list)
        best_val_preds = y_p
    else:
        print(f"[*] Training Fold {fold} for {EPOCHS} epochs...")
        tr_sub = processed_df[processed_df["fold"] != fold].copy()
        tr_ds = TriPlaneDataset(tr_sub, KAGGLE_INPUT, slices_per_plane=8, is_training=True)
        tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

        if num_gpus > 1:
            model = nn.DataParallel(raw_model).to(device)
        else:
            model = raw_model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * max(1, len(tr_loader)), eta_min=1e-6)
        scaler = GradScaler(enabled=(device.type == "cuda"))
        criterion = AsymmetricLoss()

        best_val_auc = 0.0
        best_val_preds = None

        for epoch in range(1, EPOCHS + 1):
            # Train
            model.train()
            tr_loss = 0.0
            t0 = time.time()
            for b in tqdm(tr_loader, desc=f"Fold {fold} Ep {epoch} [Train]", leave=False):
                sag, cor, ax = b["sagittal"].to(device, non_blocking=True), b["coronal"].to(device, non_blocking=True), b["axial"].to(device, non_blocking=True)
                meta, tgt = b["metadata"].to(device, non_blocking=True), b["targets"].to(device, non_blocking=True)
                mask, weight = b["loss_mask"].to(device, non_blocking=True), b["loss_weight"].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=(device.type == "cuda")):
                    logits = model(sag, cor, ax, metadata=meta)
                    loss = criterion(logits, tgt, loss_mask=mask, weights=weight)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                tr_loss += loss.item()
            tr_loss /= max(1, len(tr_loader))

            # Eval
            model.eval()
            val_loss = 0.0
            p_list, t_list, m_list = [], [], []
            with torch.no_grad():
                for b in tqdm(val_loader, desc=f"Fold {fold} Ep {epoch} [Val]", leave=False):
                    sag, cor, ax = b["sagittal"].to(device, non_blocking=True), b["coronal"].to(device, non_blocking=True), b["axial"].to(device, non_blocking=True)
                    meta, tgt = b["metadata"].to(device, non_blocking=True), b["targets"].to(device, non_blocking=True)
                    mask = b["loss_mask"].to(device, non_blocking=True)
                    with autocast(enabled=(device.type == "cuda")):
                        logits = model(sag, cor, ax, metadata=meta)
                        loss = criterion(logits, tgt, loss_mask=mask)
                    val_loss += loss.item()
                    p_list.append(torch.sigmoid(logits).cpu().numpy())
                    t_list.append(tgt.cpu().numpy())
                    m_list.append(mask.cpu().numpy())
            val_loss /= max(1, len(val_loader))

            y_p = np.vstack(p_list)
            y_t = np.vstack(t_list)
            y_m = np.vstack(m_list)

            aucs = []
            for k in range(len(TARGET_NAMES)):
                val_k = y_m[:, k]
                bin_t = (y_t[val_k, k] > 0.5).astype(int)
                if np.unique(bin_t).size > 1:
                    aucs.append(roc_auc_score(bin_t, y_p[val_k, k]))
            val_macro_auc = float(np.mean(aucs)) if aucs else 0.0
            elapsed = time.time() - t0
            print(f"Epoch {epoch:02d}/{EPOCHS:02d} | Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro AUC: {val_macro_auc:.4f} | Elapsed: {elapsed:.1f}s")

            if val_macro_auc > best_val_auc or epoch == 1:
                best_val_auc = val_macro_auc
                best_val_preds = y_p
                state_dict = raw_model.state_dict() if num_gpus <= 1 else model.module.state_dict()
                ckpt_path = CHECKPOINT_DIR / f"model_fold_{fold}_best.pt"
                torch.save({"model_state_dict": state_dict, "fold": fold, "val_macro_auc": val_macro_auc}, ckpt_path)
                if Path("/kaggle/working").exists():
                    torch.save({"model_state_dict": state_dict, "fold": fold, "val_macro_auc": val_macro_auc}, Path(f"/kaggle/working/model_fold_{fold}_best.pt"))
                    if (Path("/kaggle/working") / "checkpoints").exists():
                        torch.save({"model_state_dict": state_dict, "fold": fold, "val_macro_auc": val_macro_auc}, Path(f"/kaggle/working/checkpoints/model_fold_{fold}_best.pt"))
                print(f"  [+] Saved new best fold {fold} checkpoint -> {ckpt_path}")

    oof_predictions[val_indices] = best_val_preds
    oof_targets[val_indices] = y_t
    oof_masks[val_indices] = y_m

# 6. Full 5-Fold OOF Evaluation Report & Parquet Export
print(f"\\n{'='*60}\\n       FINAL 5-FOLD OUT-OF-FOLD (OOF) PERFORMANCE REPORT\\n{'='*60}")
oof_aucs = {}
for k, t_name in enumerate(TARGET_NAMES):
    m_k = oof_masks[:, k]
    bin_t = (oof_targets[m_k, k] > 0.5).astype(int)
    if np.unique(bin_t).size > 1:
        t_auc = float(roc_auc_score(bin_t, oof_predictions[m_k, k]))
        oof_aucs[t_name] = t_auc
        print(f"  {t_name:<30}: {t_auc:.4f} (Evaluated on {m_k.sum()} labeled validation studies)")
    else:
        print(f"  {t_name:<30}: [Single Class / Undefined]")

master_oof_macro = float(np.mean(list(oof_aucs.values())))
print("-" * 60)
print(f" OVERALL 5-FOLD OOF MACRO ROC-AUC: {master_oof_macro:.4f}")
print(f"{'='*60}")

# Save OOF Parquet
oof_df = pd.DataFrame(oof_predictions, columns=[f"{t}_pred" for t in TARGET_NAMES])
oof_df.insert(0, ID_COLUMN, processed_df[ID_COLUMN])
oof_df["fold"] = processed_df["fold"]
for k, t in enumerate(TARGET_NAMES):
    oof_df[f"{t}_target"] = oof_targets[:, k]
    oof_df[f"{t}_loss_mask"] = oof_masks[:, k]
oof_path = Path("/kaggle/working/oof_predictions_v1.parquet") if Path("/kaggle/working").exists() else Path("experiments/oof_predictions_v1.parquet")
oof_df.to_parquet(oof_path, index=False)
print(f"[+] Exported OOF predictions -> {oof_path}")
"""

nb["cells"][2]["source"] = [line + "\n" for line in train_cell_source.splitlines()]

# Save notebooks
with open("notebooks/01_kaggle_train.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

with open("kaggle/RSNA_knee/train/train.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Updated notebooks/01_kaggle_train.ipynb and kaggle/RSNA_knee/train/train.ipynb successfully.")
