#!/usr/bin/env python3
"""
CLI script to train Multimodal Tri-Plane HMIL models across 5 folds
with Asymmetric Loss, confidence weighting, and leakage-free validation.
"""

import argparse
import datetime
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from rsna_knee.constants import ID_COLUMN, TARGET_NAMES
from rsna_knee.data.tri_plane_dataset import TriPlaneMultimodalDataset
from rsna_knee.models.multimodal_hmil import MultimodalHMILModel
from rsna_knee.paths import get_data_dir, get_output_dir
from rsna_knee.training.multimodal_trainer import MultimodalHMILTrainer


def main():
    parser = argparse.ArgumentParser(description="Train Multimodal Tri-Plane HMIL Model")
    parser.add_argument("--fold", type=int, default=0, help="Fold index (0-4) or -1 for all folds")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--gamma-neg", type=float, default=4.0, help="ASL gamma negative")
    parser.add_argument("--gamma-pos", type=float, default=0.5, help="ASL gamma positive")
    parser.add_argument("--image-size", type=int, default=224, help="Image resolution")
    parser.add_argument("--slices", type=int, default=12, help="Slices per plane")
    parser.add_argument("--backbone", type=str, default="resnet18", help="Timm backbone name")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained backbone weights")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast smoke test on subset")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[*] Training on device: {device}")

    # Load pseudo-labels and folds
    pseudo_path = get_data_dir() / "pseudo_labels" / "pseudo_labels_v2.parquet"
    if not pseudo_path.exists():
        pseudo_path = get_data_dir() / "pseudo_labels" / "pseudo_labels_v1.parquet"
    
    folds_path = get_data_dir() / "metadata" / "folds.csv"

    print(f"[*] Loading pseudo-labels: {pseudo_path}")
    pseudo_df = pd.read_parquet(pseudo_path)
    print(f"[*] Loading folds: {folds_path}")
    folds_df = pd.read_csv(folds_path)

    merged_df = pd.merge(pseudo_df, folds_df[[ID_COLUMN, "fold"]], on=ID_COLUMN)

    if args.smoke_test:
        print("[!] Running smoke test mode on 32 samples...")
        merged_df = merged_df.iloc[:32].copy()
        merged_df["fold"] = [0]*16 + [1]*16
        args.epochs = 2

    folds_to_run = [args.fold] if args.fold >= 0 else list(range(5))

    checkpoint_dir = get_output_dir() / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_csv = Path("experiments/results.csv")

    for fold_idx in folds_to_run:
        print(f"\n{'='*60}")
        print(f"            STARTING TRAINING FOLD {fold_idx} / 5")
        print(f"{'='*60}")

        train_subset = merged_df[merged_df["fold"] != fold_idx].copy()
        val_subset = merged_df[merged_df["fold"] == fold_idx].copy()

        print(f"[*] Train studies: {len(train_subset)} | Val studies: {len(val_subset)}")

        train_ds = TriPlaneMultimodalDataset(
            train_subset,
            image_size=args.image_size,
            slices_per_plane=args.slices,
            is_training=True,
        )
        val_ds = TriPlaneMultimodalDataset(
            val_subset,
            image_size=args.image_size,
            slices_per_plane=args.slices,
            is_training=False,
        )

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        model = MultimodalHMILModel(
            backbone_name=args.backbone,
            pretrained=args.pretrained,
            num_targets=len(TARGET_NAMES),
            meta_dim=16,
            feature_dim=128,
        )

        trainer = MultimodalHMILTrainer(
            model=model,
            device=device,
            lr=args.lr,
            weight_decay=args.weight_decay,
            gamma_neg=args.gamma_neg,
            gamma_pos=args.gamma_pos,
            checkpoint_dir=checkpoint_dir,
        )

        start_time = time.time()
        fit_results = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            fold=fold_idx,
        )
        runtime = time.time() - start_time

        macro_auc = fit_results["best_macro_auc"]
        per_target = fit_results["per_target_auc"]

        print(f"\n[+] Fold {fold_idx} Complete | Best Val Macro ROC-AUC: {macro_auc:.4f} | Time: {runtime:.1f}s")

        # Save to experiments/results.csv
        exp_id = f"exp_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_f{fold_idx}"
        row_dict = {
            "experiment_id": exp_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "architecture": "Multimodal_TriPlane_HMIL",
            "backbone": args.backbone,
            "image_size": args.image_size,
            "slices_per_plane": args.slices,
            "loss": "AsymmetricLoss",
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "fold": fold_idx,
            "macro_auc": round(macro_auc, 4) if not np.isnan(macro_auc) else 0.0,
        }
        for t in TARGET_NAMES:
            sanitized_key = t.replace(" ", "_").replace("'", "") + "_auc"
            row_dict[sanitized_key] = round(per_target.get(t, 0.0), 4) if not np.isnan(per_target.get(t, 0.0)) else 0.0

        row_dict["runtime_sec"] = round(runtime, 1)
        row_dict["notes"] = f"Checksum change: {fit_results['init_checksum']} -> {fit_results['final_checksum']}"

        res_df = pd.DataFrame([row_dict])
        if results_csv.exists():
            res_df.to_csv(results_csv, mode="a", header=False, index=False)
        else:
            res_df.to_csv(results_csv, index=False)


if __name__ == "__main__":
    main()
