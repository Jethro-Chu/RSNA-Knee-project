#!/usr/bin/env python3
"""
Model training CLI script for RSNA Knee 2.5D models.
"""

import argparse
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml

from rsna_knee.data.dataset import KneeStudyDataset
from rsna_knee.models.model_2p5d import Knee2p5dModel
from rsna_knee.paths import get_base_dir, get_data_dir, get_output_dir
from rsna_knee.training.trainer import KneeTrainer


def main():
    parser = argparse.ArgumentParser(description="Train RSNA Knee Abnormality 2.5D Model")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to config yaml")
    parser.add_argument("--fold", type=int, default=0, help="Fold to train on")
    args = parser.parse_args()

    config_path = get_base_dir() / args.config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[*] Training on device: {device}")

    # Load dataset with folds
    folds_path = get_data_dir() / "metadata" / "folds.csv"
    if not folds_path.exists():
        folds_path = get_data_dir() / "pseudo_labels" / "pseudo_labels_v1.parquet"

    if folds_path.exists():
        if str(folds_path).endswith(".parquet"):
            df = pd.read_parquet(folds_path)
        else:
            df = pd.read_csv(folds_path)
    else:
        print("[!] No folds file found. Please run scripts/generate_pseudo_labels.py and scripts/make_folds.py first.")
        return

    train_df = df[df["fold"] != args.fold].reset_index(drop=True)
    val_df = df[df["fold"] == args.fold].reset_index(drop=True)

    print(f"[*] Training studies: {len(train_df)} | Validation studies: {len(val_df)}")

    train_ds = KneeStudyDataset(
        train_df,
        image_size=cfg["data"]["image_size"],
        slices_per_series=cfg["data"]["slices_per_series"],
        channels=cfg["data"]["channels_per_slice"],
        is_training=True,
    )
    val_ds = KneeStudyDataset(
        val_df,
        image_size=cfg["data"]["image_size"],
        slices_per_series=cfg["data"]["slices_per_series"],
        channels=cfg["data"]["channels_per_slice"],
        is_training=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=0,  # Safe local execution
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    model = Knee2p5dModel(
        backbone_name=cfg["model"]["backbone"],
        pretrained=cfg["model"]["pretrained"],
        num_targets=cfg["model"]["num_targets"],
        in_channels=cfg["data"]["channels_per_slice"],
        dropout=cfg["model"]["dropout"],
    )

    trainer = KneeTrainer(
        model=model,
        device=device,
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    print("[*] Beginning training loop...")
    metrics = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg["training"]["epochs"],
        fold=args.fold,
    )
    print(f"[+] Finished training fold {args.fold}!")
    print(f"    Validation Metrics: {metrics}")


if __name__ == "__main__":
    main()
