"""
Multimodal Tri-Plane HMIL Trainer for RSNA Knee Abnormality Detection.
Handles multi-epoch training, mixed precision, Asymmetric Loss, validation Macro ROC-AUC, and checkpointing.
"""

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from rsna_knee.constants import TARGET_NAMES
from rsna_knee.models.asymmetric_loss import AsymmetricLoss
from rsna_knee.models.multimodal_hmil import MultimodalHMILModel
from rsna_knee.paths import get_output_dir
from rsna_knee.training.metrics import compute_macro_auc


def compute_model_weight_checksum(model: nn.Module) -> str:
    """Computes MD5 checksum of model parameter tensors to verify learning."""
    hasher = hashlib.md5()
    for p in model.parameters():
        hasher.update(p.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()[:10]


class MultimodalHMILTrainer:
    """
    Trainer for Multimodal Tri-Plane HMIL model.
    """

    def __init__(
        self,
        model: MultimodalHMILModel,
        device: torch.device,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.5,
        checkpoint_dir: Optional[Path] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = AsymmetricLoss(gamma_neg=gamma_neg, gamma_pos=gamma_pos)
        self.checkpoint_dir = checkpoint_dir if checkpoint_dir is not None else get_output_dir() / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.use_cuda = (device.type == "cuda")

    def train_epoch(self, dataloader: DataLoader, scheduler=None) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(dataloader, desc="Training", leave=False):
            sagittal = batch["sagittal"].to(self.device)
            coronal = batch["coronal"].to(self.device)
            axial = batch["axial"].to(self.device)
            meta = batch["metadata"].to(self.device)
            targets = batch["targets"].to(self.device)
            loss_mask = batch["loss_mask"].to(self.device)
            loss_weight = batch["loss_weight"].to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(sagittal, coronal, axial, metadata=meta)
            loss = self.criterion(logits, targets, loss_mask=loss_mask, weights=loss_weight)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(1, n_batches)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, Dict[str, float], float, np.ndarray, np.ndarray]:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        all_preds = []
        all_targets = []
        all_masks = []

        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            sagittal = batch["sagittal"].to(self.device)
            coronal = batch["coronal"].to(self.device)
            axial = batch["axial"].to(self.device)
            meta = batch["metadata"].to(self.device)
            targets = batch["targets"].to(self.device)
            loss_mask = batch["loss_mask"].to(self.device)

            logits = self.model(sagittal, coronal, axial, metadata=meta)
            loss = self.criterion(logits, targets, loss_mask=loss_mask)

            total_loss += loss.item()
            n_batches += 1

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_preds.append(probs)
            all_targets.append(targets.detach().cpu().numpy())
            all_masks.append(loss_mask.detach().cpu().numpy())

        y_pred = np.vstack(all_preds)
        y_true = np.vstack(all_targets)
        mask = np.vstack(all_masks)

        val_loss = total_loss / max(1, n_batches)
        macro_auc, per_target_auc = compute_macro_auc(y_true, y_pred, mask=mask)

        return val_loss, per_target_auc, macro_auc, y_pred, y_true

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        fold: int = 0,
    ) -> Dict[str, any]:
        init_checksum = compute_model_weight_checksum(self.model)
        print(f"[*] Initial Model Weight Checksum: {init_checksum}")

        total_steps = epochs * len(train_loader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(1, total_steps), eta_min=1e-6)

        best_auc = 0.0
        best_metrics: Dict[str, float] = {}
        best_preds = None

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader, scheduler=scheduler)
            val_loss, per_target_auc, macro_auc, y_pred, y_true = self.evaluate(val_loader)

            pred_std = float(np.mean(np.std(y_pred, axis=0)))
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro AUC: {macro_auc:.4f} | Pred Std: {pred_std:.4f}")

            if not np.isnan(macro_auc) and (macro_auc > best_auc or epoch == 1):
                best_auc = macro_auc
                best_metrics = per_target_auc
                best_preds = y_pred
                save_path = self.checkpoint_dir / f"model_fold_{fold}_best.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "macro_auc": macro_auc,
                    "per_target_auc": per_target_auc,
                    "fold": fold,
                    "initial_checksum": init_checksum,
                    "final_checksum": compute_model_weight_checksum(self.model),
                }, save_path)
                print(f"  [+] Saved new best checkpoint to {save_path} (Macro AUC: {macro_auc:.4f})")

        final_checksum = compute_model_weight_checksum(self.model)
        print(f"[+] Training Completed | Final Checksum: {final_checksum} (Changed: {init_checksum != final_checksum})")

        return {
            "best_macro_auc": best_auc,
            "per_target_auc": best_metrics,
            "best_predictions": best_preds,
            "init_checksum": init_checksum,
            "final_checksum": final_checksum,
        }
