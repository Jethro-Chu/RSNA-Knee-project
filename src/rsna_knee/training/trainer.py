"""
Training engine and cross-validation execution for RSNA Knee 2.5D models.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from rsna_knee.constants import TARGET_NAMES
from rsna_knee.data.dataset import KneeStudyDataset
from rsna_knee.models.losses import MaskedBCEWithLogitsLoss
from rsna_knee.models.model_2p5d import Knee2p5dModel
from rsna_knee.paths import get_output_dir
from rsna_knee.training.metrics import compute_macro_auc


class KneeTrainer:
    """
    Handles model training, validation, metric computation, and model checkpointing.
    """

    def __init__(
        self,
        model: Knee2p5dModel,
        device: torch.device,
        lr: float = 3.0e-4,
        weight_decay: float = 1.0e-4,
        checkpoint_dir: Optional[Path] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = MaskedBCEWithLogitsLoss()
        self.checkpoint_dir = checkpoint_dir if checkpoint_dir is not None else get_output_dir() / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(dataloader, desc="Training", leave=False):
            images = batch["images"].to(self.device)
            targets = batch["targets"].to(self.device)
            loss_mask = batch["loss_mask"].to(self.device)
            weights = batch["weights"].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, targets, loss_mask=loss_mask, weights=weights)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(1, n_batches)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, Dict[str, float], float]:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        all_preds = []
        all_targets = []
        all_masks = []

        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            images = batch["images"].to(self.device)
            targets = batch["targets"].to(self.device)
            loss_mask = batch["loss_mask"].to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, targets, loss_mask=loss_mask)
            total_loss += loss.item()
            n_batches += 1

            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(targets.cpu().numpy())
            all_masks.append(loss_mask.cpu().numpy())

        y_pred = np.vstack(all_preds)
        y_true = np.vstack(all_targets)
        mask = np.vstack(all_masks)

        val_loss = total_loss / max(1, n_batches)
        macro_auc, per_target_auc = compute_macro_auc(y_true, y_pred, mask=mask)

        return val_loss, per_target_auc, macro_auc

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        fold: int = 0,
    ) -> Dict[str, float]:
        best_auc = 0.0
        best_metrics: Dict[str, float] = {}

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, per_target_auc, macro_auc = self.evaluate(val_loader)

            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro AUC: {macro_auc:.4f}")

            if not np.isnan(macro_auc) and macro_auc > best_auc:
                best_auc = macro_auc
                best_metrics = per_target_auc
                save_path = self.checkpoint_dir / f"model_fold_{fold}_best.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "macro_auc": macro_auc,
                    "per_target_auc": per_target_auc,
                }, save_path)
                print(f"  [+] Saved new best checkpoint to {save_path} (Macro AUC: {macro_auc:.4f})")

        return best_metrics
