"""
Loss functions for multi-label knee abnormality training with masked and confidence-weighted supervision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedBCEWithLogitsLoss(nn.Module):
    """
    Binary Cross Entropy with logits supporting sample and target-specific loss masks and confidence weights.
    Loss is calculated only for valid targets (loss_mask == True) and weighted by extraction confidence.
    """

    def __init__(self, reduction: str = "mean", pos_weight: torch.Tensor = None):
        super().__init__()
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss_mask: torch.Tensor = None,
        weights: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, num_targets)
            targets: (B, num_targets) in [0, 1]
            loss_mask: (B, num_targets) boolean mask where True = calculate loss
            weights: (B, num_targets) float confidence weights
        """
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )

        if weights is not None:
            loss = loss * weights

        if loss_mask is not None:
            loss = loss * loss_mask.float()
            num_valid = torch.clamp(loss_mask.float().sum(), min=1.0)
            if self.reduction == "mean":
                return loss.sum() / num_valid
            elif self.reduction == "sum":
                return loss.sum()
            else:
                return loss
        else:
            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            else:
                return loss
