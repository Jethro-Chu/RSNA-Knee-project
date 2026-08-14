"""
Asymmetric Loss (ASL) for multi-label medical imaging with extreme class imbalance.
Dynamically down-weights easy negatives to focus gradients on hard positive and negative clinical abnormalities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for Multi-Label Classification (Ben-Baruch et al.).
    
    L = - y * (1 - p_s)^gamma_pos * log(p_s) - (1 - y) * (p_m)^gamma_neg * log(1 - p_m)
    where p_m = max(p - margin, 0)
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.5,
        clip: float = 0.05,
        eps: float = 1e-8,
        disable_torch_grad_focal_loss: bool = True,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

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
            loss_mask: (B, num_targets) boolean mask of valid supervision
            weights: (B, num_targets) confidence weights
        """
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(logits)
        xs_pos = x_sigmoid
        xs_neg = 1.0 - x_sigmoid

        # Asymmetric Clipping for Negatives
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # Basic CE calculation
        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * targets
            pt1 = xs_neg * (1.0 - targets)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            one_sided_w = torch.pow(1.0 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        loss = -loss

        # Apply confidence weights and supervision masks
        if weights is not None:
            loss = loss * weights

        if loss_mask is not None:
            loss = loss * loss_mask.float()
            num_valid = torch.clamp(loss_mask.float().sum(), min=1.0)
            return loss.sum() / num_valid

        return loss.mean()
