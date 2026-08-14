"""
Multi-View Multi-Plane Hierarchical Multiple-Instance Learning (HMIL) Network for Knee MRI.
Processes Sagittal, Coronal, and Axial series simultaneously and performs target-specific cross-plane attention fusion.
"""

from typing import Dict, List, Optional
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from rsna_knee.models.pooling import TargetSpecificAttentionPooling


class MultiViewHMILModel(nn.Module):
    """
    Multi-View Hierarchical MIL Architecture:
    1. Input: Dict of plane tensors:
       - 'sagittal': (B, S_sag, 3, H, W)
       - 'coronal':  (B, S_cor, 3, H, W)
       - 'axial':    (B, S_ax,  3, H, W)
    2. Shared or plane-adapted 2D CNN/Transformer feature extraction per slice.
    3. Target-Specific Slice Attention Pooling per plane -> (B, num_targets, D) per plane.
    4. Target-Specific Learned Cross-Plane Gated Fusion -> (B, num_targets, D).
    5. Final Target Heads -> (B, 12) logits.
    """

    def __init__(
        self,
        backbone_name: str = "resnet34d",
        pretrained: bool = True,
        num_targets: int = 12,
        in_channels: int = 3,
        dropout: float = 0.2,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.num_targets = num_targets
        self.planes = ["sagittal", "coronal", "axial"]

        # Shared 2D feature extractor
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,
        )
        self.feature_dim = self.encoder.num_features

        # Target-Specific Attention Pooling per plane
        self.plane_pools = nn.ModuleDict({
            plane: TargetSpecificAttentionPooling(
                in_features=self.feature_dim,
                num_targets=num_targets,
                hidden_dim=hidden_dim,
            )
            for plane in self.planes
        })

        # Target-Specific View Importance Gating Nets:
        # Computes view weights [alpha_sag, alpha_cor, alpha_ax] per target abnormality
        self.view_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.feature_dim * 3, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 3),  # 3 views
            )
            for _ in range(num_targets)
        ])

        self.dropout = nn.Dropout(p=dropout)

        # 12 Independent Target Classification Heads
        self.heads = nn.ModuleList([
            nn.Linear(self.feature_dim, 1) for _ in range(num_targets)
        ])

    def forward(
        self,
        plane_inputs: Dict[str, torch.Tensor],
        plane_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            plane_inputs: Dict mapping plane_name -> (B, S, C, H, W)
            plane_masks: Optional Dict mapping plane_name -> (B, S)
        Returns:
            logits: (B, 12)
        """
        # Determine batch size and device
        first_tensor = next(iter(plane_inputs.values()))
        B = first_tensor.shape[0]
        device = first_tensor.device

        plane_reps: Dict[str, torch.Tensor] = {}

        for plane in self.planes:
            if plane in plane_inputs and plane_inputs[plane] is not None:
                x = plane_inputs[plane]  # (B, S, C, H, W)
                B_p, S, C, H, W = x.shape
                x_flat = x.view(B_p * S, C, H, W)
                feats_flat = self.encoder(x_flat)  # (B*S, feature_dim)
                feats = feats_flat.view(B_p, S, self.feature_dim)  # (B, S, feature_dim)

                mask = plane_masks[plane] if (plane_masks is not None and plane in plane_masks) else None
                pooled_targets = self.plane_pools[plane](feats, mask=mask)  # (B, num_targets, feature_dim)
                plane_reps[plane] = pooled_targets
            else:
                # Missing plane: fill with zeros
                plane_reps[plane] = torch.zeros((B, self.num_targets, self.feature_dim), device=device)

        # Stack plane representations: (B, num_targets, 3, feature_dim)
        stacked_views = torch.stack([plane_reps[p] for p in self.planes], dim=2)

        # Cross-View Target-Specific Fusion & Prediction
        logits_list = []
        for k in range(self.num_targets):
            # Target features across 3 views: (B, 3 * feature_dim)
            k_concat = stacked_views[:, k, :, :].reshape(B, 3 * self.feature_dim)
            
            # View importance weights: (B, 3)
            view_logits = self.view_gates[k](k_concat)
            view_weights = F.softmax(view_logits, dim=-1).unsqueeze(-1)  # (B, 3, 1)

            # Target fused representation: (B, feature_dim)
            k_views = stacked_views[:, k, :, :]  # (B, 3, feature_dim)
            fused_rep = torch.sum(k_views * view_weights, dim=1)  # (B, feature_dim)
            fused_rep = self.dropout(fused_rep)

            # Output logit: (B, 1)
            k_logit = self.heads[k](fused_rep)
            logits_list.append(k_logit)

        return torch.cat(logits_list, dim=-1)
