"""
Multimodal Vision + Tabular Metadata Hierarchical Multiple-Instance Learning Network.
Combines Tri-Plane (Sagittal, Coronal, Axial) visual features with clinical metadata priors
(Sex, Field Strength, Manufacturer, Slice Geometry) to optimize Macro ROC-AUC.
"""

from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from rsna_knee.models.pooling import TargetSpecificAttentionPooling


class MultimodalHMILModel(nn.Module):
    """
    Multimodal Vision + Metadata Architecture:
    1. Vision Branch: Tri-plane (Sagittal, Coronal, Axial) 2.5D slices -> Shared/Plane 2D CNN -> Target-Specific Slice Attention -> Gated Tri-Plane Fusion.
    2. Tabular Metadata Branch: 16-dim clinical/scanner feature vector -> MLP projection -> (B, feature_dim).
    3. Multimodal Gated Fusion: Fuses vision representation and clinical prior per target -> 12 classification logits.
    """

    def __init__(
        self,
        num_targets: int = 12,
        in_channels: int = 3,
        meta_dim: int = 16,
        feature_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_targets = num_targets
        self.feature_dim = feature_dim
        self.planes = ["sagittal", "coronal", "axial"]

        # 1. 2D Vision Backbone Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, feature_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(feature_dim, feature_dim),
        )

        # 2. Target-Specific Attention Pooling per plane
        self.plane_pools = nn.ModuleDict({
            p: TargetSpecificAttentionPooling(in_features=feature_dim, num_targets=num_targets, hidden_dim=feature_dim)
            for p in self.planes
        })

        # 3. Cross-Plane View Fusion Gating
        self.view_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim * 3, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 3),
            )
            for _ in range(num_targets)
        ])

        # 4. Tabular Metadata Feature Net
        self.meta_net = nn.Sequential(
            nn.Linear(meta_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, feature_dim),
        )

        # 5. Multimodal Target Classification Heads
        self.dropout = nn.Dropout(p=dropout)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim * 2, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(64, 1),
            )
            for _ in range(num_targets)
        ])

    def forward(
        self,
        sagittal_or_inputs: Union[Dict[str, torch.Tensor], torch.Tensor],
        coronal: Optional[torch.Tensor] = None,
        axial: Optional[torch.Tensor] = None,
        metadata: Optional[torch.Tensor] = None,
        meta_features: Optional[torch.Tensor] = None,
        plane_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Flexible forward supporting both Dict input or individual plane tensors.
        """
        if isinstance(sagittal_or_inputs, dict):
            plane_inputs = sagittal_or_inputs
        else:
            plane_inputs = {
                "sagittal": sagittal_or_inputs,
                "coronal": coronal,
                "axial": axial,
            }

        meta = metadata if metadata is not None else meta_features

        first_tensor = next((v for v in plane_inputs.values() if v is not None), None)
        if first_tensor is None:
            raise ValueError("All plane inputs are None")

        B = first_tensor.shape[0]
        device = first_tensor.device

        # Process Visual Planes
        plane_reps: Dict[str, torch.Tensor] = {}
        for p in self.planes:
            if p in plane_inputs and plane_inputs[p] is not None:
                x = plane_inputs[p]
                B_p, S, C, H, W = x.shape
                feats_flat = self.stem(x.view(B_p * S, C, H, W))
                feats = feats_flat.view(B_p, S, self.feature_dim)

                mask = plane_masks[p] if (plane_masks is not None and p in plane_masks) else None
                plane_reps[p] = self.plane_pools[p](feats, mask=mask)
            else:
                plane_reps[p] = torch.zeros((B, self.num_targets, self.feature_dim), device=device)

        # Stack views: (B, 12, 3, feature_dim)
        stacked_views = torch.stack([plane_reps[p] for p in self.planes], dim=2)

        # Process Metadata
        if meta is None:
            meta = torch.zeros((B, 16), device=device)
        meta_emb = self.meta_net(meta)  # (B, feature_dim)

        # Multimodal Fusion & Prediction per Target
        logits_list = []
        for k in range(self.num_targets):
            # 1. Fuse Views
            k_concat = stacked_views[:, k, :, :].reshape(B, 3 * self.feature_dim)
            view_weights = F.softmax(self.view_gates[k](k_concat), dim=-1).unsqueeze(-1)
            fused_vis = torch.sum(stacked_views[:, k, :, :] * view_weights, dim=1)  # (B, feature_dim)

            # 2. Combine Vision + Tabular Metadata
            multimodal_k = torch.cat([fused_vis, meta_emb], dim=-1)  # (B, 2 * feature_dim)
            multimodal_k = self.dropout(multimodal_k)

            # 3. Predict Logit
            k_logit = self.heads[k](multimodal_k)
            logits_list.append(k_logit)

        return torch.cat(logits_list, dim=-1)
