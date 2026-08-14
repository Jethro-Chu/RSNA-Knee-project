"""
Multimodal Vision + Tabular Metadata Hierarchical Multiple-Instance Learning Network.
Combines Tri-Plane (Sagittal, Coronal, Axial) visual features with clinical metadata priors
(Sex, Field Strength, Manufacturer, Slice Geometry) to optimize Macro ROC-AUC.
"""

from typing import Dict, List, Optional
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from rsna_knee.models.pooling import TargetSpecificAttentionPooling


class MultimodalHMILModel(nn.Module):
    """
    Multimodal Vision + Metadata Architecture:
    1. Vision Branch: Tri-plane (Sagittal, Coronal, Axial) 2.5D slices -> Shared/Plane 2D CNN -> Target-Specific Slice Attention -> Gated Tri-Plane Fusion.
    2. Tabular Metadata Branch: 16-dim clinical/scanner feature vector -> MLP projection -> (B, num_targets, meta_dim).
    3. Multimodal Gated Fusion: Fuses vision representation and clinical prior per target -> 12 classification logits.
    """

    def __init__(
        self,
        backbone_name: str = "resnet34d",
        pretrained: bool = True,
        num_targets: int = 12,
        in_channels: int = 3,
        meta_dim: int = 16,
        feature_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_targets = num_targets
        self.feature_dim = feature_dim
        self.planes = ["sagittal", "coronal", "axial"]

        # 1. 2D Vision Backbone
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,
        )
        self.enc_dim = self.encoder.num_features
        self.proj_vision = nn.Linear(self.enc_dim, feature_dim)

        # 2. Target-Specific Attention Pooling per plane
        self.plane_pools = nn.ModuleDict({
            p: TargetSpecificAttentionPooling(in_features=feature_dim, num_targets=num_targets)
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
            nn.Linear(meta_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, feature_dim),
        )

        # 5. Multimodal Target Classification Heads
        self.dropout = nn.Dropout(p=dropout)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim * 2, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(128, 1),
            )
            for _ in range(num_targets)
        ])

    def forward(
        self,
        plane_inputs: Dict[str, torch.Tensor],
        meta_features: Optional[torch.Tensor] = None,
        plane_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            plane_inputs: Dict mapping plane_name -> (B, S, C, H, W)
            meta_features: Optional tabular metadata tensor (B, 16)
        Returns:
            logits: (B, 12)
        """
        first_tensor = next(iter(plane_inputs.values()))
        B = first_tensor.shape[0]
        device = first_tensor.device

        # Process Visual Planes
        plane_reps: Dict[str, torch.Tensor] = {}
        for p in self.planes:
            if p in plane_inputs and plane_inputs[p] is not None:
                x = plane_inputs[p]
                B_p, S, C, H, W = x.shape
                x_flat = x.view(B_p * S, C, H, W)
                feats_flat = self.proj_vision(self.encoder(x_flat))
                feats = feats_flat.view(B_p, S, self.feature_dim)

                mask = plane_masks[p] if (plane_masks is not None and p in plane_masks) else None
                plane_reps[p] = self.plane_pools[p](feats, mask=mask)
            else:
                plane_reps[p] = torch.zeros((B, self.num_targets, self.feature_dim), device=device)

        # Stack views: (B, 12, 3, feature_dim)
        stacked_views = torch.stack([plane_reps[p] for p in self.planes], dim=2)

        # Process Metadata
        if meta_features is None:
            meta_features = torch.zeros((B, 16), device=device)
        meta_emb = self.meta_net(meta_features)  # (B, feature_dim)

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
