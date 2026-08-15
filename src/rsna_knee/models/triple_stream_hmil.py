"""
Triple-Stream Multimodal HMIL Architecture for RSNA Knee Abnormality Detection.

Key Innovations:
1. Dedicated Plane-Specific Encoders: Separate CNN stems for Sagittal, Coronal, and Axial planes
   to avoid spatial/contrast feature interference across orthogonal anatomical orientations.
2. Anatomical Prior Gating Initialization: View gating logits are initialized with pathological
   domain knowledge (e.g. Coronal bias for MCL & Collaterals, Sagittal bias for ACL/Cruciates,
   Axial bias for Patellofemoral tracking & Effusions).
3. Multimodal Tabular Metadata Fusion: Seamless combination of 16-dim scanner geometry priors.
"""

from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from rsna_knee.models.pooling import TargetSpecificAttentionPooling


def create_plane_stem(in_channels: int = 3, feature_dim: int = 128) -> nn.Sequential:
    """Creates a dedicated 2D feature stem for an individual anatomical plane."""
    return nn.Sequential(
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


class TripleStreamHMILModel(nn.Module):
    """
    Triple-Stream Multimodal HMIL Model:
    - Sagittal Stream -> Target-Specific Attention Pool
    - Coronal Stream -> Target-Specific Attention Pool
    - Axial Stream -> Target-Specific Attention Pool
    - Pathologically-Biased Gated View Fusion + 16-dim Metadata -> 12 Target Logits
    """

    def __init__(
        self,
        num_targets: int = 12,
        in_channels: int = 3,
        meta_dim: int = 16,
        feature_dim: int = 128,
        dropout: float = 0.2,
        use_anatomical_priors: bool = True,
    ):
        super().__init__()
        self.num_targets = num_targets
        self.feature_dim = feature_dim
        self.planes = ["sagittal", "coronal", "axial"]
        self.use_anatomical_priors = use_anatomical_priors

        # 1. Plane-Specific 2D CNN Stems
        self.stems = nn.ModuleDict({
            p: create_plane_stem(in_channels=in_channels, feature_dim=feature_dim)
            for p in self.planes
        })

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

        if self.use_anatomical_priors:
            self._init_anatomical_priors()

    def _init_anatomical_priors(self):
        """
        Initializes view gate final layer bias with clinical prior weights:
        planes: [0: Sagittal, 1: Coronal, 2: Axial]
        """
        # Target map:
        # 0: ACL (Sagittal bias)
        # 1: MCL (Coronal bias)
        # 2: Medial Meniscus (Sagittal + Coronal)
        # 3: Lateral Meniscus (Sagittal + Coronal)
        # 4: Medial OA (Coronal bias)
        # 5: Lateral OA (Coronal bias)
        # 6: PF OA (Axial bias)
        # 7: Effusion (Axial + Sagittal)
        # 8: Synovitis (Axial + Sagittal)
        # 9: Baker's (Sagittal + Axial)
        # 10: Contusion (Balanced)
        # 11: Fracture (Balanced)
        priors = {
            0: [1.5, 0.0, -0.5],
            1: [-0.5, 2.0, -0.5],
            2: [1.0, 1.0, -0.5],
            3: [1.0, 1.0, -0.5],
            4: [-0.5, 1.5, -0.5],
            5: [-0.5, 2.0, -0.5],
            6: [-0.5, -0.5, 2.0],
            7: [0.5, -0.5, 1.5],
            8: [0.5, -0.5, 1.5],
            9: [1.5, -0.5, 1.0],
            10: [0.5, 0.5, 0.5],
            11: [0.5, 0.5, 0.5],
        }
        for k, bias_vals in priors.items():
            last_linear = self.view_gates[k][2]
            with torch.no_grad():
                last_linear.bias.copy_(torch.tensor(bias_vals, dtype=torch.float32))

    def forward(
        self,
        sagittal_or_inputs: Union[Dict[str, torch.Tensor], torch.Tensor],
        coronal: Optional[torch.Tensor] = None,
        axial: Optional[torch.Tensor] = None,
        metadata: Optional[torch.Tensor] = None,
        meta_features: Optional[torch.Tensor] = None,
        plane_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
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

        # Process Visual Planes with plane-specific stems
        plane_reps: Dict[str, torch.Tensor] = {}
        for p in self.planes:
            if p in plane_inputs and plane_inputs[p] is not None:
                x = plane_inputs[p]
                B_p, S, C, H, W = x.shape
                # Pass through dedicated plane-specific stem
                feats_flat = self.stems[p](x.view(B_p * S, C, H, W))
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
        meta_emb = self.meta_net(meta)

        # Multimodal Fusion & Prediction per Target
        logits_list = []
        for k in range(self.num_targets):
            k_concat = stacked_views[:, k, :, :].reshape(B, 3 * self.feature_dim)
            view_weights = F.softmax(self.view_gates[k](k_concat), dim=-1).unsqueeze(-1)
            fused_vis = torch.sum(stacked_views[:, k, :, :] * view_weights, dim=1)

            multimodal_k = torch.cat([fused_vis, meta_emb], dim=-1)
            multimodal_k = self.dropout(multimodal_k)

            k_logit = self.heads[k](multimodal_k)
            logits_list.append(k_logit)

        return torch.cat(logits_list, dim=-1)
