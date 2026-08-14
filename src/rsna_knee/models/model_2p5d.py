"""
2.5D Multi-Slice Deep Learning Architecture for Knee MRI Abnormality Classification.
Combines 2D backbone feature extraction with Target-Specific Attention MIL pooling.
"""

from typing import Optional
import timm
import torch
import torch.nn as nn

from rsna_knee.models.pooling import TargetSpecificAttentionPooling


class Knee2p5dModel(nn.Module):
    """
    Knee 2.5D Model:
    Input: (Batch_size, Num_slices, 3, H, W)
    Encoder: 2D Pretrained CNN / Vision Transformer extracting (B*S, Feature_dim)
    Pooling: Target-Specific Slice Attention producing (B, 12, Feature_dim)
    Heads: 12 independent linear classifiers outputting raw logits for all 12 abnormalities.
    """

    def __init__(
        self,
        backbone_name: str = "resnet34d",
        pretrained: bool = True,
        num_targets: int = 12,
        in_channels: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_targets = num_targets
        self.backbone_name = backbone_name

        # Initialize 2D feature extractor
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,  # Remove default classifier head to get feature vectors
        )

        # Feature dimension
        self.feature_dim = self.encoder.num_features

        # MIL Target-Specific Attention
        self.attention_pool = TargetSpecificAttentionPooling(
            in_features=self.feature_dim,
            num_targets=num_targets,
            hidden_dim=128,
        )

        # Target classification heads
        self.dropout = nn.Dropout(p=dropout)
        self.heads = nn.ModuleList([
            nn.Linear(self.feature_dim, 1) for _ in range(num_targets)
        ])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, S, C, H, W) where:
               B = Batch size (studies)
               S = Number of sampled 2.5D slice triplets
               C = Channels (typically 3 for z-1, z, z+1)
               H, W = Image height and width
            mask: Optional boolean mask of valid slices (B, S)
        Returns:
            logits: Output logits of shape (B, 12)
        """
        B, S, C, H, W = x.shape

        # Flatten batch and slices for encoder: (B * S, C, H, W)
        x_flat = x.view(B * S, C, H, W)
        feats_flat = self.encoder(x_flat)  # (B * S, feature_dim)

        # Reshape back to bag of slices: (B, S, feature_dim)
        feats = feats_flat.view(B, S, self.feature_dim)

        # Apply target-specific MIL pooling: (B, num_targets, feature_dim)
        pooled_targets = self.attention_pool(feats, mask=mask)
        pooled_targets = self.dropout(pooled_targets)

        # Compute per-target logit
        logits_list = []
        for k in range(self.num_targets):
            k_feat = pooled_targets[:, k, :]          # (B, feature_dim)
            k_logit = self.heads[k](k_feat)           # (B, 1)
            logits_list.append(k_logit)

        # Shape: (B, num_targets)
        logits = torch.cat(logits_list, dim=-1)
        return logits
