"""
Multiple-Instance Learning (MIL) pooling layers for MRI slice bags.
Implements Target-Specific Attention Pooling and Gated Attention Pooling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TargetSpecificAttentionPooling(nn.Module):
    """
    Computes distinct attention weights over slices for each of the 12 target abnormalities.
    Since an ACL tear and a Baker's cyst are located in different slice regions,
    each target learns its own attention distribution over the bag of slices.
    """

    def __init__(self, in_features: int, num_targets: int = 12, hidden_dim: int = 128):
        super().__init__()
        self.num_targets = num_targets
        self.in_features = in_features

        # Attention projection per target
        self.attention_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(num_targets)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch_size, Num_slices, in_features)
            mask: Optional boolean tensor of shape (Batch_size, Num_slices) where True = valid slice
        Returns:
            pooled: Tensor of shape (Batch_size, num_targets, in_features)
        """
        B, S, D = x.shape
        target_reps = []

        for k in range(self.num_targets):
            # Compute raw attention logits: (B, S, 1)
            attn_logits = self.attention_nets[k](x)  # (B, S, 1)
            attn_logits = attn_logits.squeeze(-1)    # (B, S)

            if mask is not None:
                attn_logits = attn_logits.masked_fill(~mask, -1e9)

            attn_weights = F.softmax(attn_logits, dim=-1)  # (B, S)
            attn_weights = attn_weights.unsqueeze(-1)      # (B, S, 1)

            # Weighted sum over slice dimension: (B, D)
            rep = torch.sum(x * attn_weights, dim=1)       # (B, D)
            target_reps.append(rep)

        # Stack into (B, num_targets, D)
        return torch.stack(target_reps, dim=1)


class GatedAttentionPooling(nn.Module):
    """
    Ilse et al. Gated Attention MIL mechanism:
    a = softmax(w^T (tanh(V h) * sigm(U h)))
    """

    def __init__(self, in_features: int, hidden_dim: int = 128):
        super().__init__()
        self.v_net = nn.Linear(in_features, hidden_dim)
        self.u_net = nn.Linear(in_features, hidden_dim)
        self.w_net = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (B, S, D)
        Returns:
            pooled: (B, D)
        """
        v = torch.tanh(self.v_net(x))   # (B, S, hidden)
        u = torch.sigmoid(self.u_net(x)) # (B, S, hidden)
        gated = v * u

        attn_logits = self.w_net(gated).squeeze(-1)  # (B, S)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(~mask, -1e9)

        attn_weights = F.softmax(attn_logits, dim=-1).unsqueeze(-1)  # (B, S, 1)
        pooled = torch.sum(x * attn_weights, dim=1)  # (B, D)
        return pooled
