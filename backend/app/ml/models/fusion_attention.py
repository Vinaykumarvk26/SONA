from __future__ import annotations

import torch
import torch.nn as nn


class FeatureFusionAttention(nn.Module):
    """Feature-level fusion: concat embeddings -> attention-like gate -> logits."""

    def __init__(self, face_dim: int, speech_dim: int, num_classes: int = 7):
        super().__init__()
        fused_dim = face_dim + speech_dim
        self.gate = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fused_dim, fused_dim),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, face_emb: torch.Tensor, speech_emb: torch.Tensor):
        x = torch.cat([face_emb, speech_emb], dim=1)
        gated = x * self.gate(x)
        logits = self.head(gated)
        return logits
