from __future__ import annotations

import torch
import torch.nn as nn


class ViTBlock(nn.Module):
    def __init__(self, embed_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class FERCNNViT(nn.Module):
    """CNN feature extractor + ViT block on patches for FER2013 grayscale 48x48."""

    def __init__(self, num_classes: int = 7, dropout: float = 0.5, embedding_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 24x24
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 12x12
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 6x6
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.patch_proj = nn.Conv2d(256, embedding_dim, kernel_size=1)
        self.vit = ViTBlock(embed_dim=embedding_dim, heads=4, dropout=0.1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)  # B,256,6,6
        x = self.patch_proj(x)  # B,E,6,6
        x = x.flatten(2).transpose(1, 2)  # B,36,E
        x = self.vit(x)
        emb = x.mean(dim=1)  # B,E
        return emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.forward_features(x)
        emb = self.dropout(emb)
        return self.fc(emb)
