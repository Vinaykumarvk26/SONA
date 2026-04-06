from __future__ import annotations

import torch
import torch.nn as nn


class SERCNNBiLSTM(nn.Module):
    def __init__(self, n_mels: int = 128, num_classes: int = 7, hidden_size: int = 160, dropout: float = 0.45):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d((2, 2)),
        )

        self.lstm = nn.LSTM(
            input_size=128 * (n_mels // 8),
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.25,
            bidirectional=True,
        )

        self.norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        out, _ = self.lstm(x)
        emb = out.mean(dim=1)
        return self.norm(emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.forward_features(x)
        emb = self.dropout(emb)
        return self.fc(emb)
