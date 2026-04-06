from __future__ import annotations

import torch
import torch.nn as nn


class SERCNNLSTM(nn.Module):
    def __init__(self, n_mfcc: int = 40, num_classes: int = 7, hidden_size: int = 128, dropout: float = 0.5):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
        )

        self.lstm = nn.LSTM(
            input_size=64 * (n_mfcc // 4),
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=False,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)  # B,C,F,T
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)  # B,T,C*F
        out, _ = self.lstm(x)
        emb = out[:, -1, :]  # final time-step embedding
        return emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.forward_features(x)
        emb = self.dropout(emb)
        return self.fc(emb)
