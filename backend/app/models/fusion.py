from typing import Dict

import torch
import torch.nn as nn

from app.schemas import EMOTIONS


DEVICE_MAP = {
    "mobile": 0,
    "desktop": 1,
    "tablet": 2,
    "smart_speaker": 3,
}


class ContextEncoder(nn.Module):
    def __init__(self, history_size: int = 20, embedding_dim: int = 64):
        super().__init__()
        self.history_size = history_size
        input_dim = 2 + len(DEVICE_MAP) + history_size
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, time_of_day: torch.Tensor, skip_rate: torch.Tensor, device_one_hot: torch.Tensor, history: torch.Tensor):
        x = torch.cat([time_of_day, skip_rate, device_one_hot, history], dim=1)
        return self.net(x)


class AttentionFusion(nn.Module):
    def __init__(self, face_dim: int, speech_dim: int, context_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.face_proj = nn.Linear(face_dim, hidden_dim)
        self.speech_proj = nn.Linear(speech_dim, hidden_dim)
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.classifier = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, len(EMOTIONS)))

    def forward(self, face_vec: torch.Tensor, speech_vec: torch.Tensor, context_vec: torch.Tensor):
        tokens = torch.stack([
            self.face_proj(face_vec),
            self.speech_proj(speech_vec),
            self.context_proj(context_vec),
        ], dim=1)
        fused, _ = self.attn(tokens, tokens, tokens)
        pooled = fused.mean(dim=1)
        logits = self.classifier(pooled)
        return logits, pooled


def softmax_dict(logits: torch.Tensor) -> Dict[str, float]:
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy().tolist()
    return {emotion: float(prob) for emotion, prob in zip(EMOTIONS, probs)}
