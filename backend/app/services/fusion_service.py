from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch

from app.models.fusion import AttentionFusion, ContextEncoder, DEVICE_MAP
from app.schemas import EMOTIONS


@dataclass
class FusionOutput:
    scores: Dict[str, float]
    label: str
    confidence: float
    weights: Dict[str, float]


class MultimodalFusionService:
    def __init__(self, device: torch.device):
        self.device = device
        self.context_encoder = ContextEncoder(history_size=20, embedding_dim=64).to(device)
        self.attention_fusion = AttentionFusion(face_dim=7, speech_dim=7, context_dim=64, hidden_dim=128).to(device)
        self.context_encoder.eval()
        self.attention_fusion.eval()

    def _device_one_hot(self, name: str):
        vec = torch.zeros(1, len(DEVICE_MAP), device=self.device)
        idx = DEVICE_MAP.get(name.lower(), 1)
        vec[0, idx] = 1.0
        return vec

    def fuse(self, face_scores: Dict[str, float], speech_scores: Dict[str, float], context: dict) -> FusionOutput:
        face_vec = torch.tensor([[face_scores[e] for e in EMOTIONS]], device=self.device, dtype=torch.float32)
        speech_vec = torch.tensor([[speech_scores[e] for e in EMOTIONS]], device=self.device, dtype=torch.float32)

        time_of_day = torch.tensor([[context["time_of_day"]]], device=self.device)
        skip_rate = torch.tensor([[context["skip_rate"]]], device=self.device)
        device_one_hot = self._device_one_hot(context["device_type"])

        history = context.get("listening_history", [])[:20]
        if len(history) < 20:
            history = history + [0.0] * (20 - len(history))
        history_tensor = torch.tensor([history], device=self.device, dtype=torch.float32)

        with torch.no_grad():
            context_vec = self.context_encoder(time_of_day, skip_rate, device_one_hot, history_tensor)
            attn_logits, _ = self.attention_fusion(face_vec, speech_vec, context_vec)
            attn_probs = torch.softmax(attn_logits, dim=-1).cpu().numpy()[0]

        face_conf = max(face_scores.values())
        speech_conf = max(speech_scores.values())
        ctx_conf = float(1.0 - min(1.0, context["skip_rate"]))

        total = face_conf + speech_conf + ctx_conf + 1e-8
        wf = face_conf / total
        ws = speech_conf / total
        wc = ctx_conf / total

        decision = {}
        for i, emotion in enumerate(EMOTIONS):
            base_mix = wf * face_scores[emotion] + ws * speech_scores[emotion]
            decision[emotion] = 0.65 * base_mix + 0.35 * float(attn_probs[i])

        z = np.array([decision[e] for e in EMOTIONS], dtype=np.float32)
        z = np.exp(z - z.max())
        z = z / z.sum()

        scores = {emotion: float(prob) for emotion, prob in zip(EMOTIONS, z.tolist())}
        idx = int(np.argmax(z))

        return FusionOutput(
            scores=scores,
            label=EMOTIONS[idx],
            confidence=float(z[idx]),
            weights={"facial": float(wf), "speech": float(ws), "context": float(wc)},
        )
