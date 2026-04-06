from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from app.ml.constants import FER_EMOTIONS
from app.services.face_service import FaceEmotionService
from app.services.speech_service import SpeechEmotionService


@dataclass
class LegacyEmotionPrediction:
    emotion: str
    confidence: float
    scores: Dict[str, float]
    embedding: torch.Tensor


class LegacyInferenceEngine:
    """
    Compatibility inference engine for previously trained checkpoints
    (fer_hybrid.pt, speech_emotion.pt). This keeps real inference working
    while newer FER/SER checkpoints are being trained.
    """

    def __init__(self, fer_model: torch.nn.Module, ser_model: torch.nn.Module, device: torch.device):
        self.device = device
        self.face_service = FaceEmotionService(fer_model, device)
        self.speech_service = SpeechEmotionService(ser_model, device)

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        merged = {emotion: float(scores.get(emotion, 0.0)) for emotion in FER_EMOTIONS}
        total = sum(merged.values()) + 1e-8
        return {k: v / total for k, v in merged.items()}

    def predict_face(self, image_bytes: bytes) -> LegacyEmotionPrediction:
        scores, label, conf = self.face_service.predict_from_bytes(image_bytes)
        norm = self._normalize_scores(scores)
        embedding = torch.zeros(1, 256, device=self.device)
        return LegacyEmotionPrediction(emotion=label, confidence=float(conf), scores=norm, embedding=embedding)

    def predict_voice(self, wav_bytes: bytes) -> LegacyEmotionPrediction:
        scores, label, conf = self.speech_service.predict_from_wav_bytes(wav_bytes)
        norm = self._normalize_scores(scores)
        embedding = torch.zeros(1, 128, device=self.device)
        return LegacyEmotionPrediction(emotion=label, confidence=float(conf), scores=norm, embedding=embedding)

    def predict_multimodal(self, face_pred: LegacyEmotionPrediction, voice_pred: LegacyEmotionPrediction):
        f_conf = max(face_pred.scores.values())
        v_conf = max(voice_pred.scores.values())
        total = f_conf + v_conf + 1e-8
        wf = f_conf / total
        wv = v_conf / total

        final_scores = {
            e: float(wf * face_pred.scores[e] + wv * voice_pred.scores[e])
            for e in FER_EMOTIONS
        }

        label, conf = max(final_scores.items(), key=lambda kv: kv[1])
        return {
            "emotion": label,
            "confidence": float(conf),
            "scores": final_scores,
            "weights": {
                "face": float(wf),
                "voice": float(wv),
                "feature_blend_alpha": 0.0,
            },
        }


class MixedInferenceEngine:
    """
    Uses the original first-generation facial pipeline while preserving the
    current speech pipeline. This is the safest rollback path when face
    accuracy regresses after newer FER updates.
    """

    def __init__(self, face_model: torch.nn.Module, voice_engine, device: torch.device):
        self.device = device
        self.face_service = FaceEmotionService(face_model, device)
        self.voice_engine = voice_engine

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        merged = {emotion: float(scores.get(emotion, 0.0)) for emotion in FER_EMOTIONS}
        total = sum(merged.values()) + 1e-8
        return {k: v / total for k, v in merged.items()}

    def predict_face(self, image_bytes: bytes) -> LegacyEmotionPrediction:
        scores, label, conf = self.face_service.predict_from_bytes(image_bytes)
        norm = self._normalize_scores(scores)
        embedding = torch.zeros(1, 256, device=self.device)
        return LegacyEmotionPrediction(emotion=label, confidence=float(conf), scores=norm, embedding=embedding)

    def predict_voice(self, wav_bytes: bytes):
        return self.voice_engine.predict_voice(wav_bytes)

    def predict_multimodal(
        self,
        face_pred,
        voice_pred,
        has_face: bool = True,
        has_voice: bool = True,
    ):
        if hasattr(self.voice_engine, "predict_multimodal"):
            return self.voice_engine.predict_multimodal(
                face_pred,
                voice_pred,
                has_face=has_face,
                has_voice=has_voice,
            )

        f_conf = max(face_pred.scores.values())
        v_conf = max(voice_pred.scores.values())
        total = f_conf + v_conf + 1e-8
        wf = f_conf / total
        wv = v_conf / total
        final_scores = {
            e: float(wf * face_pred.scores[e] + wv * voice_pred.scores[e])
            for e in FER_EMOTIONS
        }
        label, conf = max(final_scores.items(), key=lambda kv: kv[1])
        return {
            "emotion": label,
            "confidence": float(conf),
            "scores": final_scores,
            "weights": {
                "face": float(wf),
                "voice": float(wv),
                "feature_blend_alpha": 0.0,
            },
        }
