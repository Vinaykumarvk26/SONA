from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Dict

import cv2
import librosa
import numpy as np
import soundfile as sf
import torch
from PIL import Image
from torchvision import transforms
from typing import TYPE_CHECKING

from app.ml.constants import FER_EMOTIONS
from app.ml.models import FERCNNViT, FeatureFusionAttention, SERCNNLSTM

if TYPE_CHECKING:
    from app.ml.hf_fer import HFFaceEmotionModel
    from app.ml.hf_ser import HFWav2VecSER


logger = logging.getLogger("ml_inference")


@dataclass
class EmotionPrediction:
    emotion: str
    confidence: float
    scores: Dict[str, float]
    embedding: torch.Tensor


class EmotionInferenceEngine:
    def __init__(
        self,
        fer_model: FERCNNViT,
        ser_model: SERCNNLSTM,
        fusion_model: FeatureFusionAttention,
        device: torch.device,
        sample_rate: int = 16000,
        n_mfcc: int = 40,
        use_feature_fusion: bool = True,
        fer_hf_model: HFFaceEmotionModel | None = None,
        ser_hf_model: HFWav2VecSER | None = None,
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.use_feature_fusion = use_feature_fusion
        self.fer_hf_model = fer_hf_model
        self.ser_hf_model = ser_hf_model

        self.fer_model = fer_model.to(device).eval()
        self.ser_model = ser_model.to(device).eval()
        self.fusion_model = fusion_model.to(device).eval()

        self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.smile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
        self.face_transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

    def _softmax_scores(self, logits: torch.Tensor) -> Dict[str, float]:
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
        return {e: float(p) for e, p in zip(FER_EMOTIONS, probs)}

    def _to_prediction(self, logits: torch.Tensor, embedding: torch.Tensor) -> EmotionPrediction:
        scores = self._softmax_scores(logits)
        label, conf = max(scores.items(), key=lambda kv: kv[1])
        return EmotionPrediction(emotion=label, confidence=float(conf), scores=scores, embedding=embedding)

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        z = np.array([max(1e-9, float(scores.get(e, 0.0))) for e in FER_EMOTIONS], dtype=np.float64)
        z = z / z.sum()
        return {e: float(v) for e, v in zip(FER_EMOTIONS, z.tolist())}

    def _prediction_from_scores(self, scores: Dict[str, float], embedding: torch.Tensor) -> EmotionPrediction:
        scores = self._normalize_scores(scores)
        label, conf = max(scores.items(), key=lambda kv: kv[1])
        return EmotionPrediction(emotion=label, confidence=float(conf), scores=scores, embedding=embedding)

    def _detect_face_crop(self, bgr: np.ndarray) -> np.ndarray | None:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        img_h, img_w = gray.shape[:2]
        min_dim = min(img_h, img_w)
        min_size = max(56, int(min_dim * 0.12))
        faces = self.haar.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=6,
            minSize=(min_size, min_size),
        )
        if len(faces) == 0:
            return None

        x, y, w, h = max(faces, key=lambda box: int(box[2]) * int(box[3]))
        area_ratio = (w * h) / float(img_h * img_w)
        if area_ratio < 0.06:
            return None

        return bgr[y : y + h, x : x + w]

    def _has_smile(self, face_bgr: np.ndarray) -> bool:
        if self.smile.empty():
            return False
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        smiles = self.smile.detectMultiScale(
            gray,
            scaleFactor=1.7,
            minNeighbors=18,
            minSize=(24, 24),
        )
        return len(smiles) > 0

    def predict_face(self, image_bytes: bytes) -> EmotionPrediction:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        face = self._detect_face_crop(bgr)
        if face is None:
            logger.warning("No face detected; rejecting image")
            raise ValueError("Invalid photo: no human face detected. Please upload a clear front-face image.")

        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        x = self.face_transform(Image.fromarray(face_rgb)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.fer_model.forward_features(x)
            logits = self.fer_model.fc(self.fer_model.dropout(emb))
        local_scores = self._softmax_scores(logits)

        hf_scores = None
        if self.fer_hf_model is not None:
            try:
                hf_scores = self.fer_hf_model.predict_scores(face_rgb)
            except Exception as exc:
                logger.warning("HF FER inference failed; falling back to local FER only: %s", exc)

        if hf_scores is None:
            scores = self._normalize_scores(local_scores)
        else:
            local_quality = self._prediction_quality(local_scores)
            hf_quality = self._prediction_quality(hf_scores)
            local_weight = max(0.20, local_quality)
            hf_weight = max(0.25, hf_quality) * 1.25
            norm = local_weight + hf_weight + 1e-8
            local_weight /= norm
            hf_weight /= norm
            scores = {
                emotion: float(local_weight * local_scores[emotion] + hf_weight * hf_scores[emotion])
                for emotion in FER_EMOTIONS
            }
            scores = self._normalize_scores(scores)

        has_smile = self._has_smile(face)
        if has_smile:
            scores["happy"] = scores["happy"] + 0.20
            scores["sad"] = scores["sad"] * 0.70
            scores["fear"] = scores["fear"] * 0.72
            scores["disgust"] = scores["disgust"] * 0.65
        calibrated = self._normalize_scores(scores)
        top_label, top_conf = max(calibrated.items(), key=lambda kv: kv[1])
        if has_smile and top_label in {"sad", "fear", "neutral"} and calibrated["happy"] >= max(0.20, top_conf - 0.10):
            calibrated["happy"] = max(calibrated["happy"], top_conf + 0.08)
        return self._prediction_from_scores(calibrated, emb)

    def _extract_mfcc(self, wav_bytes: bytes) -> tuple[torch.Tensor, Dict[str, float], np.ndarray]:
        y, sr = sf.read(io.BytesIO(wav_bytes))
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = y.astype(np.float32, copy=False)
        if sr != self.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
        if len(y) == 0:
            raise ValueError("Voice input is empty. Please record your voice again.")

        y, _ = librosa.effects.trim(y, top_db=26)
        if len(y) < int(self.sample_rate * 0.75):
            raise ValueError("Voice clip too short. Please speak clearly for at least 1 second.")

        rms_energy = float(np.sqrt(np.mean(np.square(y)))) if len(y) > 0 else 0.0
        peak = float(np.max(np.abs(y))) if len(y) > 0 else 0.0
        if rms_energy < 0.003 or peak < 0.02:
            raise ValueError("Voice too low/noisy. Please speak louder and closer to the microphone.")

        y = y / (peak + 1e-6)

        zcr = float(librosa.feature.zero_crossing_rate(y, frame_length=1024, hop_length=256).mean())
        flatness = float(librosa.feature.spectral_flatness(y=y + 1e-9).mean())
        if zcr > 0.22 and flatness > 0.24:
            raise ValueError("Could not detect clear speech content. Please speak clearly with less background noise.")

        centroid = float(librosa.feature.spectral_centroid(y=y, sr=self.sample_rate).mean())
        centroid_norm = float(np.clip(centroid / (self.sample_rate / 2.0), 0.0, 1.0))

        mfcc = librosa.feature.mfcc(y=y, sr=self.sample_rate, n_mfcc=self.n_mfcc)
        mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-6)
        if mfcc.shape[1] < 180:
            mfcc = np.pad(mfcc, ((0, 0), (0, 180 - mfcc.shape[1])), mode="constant")
        else:
            mfcc = mfcc[:, :180]

        x = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        voice_stats = {
            "rms": float(rms_energy),
            "peak": float(peak),
            "zcr": float(zcr),
            "flatness": float(flatness),
            "centroid_norm": float(centroid_norm),
        }
        return x, voice_stats, y

    def _prediction_quality(self, scores: Dict[str, float]) -> float:
        probs = np.array([max(1e-9, float(scores.get(e, 0.0))) for e in FER_EMOTIONS], dtype=np.float64)
        probs = probs / probs.sum()
        ranked = sorted(probs.tolist(), reverse=True)
        top = ranked[0]
        second = ranked[1] if len(ranked) > 1 else 0.0
        margin = max(0.0, top - second)
        entropy = float(-(probs * np.log(probs + 1e-9)).sum() / np.log(len(probs)))
        certainty = 1.0 - entropy
        return float(top * (0.55 + 0.45 * margin) * (0.6 + 0.4 * certainty))

    def predict_voice(self, wav_bytes: bytes) -> EmotionPrediction:
        x, voice_stats, waveform = self._extract_mfcc(wav_bytes)
        with torch.no_grad():
            emb = self.ser_model.forward_features(x)
            logits = self.ser_model.fc(self.ser_model.dropout(emb))
        local_scores = self._normalize_scores(self._softmax_scores(logits))

        hf_scores = None
        if self.ser_hf_model is not None:
            try:
                hf_scores = self.ser_hf_model.predict_scores(waveform, sample_rate=self.sample_rate)
            except Exception as exc:
                logger.warning("HF SER inference failed; falling back to local SER only: %s", exc)

        if hf_scores is None:
            final_scores = local_scores
        else:
            local_quality = self._prediction_quality(local_scores)
            hf_quality = self._prediction_quality(hf_scores)
            local_weight = max(0.15, local_quality)
            hf_weight = max(0.25, hf_quality) * 1.35
            norm = local_weight + hf_weight + 1e-8
            local_weight /= norm
            hf_weight /= norm

            final_scores = {
                emotion: float(local_weight * local_scores[emotion] + hf_weight * hf_scores[emotion])
                for emotion in FER_EMOTIONS
            }
            final_scores = self._normalize_scores(final_scores)

        ranked = sorted(final_scores.items(), key=lambda kv: kv[1], reverse=True)
        top_label, top_conf = ranked[0]
        second_label, second_conf = ranked[1]
        if top_label == "neutral" and top_conf < 0.58 and second_conf >= max(0.16, top_conf - 0.09):
            adjusted = dict(final_scores)
            adjusted["neutral"] *= 0.78
            adjusted[second_label] *= 1.22
            final_scores = self._normalize_scores(adjusted)

        ranked = sorted(final_scores.items(), key=lambda kv: kv[1], reverse=True)
        top_label, top_conf = ranked[0]
        happy_score = float(final_scores.get("happy", 0.0))
        energetic_or_bright = (
            voice_stats.get("rms", 0.0) >= 0.045
            or voice_stats.get("centroid_norm", 0.0) >= 0.26
        )
        close_to_happy = happy_score >= max(0.17, top_conf - 0.12)
        if top_label in {"sad", "neutral"} and energetic_or_bright and close_to_happy:
            adjusted = dict(final_scores)
            adjusted["happy"] *= 1.20
            if top_label == "sad":
                adjusted["sad"] *= 0.84
            else:
                adjusted["neutral"] *= 0.88
            final_scores = self._normalize_scores(adjusted)

        label, conf = max(final_scores.items(), key=lambda kv: kv[1])
        if conf < 0.21:
            raise ValueError("Unable to detect a clear voice emotion. Please speak with clearer tone for 2-3 seconds.")

        return self._prediction_from_scores(final_scores, emb)

    def predict_multimodal(
        self,
        face_pred: EmotionPrediction,
        voice_pred: EmotionPrediction,
        has_face: bool = True,
        has_voice: bool = True,
    ):
        if has_face and not has_voice:
            return {
                "emotion": face_pred.emotion,
                "confidence": float(face_pred.confidence),
                "scores": face_pred.scores,
                "weights": {"face": 1.0, "voice": 0.0, "feature_blend_alpha": 0.0},
                "feature_level_scores": None,
                "decision_level_scores": face_pred.scores,
            }

        if has_voice and not has_face:
            return {
                "emotion": voice_pred.emotion,
                "confidence": float(voice_pred.confidence),
                "scores": voice_pred.scores,
                "weights": {"face": 0.0, "voice": 1.0, "feature_blend_alpha": 0.0},
                "feature_level_scores": None,
                "decision_level_scores": voice_pred.scores,
            }

        f_conf = max(face_pred.scores.values())
        v_conf = max(voice_pred.scores.values())
        total = f_conf + v_conf + 1e-8
        wf = f_conf / total
        wv = v_conf / total

        decision_scores = {}
        for e in FER_EMOTIONS:
            decision_scores[e] = wf * face_pred.scores[e] + wv * voice_pred.scores[e]

        feature_scores = None
        alpha = 0.0
        if self.use_feature_fusion:
            with torch.no_grad():
                feature_logits = self.fusion_model(face_pred.embedding, voice_pred.embedding)
                feature_scores = self._softmax_scores(feature_logits)

            joint_conf = min(f_conf, v_conf)
            alpha = float(0.25 + 0.75 * joint_conf)

        if feature_scores is None:
            final_scores = dict(decision_scores)
        else:
            final_scores = {e: alpha * feature_scores[e] + (1 - alpha) * decision_scores[e] for e in FER_EMOTIONS}

        z = np.array([final_scores[e] for e in FER_EMOTIONS], dtype=np.float32)
        z = np.exp(z - z.max())
        z = z / z.sum()
        final_scores = {e: float(p) for e, p in zip(FER_EMOTIONS, z.tolist())}

        label, conf = max(final_scores.items(), key=lambda kv: kv[1])

        return {
            "emotion": label,
            "confidence": float(conf),
            "scores": final_scores,
            "weights": {
                "face": float(wf),
                "voice": float(wv),
                "feature_blend_alpha": alpha,
            },
            "feature_level_scores": feature_scores,
            "decision_level_scores": decision_scores,
        }
