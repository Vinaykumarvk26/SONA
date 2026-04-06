from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from app.ml.constants import FER_EMOTIONS


logger = logging.getLogger("hf_fer")

LABEL_MAP = {
    "angry": "angry",
    "anger": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "fearful": "fear",
    "happy": "happy",
    "happiness": "happy",
    "joy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "sadness": "sad",
    "surprise": "surprise",
    "surprised": "surprise",
}


def _normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    z = np.array([max(1e-9, float(scores.get(e, 0.0))) for e in FER_EMOTIONS], dtype=np.float64)
    z = z / z.sum()
    return {e: float(v) for e, v in zip(FER_EMOTIONS, z.tolist())}


class HFFaceEmotionModel:
    def __init__(
        self,
        model_id: str,
        device: torch.device,
        local_files_only: bool = False,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.local_files_only = local_files_only

        self.image_processor = AutoImageProcessor.from_pretrained(
            self.model_id,
            local_files_only=self.local_files_only,
        )
        self.model = AutoModelForImageClassification.from_pretrained(
            self.model_id,
            local_files_only=self.local_files_only,
        ).to(self.device).eval()

        raw_id2label = self.model.config.id2label or {}
        self.id2label: dict[int, str] = {}
        for key, value in raw_id2label.items():
            try:
                idx = int(key)
            except Exception:
                continue
            self.id2label[idx] = str(value)
        logger.info("Loaded HF FER model '%s' with labels: %s", self.model_id, self.id2label)

    def predict_scores(self, face_rgb: np.ndarray) -> Dict[str, float]:
        if face_rgb.size == 0:
            raise ValueError("Invalid face image")

        image = Image.fromarray(face_rgb)
        inputs = self.image_processor(images=image, return_tensors="pt")
        tensor_inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**tensor_inputs).logits.squeeze(0)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

        scores = {emotion: 0.0 for emotion in FER_EMOTIONS}
        for idx, prob in enumerate(probs.tolist()):
            raw_label = self.id2label.get(idx, "neutral").lower().strip()
            mapped = LABEL_MAP.get(raw_label, raw_label)
            if mapped in scores:
                scores[mapped] += float(prob)
        return _normalize_scores(scores)
