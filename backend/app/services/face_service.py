import io
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from app.config import get_settings
from app.schemas import EMOTIONS


class FaceEmotionService:
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.settings = get_settings()
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        self.mtcnn = None
        if self.settings.face_detector.lower() == "mtcnn":
            try:
                from facenet_pytorch import MTCNN  # type: ignore

                self.mtcnn = MTCNN(keep_all=False, device=device)
            except Exception:
                self.mtcnn = None
        self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.smile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _detect_face(self, bgr_image: np.ndarray) -> Optional[np.ndarray]:
        if self.mtcnn is not None:
            rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            boxes, _ = self.mtcnn.detect(rgb)
            if boxes is not None and len(boxes) > 0:
                x1, y1, x2, y2 = [int(v) for v in boxes[0]]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(bgr_image.shape[1], x2), min(bgr_image.shape[0], y2)
                return bgr_image[y1:y2, x1:x2]

        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self.haar.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48),
        )
        if len(faces) == 0:
            return None

        # Use largest face when multiple detections exist.
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        return bgr_image[y : y + h, x : x + w]

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

    def predict_from_bytes(self, image_bytes: bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        face = self._detect_face(bgr)
        if face is None:
            raise ValueError("No clear face detected. Please keep your face centered and try again.")

        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(face_rgb)
        x = self.transform(pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        scores = {emotion: float(prob) for emotion, prob in zip(EMOTIONS, probs)}
        has_smile = self._has_smile(face)
        if has_smile:
            scores["happy"] = scores.get("happy", 0.0) + 0.18
            scores["surprise"] = scores.get("surprise", 0.0) * 0.82
            scores["fear"] = scores.get("fear", 0.0) * 0.82
            total = sum(scores.values()) + 1e-8
            scores = {emotion: float(score / total) for emotion, score in scores.items()}
        idx = int(np.argmax(probs))
        label, conf = max(scores.items(), key=lambda item: item[1])
        return scores, label, float(conf)

    def predict_from_frame(self, frame: np.ndarray) -> Tuple[dict, str, float]:
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            return {emotion: 1.0 / len(EMOTIONS) for emotion in EMOTIONS}, "neutral", 0.14
        return self.predict_from_bytes(encoded.tobytes())
