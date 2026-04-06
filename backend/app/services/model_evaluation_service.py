from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sqlalchemy import delete, select
from sqlalchemy.orm import Session
from torchvision import transforms

from app.config import get_settings
from app.ml.constants import FER_EMOTIONS
from app.ml.datasets.ravdess_dataset import build_ravdess_loaders
from app.ml.models import FERCNNViT, SERCNNLSTM
from app.models.model_metadata import ModelMetadata
from app.models.prediction_log import PredictionLog
from app.models.fer_model import FERHybridNet
from app.services.model_loader import ensure_model_checkpoint


@dataclass
class EvaluationResult:
    input_type: str
    dataset_name: str
    run_tag: str
    sample_count: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: list[list[int]]
    labels: list[str]
    emotion_distribution: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_type": self.input_type,
            "dataset_name": self.dataset_name,
            "run_tag": self.run_tag,
            "sample_count": self.sample_count,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "confusion_matrix": self.confusion_matrix,
            "labels": self.labels,
            "emotion_distribution": self.emotion_distribution,
        }


class ModelEvaluationService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.face_legacy_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.face_modern_transform = transforms.Compose(
            [
                transforms.Resize((48, 48)),
                transforms.ToTensor(),
            ]
        )

    def _utc_tag(self, prefix: str) -> str:
        return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

    def _resolve_fer_csv_path(self, csv_path: str | None = None) -> Path:
        if csv_path:
            path = Path(csv_path)
            if path.exists():
                return path

        candidates = [
            Path("backend/datasets/fer2013/fer2013/fer2013.csv"),
            Path("datasets/fer2013/fer2013/fer2013.csv"),
            Path("backend/datasets/fer2013/fer2013.csv"),
            Path("datasets/fer2013/fer2013.csv"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError("FER2013 CSV not found. Expected it under backend/datasets/fer2013.")

    def _load_face_model(self, device: torch.device) -> tuple[torch.nn.Module, str]:
        # Prefer the CNN+ViT checkpoint for evaluation (matches FER2013 label order and metrics).
        fer_path = Path(self.settings.fer_model_path)
        if fer_path.exists():
            model = FERCNNViT(num_classes=7, dropout=0.5)
            ensure_model_checkpoint(model, self.settings.fer_model_path, self.settings.fer_model_url)
            return model.to(device).eval(), "cnn_vit"

        # Fall back to legacy hybrid only if the CNN+ViT checkpoint is missing.
        legacy_path = Path(self.settings.legacy_fer_model_path)
        if legacy_path.exists():
            model = FERHybridNet(num_classes=7)
            ensure_model_checkpoint(model, str(legacy_path), "")
            return model.to(device).eval(), "legacy_hybrid"

        # Default to CNN+ViT if neither file is present (will trigger a download if URL is set).
        model = FERCNNViT(num_classes=7, dropout=0.5)
        ensure_model_checkpoint(model, self.settings.fer_model_path, self.settings.fer_model_url)
        return model.to(device).eval(), "cnn_vit"

    def _load_voice_model(self, device: torch.device) -> torch.nn.Module:
        model = SERCNNLSTM(n_mfcc=40, num_classes=7, dropout=0.5)
        ensure_model_checkpoint(model, self.settings.speech_model_path, self.settings.speech_model_url)
        return model.to(device).eval()

    def _clear_run_logs(self, db: Session, run_tag: str) -> None:
        db.execute(delete(PredictionLog).where(PredictionLog.run_tag == run_tag))

    def _store_logs(
        self,
        db: Session,
        *,
        input_type: str,
        dataset_name: str,
        run_tag: str,
        rows: list[dict[str, Any]],
    ) -> None:
        self._clear_run_logs(db, run_tag)
        db.add_all(
            [
                PredictionLog(
                    input_type=input_type,
                    dataset_name=dataset_name,
                    actual_label=row["actual_label"],
                    predicted_label=row["predicted_label"],
                    confidence_score=row["confidence_score"],
                    run_tag=run_tag,
                )
                for row in rows
            ]
        )
        db.commit()

    def _metrics_from_rows(
        self,
        *,
        input_type: str,
        dataset_name: str,
        run_tag: str,
        rows: list[dict[str, Any]],
    ) -> EvaluationResult:
        if not rows:
            return EvaluationResult(
                input_type=input_type,
                dataset_name=dataset_name,
                run_tag=run_tag,
                sample_count=0,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                confusion_matrix=[],
                labels=FER_EMOTIONS,
                emotion_distribution=[],
            )

        y_true = [row["actual_label"] for row in rows]
        y_pred = [row["predicted_label"] for row in rows]
        cm = confusion_matrix(y_true, y_pred, labels=FER_EMOTIONS)
        distribution = []
        for emotion in FER_EMOTIONS:
            distribution.append(
                {
                    "emotion": emotion,
                    "actual_count": int(sum(1 for label in y_true if label == emotion)),
                    "predicted_count": int(sum(1 for label in y_pred if label == emotion)),
                }
            )

        return EvaluationResult(
            input_type=input_type,
            dataset_name=dataset_name,
            run_tag=run_tag,
            sample_count=len(rows),
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred, labels=FER_EMOTIONS, average="weighted", zero_division=0)),
            recall=float(recall_score(y_true, y_pred, labels=FER_EMOTIONS, average="weighted", zero_division=0)),
            f1_score=float(f1_score(y_true, y_pred, labels=FER_EMOTIONS, average="weighted", zero_division=0)),
            confusion_matrix=cm.astype(int).tolist(),
            labels=list(FER_EMOTIONS),
            emotion_distribution=distribution,
        )

    def _upsert_model_metadata(self, db: Session, *, model_type: str, model_version: str, result: EvaluationResult) -> None:
        row = db.execute(
            select(ModelMetadata).where(
                ModelMetadata.model_type == model_type,
                ModelMetadata.model_version == model_version,
            )
        ).scalar_one_or_none()
        if row is None:
            row = ModelMetadata(model_type=model_type, model_version=model_version)
            db.add(row)

        row.accuracy = result.accuracy
        row.precision = result.precision
        row.recall = result.recall
        row.f1_score = result.f1_score
        db.commit()

    def run_face_evaluation(self, db: Session, csv_path: str | None = None) -> dict[str, Any]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, model_version = self._load_face_model(device)
        csv_file = self._resolve_fer_csv_path(csv_path)
        frame = pd.read_csv(csv_file)
        if {"emotion", "pixels"}.difference(frame.columns):
            raise ValueError("FER2013 CSV must contain 'emotion' and 'pixels' columns")

        if "Usage" in frame.columns:
            eval_frame = frame[frame["Usage"].isin(["PrivateTest", "PublicTest"])].copy()
            if eval_frame.empty:
                eval_frame = frame.copy()
        else:
            eval_frame = frame.copy()

        run_tag = self._utc_tag("face")
        rows: list[dict[str, Any]] = []
        for _, row in eval_frame.iterrows():
            pixels = np.fromstring(str(row["pixels"]), dtype=np.uint8, sep=" ").reshape(48, 48)
            if model_version == "legacy_hybrid":
                image = Image.fromarray(pixels, mode="L").convert("RGB")
                x = self.face_legacy_transform(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(x)
            else:
                image = Image.fromarray(pixels, mode="L")
                x = self.face_modern_transform(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model.forward_features(x)
                    logits = model.fc(model.dropout(emb))

            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            true_idx = int(row["emotion"])
            rows.append(
                {
                    "actual_label": FER_EMOTIONS[true_idx],
                    "predicted_label": FER_EMOTIONS[pred_idx],
                    "confidence_score": float(probs[pred_idx]),
                }
            )

        self._store_logs(db, input_type="face", dataset_name="FER2013", run_tag=run_tag, rows=rows)
        result = self._metrics_from_rows(input_type="face", dataset_name="FER2013", run_tag=run_tag, rows=rows)
        self._upsert_model_metadata(db, model_type="FER", model_version=model_version, result=result)
        return result.to_dict()

    def run_voice_evaluation(self, db: Session, data_dir: str | None = None) -> dict[str, Any]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self._load_voice_model(device)
        root = data_dir or "backend/datasets/ravdess"
        bundle = build_ravdess_loaders(root, batch_size=32, num_workers=0, seed=42, feature_type="mfcc", max_frames=180)

        run_tag = self._utc_tag("voice")
        rows: list[dict[str, Any]] = []
        with torch.no_grad():
            for x, y in bundle.test_loader:
                x = x.to(device, non_blocking=True)
                logits = model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                for idx, pred_idx in enumerate(preds.tolist()):
                    true_idx = int(y[idx].item())
                    rows.append(
                        {
                            "actual_label": FER_EMOTIONS[true_idx],
                            "predicted_label": FER_EMOTIONS[pred_idx],
                            "confidence_score": float(probs[idx][pred_idx]),
                        }
                    )

        self._store_logs(db, input_type="voice", dataset_name="RAVDESS", run_tag=run_tag, rows=rows)
        result = self._metrics_from_rows(input_type="voice", dataset_name="RAVDESS", run_tag=run_tag, rows=rows)
        self._upsert_model_metadata(db, model_type="SER", model_version="cnn_lstm", result=result)
        return result.to_dict()

    def run_evaluation(self, db: Session, input_type: str = "all") -> dict[str, Any]:
        requested = (input_type or "all").strip().lower()
        payload: dict[str, Any] = {}
        if requested in {"all", "face"}:
            payload["face"] = self.run_face_evaluation(db)
        if requested in {"all", "voice"}:
            payload["voice"] = self.run_voice_evaluation(db)
        if requested == "all":
            payload["overall"] = self.get_metrics_summary(db)
        return payload

    def _rows_for_latest_run(self, db: Session, input_type: str) -> tuple[str | None, list[PredictionLog]]:
        latest_tag = db.execute(
            select(PredictionLog.run_tag)
            .where(PredictionLog.input_type == input_type)
            .order_by(PredictionLog.timestamp.desc(), PredictionLog.id.desc())
            .limit(1)
        ).scalar_one_or_none()
        if latest_tag is None:
            return None, []
        rows = db.execute(
            select(PredictionLog).where(
                PredictionLog.input_type == input_type,
                PredictionLog.run_tag == latest_tag,
            )
        ).scalars().all()
        return latest_tag, rows

    def get_metrics_summary(self, db: Session, input_type: str | None = None) -> dict[str, Any]:
        requested = (input_type or "all").strip().lower()
        if requested in {"face", "voice"}:
            run_tag, rows = self._rows_for_latest_run(db, requested)
            result = self._metrics_from_rows(
                input_type=requested,
                dataset_name="FER2013" if requested == "face" else "RAVDESS",
                run_tag=run_tag or "",
                rows=[
                    {
                        "actual_label": row.actual_label,
                        "predicted_label": row.predicted_label,
                        "confidence_score": row.confidence_score,
                    }
                    for row in rows
                ],
            )
            return result.to_dict()

        face = self.get_metrics_summary(db, "face")
        voice = self.get_metrics_summary(db, "voice")
        merged_rows: list[dict[str, Any]] = []
        for sub in ("face", "voice"):
            run_tag, rows = self._rows_for_latest_run(db, sub)
            merged_rows.extend(
                {
                    "actual_label": row.actual_label,
                    "predicted_label": row.predicted_label,
                    "confidence_score": row.confidence_score,
                }
                for row in rows
            )
        overall = self._metrics_from_rows(
            input_type="all",
            dataset_name="FER2013+RAVDESS",
            run_tag="combined-latest",
            rows=merged_rows,
        ).to_dict()
        overall["by_input_type"] = {"face": face, "voice": voice}
        return overall


evaluation_service = ModelEvaluationService()
