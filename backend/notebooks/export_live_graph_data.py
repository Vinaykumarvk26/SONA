# %% [markdown]
# # SONA Live Graph Data Export
# Exports live emotion events (predicted labels + confidence) from analytics logs.
# This is NOT ground-truth. It is for trend/distribution graphs.

# %%
import sys
from pathlib import Path
import argparse

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import numpy as np
import pandas as pd

from app.database import SessionLocal, init_db
from app.models.analytics_event import AnalyticsEvent
from app.models.prediction_log import PredictionLog
from app.services.user_lookup_service import resolve_user
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
RESULTS_DIR = ROOT_DIR / "notebooks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def export_live(user_identifier: str) -> Path:
    try:
        init_db()
    except OperationalError as exc:
        if "already exists" not in str(exc):
            raise

    with SessionLocal() as db:
        user = resolve_user(db, user_identifier)
        if user is None:
            raise RuntimeError(f"User not found for identifier: {user_identifier}")

        rows = db.execute(
            select(AnalyticsEvent)
            .where(AnalyticsEvent.user_id == user.id, AnalyticsEvent.category == "emotion")
            .order_by(AnalyticsEvent.created_at.asc(), AnalyticsEvent.id.asc())
        ).scalars().all()

    if not rows:
        raise RuntimeError("No live emotion events found for this user yet.")

    data = {
        "timestamp": [r.created_at for r in rows],
        "channel": [r.action for r in rows],
        "emotion": [r.emotion for r in rows],
        "confidence": [r.confidence for r in rows],
        "transcript": [r.transcript for r in rows],
    }
    df = pd.DataFrame(data)

    np.save(RESULTS_DIR / "live_emotion_labels.npy", df["emotion"].values)
    np.save(RESULTS_DIR / "live_emotion_confidence.npy", df["confidence"].values)
    df.to_csv(RESULTS_DIR / "live_emotion_events.csv", index=False)

    # Summary exports for more meaningful graphs (no ground-truth required).
    emotion_counts = (
        df["emotion"]
        .fillna("unknown")
        .value_counts()
        .rename_axis("emotion")
        .reset_index(name="count")
    )
    emotion_counts["percent"] = (emotion_counts["count"] / emotion_counts["count"].sum()) * 100.0
    emotion_counts.to_csv(RESULTS_DIR / "live_emotion_distribution.csv", index=False)

    confidence_stats = (
        df.dropna(subset=["emotion", "confidence"])
        .groupby("emotion")["confidence"]
        .agg(count="count", mean="mean", median="median", min="min", max="max")
        .reset_index()
    )
    confidence_stats.to_csv(RESULTS_DIR / "live_emotion_confidence_stats.csv", index=False)

    channel_counts = (
        df["channel"]
        .fillna("unknown")
        .value_counts()
        .rename_axis("channel")
        .reset_index(name="count")
    )
    channel_counts["percent"] = (channel_counts["count"] / channel_counts["count"].sum()) * 100.0
    channel_counts.to_csv(RESULTS_DIR / "live_channel_distribution.csv", index=False)

    emotion_by_channel = (
        df.pivot_table(index="emotion", columns="channel", values="confidence", aggfunc="count", fill_value=0)
        .reset_index()
    )
    emotion_by_channel.to_csv(RESULTS_DIR / "live_emotion_by_channel.csv", index=False)

    return RESULTS_DIR


def _latest_tag(db, input_type: str):
    row = db.execute(
        select(PredictionLog.run_tag)
        .where(PredictionLog.input_type == input_type)
        .order_by(PredictionLog.timestamp.desc(), PredictionLog.id.desc())
        .limit(1)
    ).scalar_one_or_none()
    return row


def _load_rows(db, input_type: str, run_tag: str):
    rows = db.execute(
        select(PredictionLog)
        .where(PredictionLog.input_type == input_type, PredictionLog.run_tag == run_tag)
    ).scalars().all()
    return rows


def export_evaluation() -> Path:
    try:
        init_db()
    except OperationalError as exc:
        if "already exists" not in str(exc):
            raise

    with SessionLocal() as db:
        face_tag = _latest_tag(db, "face")
        voice_tag = _latest_tag(db, "voice")

        face_rows = _load_rows(db, "face", face_tag) if face_tag else []
        voice_rows = _load_rows(db, "voice", voice_tag) if voice_tag else []

    face_true = [r.actual_label for r in face_rows]
    face_pred = [r.predicted_label for r in face_rows]
    voice_true = [r.actual_label for r in voice_rows]
    voice_pred = [r.predicted_label for r in voice_rows]

    all_true = face_true + voice_true
    all_pred = face_pred + voice_pred

    labels = ["happy", "sad", "angry", "fear", "disgust", "surprise", "neutral"]

    if not all_true:
        raise RuntimeError("No evaluation prediction logs found. Run evaluation first.")

    metrics = {
        "accuracy": float(accuracy_score(all_true, all_pred)),
        "precision": float(precision_score(all_true, all_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(all_true, all_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(all_true, all_pred, average="weighted", zero_division=0)),
    }

    with open(RESULTS_DIR / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    cm = confusion_matrix(all_true, all_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(RESULTS_DIR / "evaluation_confusion_matrix.csv")

    face_metrics = {
        "Accuracy": accuracy_score(face_true, face_pred) if face_true else 0.0,
        "Precision": precision_score(face_true, face_pred, average="weighted", zero_division=0) if face_true else 0.0,
        "Recall": recall_score(face_true, face_pred, average="weighted", zero_division=0) if face_true else 0.0,
        "F1": f1_score(face_true, face_pred, average="weighted", zero_division=0) if face_true else 0.0,
    }
    voice_metrics = {
        "Accuracy": accuracy_score(voice_true, voice_pred) if voice_true else 0.0,
        "Precision": precision_score(voice_true, voice_pred, average="weighted", zero_division=0) if voice_true else 0.0,
        "Recall": recall_score(voice_true, voice_pred, average="weighted", zero_division=0) if voice_true else 0.0,
        "F1": f1_score(voice_true, voice_pred, average="weighted", zero_division=0) if voice_true else 0.0,
    }
    face_voice_df = pd.DataFrame(
        {"metric": list(face_metrics.keys()), "face": list(face_metrics.values()), "voice": list(voice_metrics.values())}
    )
    face_voice_df.to_csv(RESULTS_DIR / "evaluation_face_vs_voice.csv", index=False)

    emotion_acc = []
    for emo in labels:
        idx = [i for i, y in enumerate(all_true) if y == emo]
        if not idx:
            emotion_acc.append({"emotion": emo, "accuracy": 0.0, "count": 0})
            continue
        y_t = [all_true[i] for i in idx]
        y_p = [all_pred[i] for i in idx]
        emotion_acc.append(
            {"emotion": emo, "accuracy": float(accuracy_score(y_t, y_p)), "count": len(idx)}
        )
    pd.DataFrame(emotion_acc).to_csv(RESULTS_DIR / "evaluation_emotion_accuracy.csv", index=False)

    return RESULTS_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="default-user", help="username or email used in the app")
    parser.add_argument(
        "--evaluation",
        action="store_true",
        help="Export evaluation metrics from prediction_logs (requires evaluation runs).",
    )
    args = parser.parse_args()

    if args.evaluation:
        out_dir = export_evaluation()
    else:
        out_dir = export_live(args.user)
    print("Saved live export in:", out_dir)
