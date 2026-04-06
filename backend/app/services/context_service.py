from datetime import datetime
from typing import List

from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.models.emotion_record import EmotionRecord
from app.models.feedback import Feedback
from app.services.user_lookup_service import get_or_create_user, resolve_user


def get_time_of_day_feature(ts: datetime | None = None) -> float:
    ts = ts or datetime.utcnow()
    return (ts.hour * 3600 + ts.minute * 60 + ts.second) / 86400.0


def _action_to_value(action: str) -> float:
    if action == "like":
        return 1.0
    if action == "dislike":
        return -1.0
    if action == "skip":
        return -0.8
    return 0.0


def load_user_history_vector(db: Session, user_identifier: str, size: int = 20) -> List[float]:
    user = resolve_user(db, user_identifier)
    if user is None:
        return [0.0] * size

    rows = (
        db.query(Feedback)
        .filter(Feedback.user_id == user.id)
        .order_by(desc(Feedback.created_at))
        .limit(size)
        .all()
    )

    values = [_action_to_value(row.action) for row in rows]
    if len(values) < size:
        values.extend([0.0] * (size - len(values)))
    return values[:size]


def get_recent_skip_rate(db: Session, user_identifier: str, lookback: int = 50) -> float:
    user = resolve_user(db, user_identifier)
    if user is None:
        return 0.0

    rows = (
        db.query(Feedback)
        .filter(Feedback.user_id == user.id)
        .order_by(desc(Feedback.created_at))
        .limit(lookback)
        .all()
    )
    if not rows:
        return 0.0

    skips = sum(1 for row in rows if row.action == "skip")
    return skips / len(rows)


def save_emotion_timeline(
    db: Session,
    user_identifier: str,
    face_emotion: str,
    face_confidence: float,
    voice_emotion: str,
    voice_confidence: float,
    fused_emotion: str,
    fused_confidence: float,
) -> EmotionRecord:
    user = get_or_create_user(db, user_identifier)

    row = EmotionRecord(
        user_id=user.id,
        face_emotion=face_emotion,
        face_confidence=face_confidence,
        voice_emotion=voice_emotion,
        voice_confidence=voice_confidence,
        fused_emotion=fused_emotion,
        fused_confidence=fused_confidence,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def get_emotion_timeline(db: Session, user_identifier: str, limit: int = 200) -> List[EmotionRecord]:
    user = resolve_user(db, user_identifier)
    if user is None:
        return []

    rows = (
        db.query(EmotionRecord)
        .filter(EmotionRecord.user_id == user.id)
        .order_by(desc(EmotionRecord.created_at))
        .limit(limit)
        .all()
    )
    return list(reversed(rows))


def get_latest_emotion_record(db: Session, user_identifier: str) -> EmotionRecord | None:
    user = resolve_user(db, user_identifier)
    if user is None:
        return None

    return (
        db.query(EmotionRecord)
        .filter(EmotionRecord.user_id == user.id)
        .order_by(desc(EmotionRecord.created_at))
        .first()
    )
