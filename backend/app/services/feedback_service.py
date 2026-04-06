from datetime import datetime, timedelta
from typing import Dict

from sqlalchemy.orm import Session

from app.models.feedback import Feedback
from app.services.user_lookup_service import get_or_create_user, resolve_user


EMOTION_BASE = {
    "happy": {"target_tempo": 128, "target_valence": 0.85, "target_energy": 0.8},
    # SONA uses a light regulation strategy instead of mirroring low moods too literally.
    "sad": {"target_tempo": 102, "target_valence": 0.6, "target_energy": 0.52},
    # Angry is intentionally mapped to calmer, more grounding recommendations.
    "angry": {"target_tempo": 76, "target_valence": 0.38, "target_energy": 0.24},
    "fear": {"target_tempo": 80, "target_valence": 0.4, "target_energy": 0.28},
    "disgust": {"target_tempo": 88, "target_valence": 0.42, "target_energy": 0.34},
    "surprise": {"target_tempo": 120, "target_valence": 0.72, "target_energy": 0.68},
    "neutral": {"target_tempo": 104, "target_valence": 0.5, "target_energy": 0.45},
}


class FeedbackBandit:
    def reward_from_event(self, event_type: str, value: float) -> float:
        event_type = (event_type or "").lower()
        if event_type == "like":
            return 1.0
        if event_type == "dislike":
            return -1.0
        if event_type == "skip":
            return -0.8 - min(1.0, max(0.0, value))
        return 0.0

    def update(self, db: Session, user_id: str, emotion: str, reward: float) -> None:
        # Adaptation is computed online from feedback history; no separate state table required.
        _ = (db, user_id, emotion, reward)

    def personalized_features(self, db: Session, user_id: str, emotion: str) -> Dict[str, float]:
        emotion = (emotion or "neutral").strip().lower()
        base = dict(EMOTION_BASE.get(emotion, EMOTION_BASE["neutral"]))

        user = resolve_user(db, user_id)
        if user is None:
            user = get_or_create_user(db, user_id)

        rows = (
            db.query(Feedback)
            .filter(Feedback.user_id == user.id, Feedback.emotion_at_time == emotion)
            .order_by(Feedback.created_at.desc())
            .limit(200)
            .all()
        )
        if rows:
            rewards = [self.reward_from_event(row.action, 1.0) for row in rows]
            reward_mean = sum(rewards) / len(rewards)
            drift = max(-0.15, min(0.15, reward_mean * 0.12))

            base["target_valence"] = float(min(1.0, max(0.0, base["target_valence"] + drift)))
            base["target_energy"] = float(min(1.0, max(0.0, base["target_energy"] + drift)))
            base["target_tempo"] = float(min(180.0, max(60.0, base["target_tempo"] * (1.0 + drift))))

        recent = (
            db.query(Feedback)
            .filter(Feedback.user_id == user.id, Feedback.created_at >= datetime.utcnow() - timedelta(days=7))
            .all()
        )
        if recent:
            skip_rate = sum(1 for row in recent if row.action == "skip") / len(recent)
            base["target_energy"] = float(max(0.1, base["target_energy"] * (1.0 - 0.25 * skip_rate)))

        return base
