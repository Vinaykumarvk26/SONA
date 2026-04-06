from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from pymongo import ASCENDING, DESCENDING
from sqlalchemy import func

from app.config import get_settings
from app.database import SessionLocal
from app.models.analytics_event import AnalyticsEvent
from app.models.emotion_record import EmotionRecord
from app.models.recommended_song import RecommendedSong
from app.mongo import get_mongo_db
from app.services.user_lookup_service import get_or_create_user, resolve_user


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_text(value: Any, fallback: str = "") -> str:
    text = str(value or "").strip()
    return text or fallback


def _json_text(value: dict[str, Any] | None) -> str:
    if not value:
        return "{}"
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return "{}"


class AnalyticsService:
    EMOTION_EVENTS = "emotion_events"
    RECOMMENDATION_EVENTS = "recommendation_events"
    PLAYBACK_EVENTS = "playback_events"
    FEEDBACK_EVENTS = "feedback_events"
    UI_EVENTS = "ui_events"

    def __init__(self) -> None:
        self.db = get_mongo_db()
        self.settings = get_settings()
        self._indexes_ready = False

    @property
    def enabled(self) -> bool:
        return self.db is not None

    def ensure_indexes(self) -> None:
        if not self.enabled or self._indexes_ready:
            return

        self.db[self.EMOTION_EVENTS].create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
        self.db[self.EMOTION_EVENTS].create_index([("channel", ASCENDING), ("created_at", DESCENDING)])
        self.db[self.EMOTION_EVENTS].create_index([("label", ASCENDING), ("created_at", DESCENDING)])

        self.db[self.RECOMMENDATION_EVENTS].create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
        self.db[self.PLAYBACK_EVENTS].create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
        self.db[self.PLAYBACK_EVENTS].create_index([("action", ASCENDING), ("created_at", DESCENDING)])
        self.db[self.FEEDBACK_EVENTS].create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
        self.db[self.FEEDBACK_EVENTS].create_index([("action", ASCENDING), ("created_at", DESCENDING)])
        self.db[self.UI_EVENTS].create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
        self.db[self.UI_EVENTS].create_index([("action", ASCENDING), ("created_at", DESCENDING)])
        self._indexes_ready = True

    def _insert_mongo(self, collection_name: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            raise RuntimeError("Mongo analytics is unavailable")
        self.ensure_indexes()
        document = {
            **payload,
            "created_at": payload.get("created_at") or _utcnow(),
        }
        self.db[collection_name].insert_one(document)

    def _insert_sql_event(
        self,
        *,
        user_identifier: str,
        category: str,
        action: str,
        emotion: str | None = None,
        confidence: float | None = None,
        track_id: str | None = None,
        track_name: str | None = None,
        artist: str | None = None,
        source: str | None = None,
        transcript: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with SessionLocal() as db:
            user = get_or_create_user(db, user_identifier)
            row = AnalyticsEvent(
                user_id=user.id,
                category=_safe_text(category, "unknown"),
                action=_safe_text(action, "unknown"),
                emotion=_safe_text(emotion).lower() or None,
                confidence=_safe_float(confidence) if confidence is not None else None,
                track_id=_safe_text(track_id) or None,
                track_name=_safe_text(track_name) or None,
                artist=_safe_text(artist) or None,
                source=_safe_text(source) or None,
                transcript=_safe_text(transcript) or None,
                metadata_json=_json_text(metadata),
            )
            db.add(row)
            db.commit()

    def _write_event(
        self,
        *,
        mongo_collection: str,
        mongo_payload: dict[str, Any],
        user_identifier: str,
        category: str,
        action: str,
        emotion: str | None = None,
        confidence: float | None = None,
        track_id: str | None = None,
        track_name: str | None = None,
        artist: str | None = None,
        source: str | None = None,
        transcript: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        try:
            self._insert_mongo(mongo_collection, mongo_payload)
            return
        except Exception:
            pass

        self._insert_sql_event(
            user_identifier=user_identifier,
            category=category,
            action=action,
            emotion=emotion,
            confidence=confidence,
            track_id=track_id,
            track_name=track_name,
            artist=artist,
            source=source,
            transcript=transcript,
            metadata=metadata,
        )

    def log_emotion_event(
        self,
        *,
        user_id: str,
        channel: str,
        label: str,
        confidence: float,
        transcript: str | None = None,
        source_weights: dict[str, float] | None = None,
    ) -> None:
        payload = {
            "user_id": _safe_text(user_id, "default-user"),
            "channel": _safe_text(channel, "unknown"),
            "label": _safe_text(label, "neutral").lower(),
            "confidence": _safe_float(confidence),
            "transcript": _safe_text(transcript),
            "source_weights": source_weights or {},
        }
        self._write_event(
            mongo_collection=self.EMOTION_EVENTS,
            mongo_payload=payload,
            user_identifier=user_id,
            category="emotion",
            action=channel,
            emotion=label,
            confidence=confidence,
            transcript=transcript,
            metadata={"source_weights": source_weights or {}},
        )

    def log_recommendation_event(
        self,
        *,
        user_id: str,
        emotion: str,
        confidence: float,
        tracks: list[dict[str, Any]],
        target_features: dict[str, Any] | None = None,
    ) -> None:
        compact_tracks = [
            {
                "track_id": _safe_text(track.get("id")),
                "name": _safe_text(track.get("name"), "Unknown"),
                "artist": _safe_text(track.get("artist"), "Unknown"),
                "preview_url": _safe_text(track.get("preview_url")),
                "external_url": _safe_text(track.get("external_url")),
                "image_url": _safe_text(track.get("image_url")),
            }
            for track in tracks[:10]
        ]
        payload = {
            "user_id": _safe_text(user_id, "default-user"),
            "emotion": _safe_text(emotion, "neutral").lower(),
            "confidence": _safe_float(confidence),
            "target_features": target_features or {},
            "track_count": len(compact_tracks),
            "tracks": compact_tracks,
        }
        first_track = compact_tracks[0] if compact_tracks else {}
        self._write_event(
            mongo_collection=self.RECOMMENDATION_EVENTS,
            mongo_payload=payload,
            user_identifier=user_id,
            category="recommendation",
            action="playlist_generated",
            emotion=emotion,
            confidence=confidence,
            track_id=first_track.get("track_id"),
            track_name=first_track.get("name"),
            artist=first_track.get("artist"),
            metadata={"track_count": len(compact_tracks), "tracks": compact_tracks, "target_features": target_features or {}},
        )

    def log_playback_event(
        self,
        *,
        user_id: str,
        action: str,
        track_id: str | None = None,
        track_name: str | None = None,
        artist: str | None = None,
        emotion: str | None = None,
        source: str | None = None,
    ) -> None:
        payload = {
            "user_id": _safe_text(user_id, "default-user"),
            "action": _safe_text(action, "play"),
            "track_id": _safe_text(track_id),
            "track_name": _safe_text(track_name, "Unknown"),
            "artist": _safe_text(artist, "Unknown"),
            "emotion": _safe_text(emotion, "neutral").lower(),
            "source": _safe_text(source),
        }
        self._write_event(
            mongo_collection=self.PLAYBACK_EVENTS,
            mongo_payload=payload,
            user_identifier=user_id,
            category="playback",
            action=action,
            emotion=emotion,
            track_id=track_id,
            track_name=track_name,
            artist=artist,
            source=source,
        )

    def log_feedback_event(
        self,
        *,
        user_id: str,
        action: str,
        song_id: int | None = None,
        track_id: str | None = None,
        emotion: str | None = None,
        value: float = 0.0,
        comment: str | None = None,
    ) -> None:
        payload = {
            "user_id": _safe_text(user_id, "default-user"),
            "action": _safe_text(action),
            "song_id": song_id,
            "track_id": _safe_text(track_id),
            "emotion": _safe_text(emotion, "neutral").lower(),
            "value": _safe_float(value),
            "comment": _safe_text(comment),
        }
        self._write_event(
            mongo_collection=self.FEEDBACK_EVENTS,
            mongo_payload=payload,
            user_identifier=user_id,
            category="feedback",
            action=action,
            emotion=emotion,
            track_id=track_id or (str(song_id) if song_id is not None else None),
            confidence=value,
            metadata={"song_id": song_id, "comment": comment},
        )

    def log_ui_event(
        self,
        *,
        user_id: str,
        action: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "user_id": _safe_text(user_id, "default-user"),
            "action": _safe_text(action),
            "metadata": metadata or {},
        }
        self._write_event(
            mongo_collection=self.UI_EVENTS,
            mongo_payload=payload,
            user_identifier=user_id,
            category="ui",
            action=action,
            metadata=metadata or {},
        )

    def _metrics_json_for_checkpoint(self, checkpoint_path: str) -> dict[str, Any]:
        checkpoint_name = Path(checkpoint_path).name
        if checkpoint_name == "ser_mel_bilstm_best.pt":
            metrics_name = "ser_mel_bilstm_test_metrics.json"
        elif checkpoint_name.startswith("ser_"):
            metrics_name = "ser_test_metrics.json"
        elif checkpoint_name.startswith("fer_"):
            metrics_name = "fer_test_metrics.json"
        else:
            metrics_name = ""

        if not metrics_name:
            return {}

        metrics_path = Path("backend/checkpoints") / metrics_name
        if not metrics_path.exists():
            metrics_path = Path("checkpoints") / metrics_name
        if not metrics_path.exists():
            return {}

        try:
            return json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _empty_metrics_payload(self) -> dict[str, Any]:
        return {
            "mongo_enabled": False,
            "live": {
                "emotions_detected": 0,
                "songs_played": 0,
                "retakes": 0,
                "recommendations_generated": 0,
                "top_emotions": [],
                "timeline": [],
                "average_confidence": 0.0,
            },
            "facial_model": self._metrics_json_for_checkpoint(self.settings.fer_model_path),
            "speech_model": self._metrics_json_for_checkpoint(self.settings.speech_model_path),
            "overall_model": {},
        }

    def _mongo_metrics_overview(self, user_id: str) -> dict[str, Any]:
        self.ensure_indexes()
        user_filter = {"user_id": _safe_text(user_id, "default-user")}

        emotion_coll = self.db[self.EMOTION_EVENTS]
        playback_coll = self.db[self.PLAYBACK_EVENTS]
        rec_coll = self.db[self.RECOMMENDATION_EVENTS]
        ui_coll = self.db[self.UI_EVENTS]

        emotion_count = emotion_coll.count_documents(user_filter)
        songs_played = playback_coll.count_documents({**user_filter, "action": "track_start"})
        retakes = ui_coll.count_documents({**user_filter, "action": {"$in": ["retake_face", "retake_voice"]}})
        recommendations_generated = rec_coll.count_documents(user_filter)

        timeline_docs = list(
            emotion_coll.find(user_filter, {"_id": 0})
            .sort("created_at", DESCENDING)
            .limit(30)
        )
        timeline_docs.reverse()
        timeline = [
            {
                "timestamp": row["created_at"].isoformat(),
                "emotion": row.get("label", "neutral"),
                "confidence": _safe_float(row.get("confidence")),
                "channel": row.get("channel", "unknown"),
            }
            for row in timeline_docs
        ]

        top_emotions_raw = list(
            emotion_coll.aggregate(
                [
                    {"$match": user_filter},
                    {"$group": {"_id": "$label", "count": {"$sum": 1}}},
                    {"$sort": {"count": -1}},
                    {"$limit": 5},
                ]
            )
        )
        top_emotions = [{"emotion": row["_id"], "count": int(row["count"])} for row in top_emotions_raw]
        recent_confidences = [_safe_float(row.get("confidence")) for row in timeline_docs]
        avg_conf = mean(recent_confidences) if recent_confidences else 0.0

        facial_model = self._metrics_json_for_checkpoint(self.settings.fer_model_path)
        speech_model = self._metrics_json_for_checkpoint(self.settings.speech_model_path)
        overall_model = {
            key: mean([entry[key] for entry in [facial_model, speech_model] if key in entry])
            if any(key in entry for entry in [facial_model, speech_model])
            else 0.0
            for key in ("accuracy", "precision", "recall", "f1_score")
        }

        return {
            "mongo_enabled": True,
            "live": {
                "emotions_detected": int(emotion_count),
                "songs_played": int(songs_played),
                "retakes": int(retakes),
                "recommendations_generated": int(recommendations_generated),
                "top_emotions": top_emotions,
                "timeline": timeline,
                "average_confidence": avg_conf,
            },
            "facial_model": facial_model,
            "speech_model": speech_model,
            "overall_model": overall_model,
        }

    def _sql_metrics_overview(self, user_id: str) -> dict[str, Any]:
        payload = self._empty_metrics_payload()
        with SessionLocal() as db:
            user = resolve_user(db, user_id)
            if user is None:
                return payload

            emotion_events_q = db.query(AnalyticsEvent).filter(
                AnalyticsEvent.user_id == user.id,
                AnalyticsEvent.category == "emotion",
            )
            playback_q = db.query(AnalyticsEvent).filter(
                AnalyticsEvent.user_id == user.id,
                AnalyticsEvent.category == "playback",
                AnalyticsEvent.action == "track_start",
            )
            retake_q = db.query(AnalyticsEvent).filter(
                AnalyticsEvent.user_id == user.id,
                AnalyticsEvent.category == "ui",
                AnalyticsEvent.action.in_(["retake_face", "retake_voice"]),
            )
            recommendation_q = db.query(AnalyticsEvent).filter(
                AnalyticsEvent.user_id == user.id,
                AnalyticsEvent.category == "recommendation",
                AnalyticsEvent.action == "playlist_generated",
            )

            emotion_count = emotion_events_q.count()
            songs_played = playback_q.count()
            retakes = retake_q.count()
            recommendations_generated = recommendation_q.count()

            timeline_rows = (
                emotion_events_q.order_by(AnalyticsEvent.created_at.desc()).limit(30).all()
            )
            timeline_rows.reverse()
            timeline = [
                {
                    "timestamp": row.created_at.isoformat() if row.created_at else _utcnow().isoformat(),
                    "emotion": row.emotion or "neutral",
                    "confidence": _safe_float(row.confidence),
                    "channel": row.action or "unknown",
                }
                for row in timeline_rows
            ]

            top_emotions_rows = (
                db.query(AnalyticsEvent.emotion, func.count(AnalyticsEvent.id))
                .filter(AnalyticsEvent.user_id == user.id, AnalyticsEvent.category == "emotion")
                .group_by(AnalyticsEvent.emotion)
                .order_by(func.count(AnalyticsEvent.id).desc())
                .limit(5)
                .all()
            )
            top_emotions = [
                {"emotion": emotion or "neutral", "count": int(count)}
                for emotion, count in top_emotions_rows
            ]

            if not timeline and emotion_count == 0:
                # Fall back to existing emotion history even if analytics events were not stored before.
                emotion_rows = (
                    db.query(EmotionRecord)
                    .filter(EmotionRecord.user_id == user.id)
                    .order_by(EmotionRecord.created_at.desc())
                    .limit(30)
                    .all()
                )
                emotion_rows.reverse()
                timeline = [
                    {
                        "timestamp": row.created_at.isoformat() if row.created_at else _utcnow().isoformat(),
                        "emotion": row.fused_emotion,
                        "confidence": _safe_float(row.fused_confidence),
                        "channel": "fusion",
                    }
                    for row in emotion_rows
                ]
                emotion_count = (
                    db.query(func.count(EmotionRecord.id))
                    .filter(EmotionRecord.user_id == user.id)
                    .scalar()
                    or 0
                )
                top_emotion_rows = (
                    db.query(EmotionRecord.fused_emotion, func.count(EmotionRecord.id))
                    .filter(EmotionRecord.user_id == user.id)
                    .group_by(EmotionRecord.fused_emotion)
                    .order_by(func.count(EmotionRecord.id).desc())
                    .limit(5)
                    .all()
                )
                top_emotions = [
                    {"emotion": emotion or "neutral", "count": int(count)}
                    for emotion, count in top_emotion_rows
                ]
                if recommendations_generated == 0:
                    recommendations_generated = (
                        db.query(func.count(func.distinct(RecommendedSong.emotion_record_id)))
                        .join(EmotionRecord, RecommendedSong.emotion_record_id == EmotionRecord.id)
                        .filter(EmotionRecord.user_id == user.id)
                        .scalar()
                        or 0
                    )

            recent_confidences = [row["confidence"] for row in timeline]
            avg_conf = mean(recent_confidences) if recent_confidences else 0.0

            facial_model = self._metrics_json_for_checkpoint(self.settings.fer_model_path)
            speech_model = self._metrics_json_for_checkpoint(self.settings.speech_model_path)
            overall_model = {
                key: mean([entry[key] for entry in [facial_model, speech_model] if key in entry])
                if any(key in entry for entry in [facial_model, speech_model])
                else 0.0
                for key in ("accuracy", "precision", "recall", "f1_score")
            }

            return {
                "mongo_enabled": False,
                "live": {
                    "emotions_detected": int(emotion_count),
                    "songs_played": int(songs_played),
                    "retakes": int(retakes),
                    "recommendations_generated": int(recommendations_generated),
                    "top_emotions": top_emotions,
                    "timeline": timeline,
                    "average_confidence": avg_conf,
                },
                "facial_model": facial_model,
                "speech_model": speech_model,
                "overall_model": overall_model,
            }

    def get_metrics_overview(self, user_id: str) -> dict[str, Any]:
        if self.enabled:
            try:
                return self._mongo_metrics_overview(user_id)
            except Exception:
                pass
        return self._sql_metrics_overview(user_id)


analytics = AnalyticsService()
