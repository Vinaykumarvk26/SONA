from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pymongo import ASCENDING, DESCENDING

from app.config import get_settings
from app.mongo import get_mongo_db

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _text(value: Any, fallback: str = "") -> str:
    text = str(value or "").strip()
    return text or fallback


class MongoAppService:
    USERS = "users"
    AUTH_SESSIONS = "auth_sessions"
    USER_PREFERENCES = "user_preferences"
    USER_PROFILES = "user_profiles"
    AUDIT_LOGS = "audit_logs"
    MODEL_METRICS = "model_metrics"

    def __init__(self) -> None:
        self.db = get_mongo_db()
        self.settings = get_settings()
        self._indexes_ready = False
        self._runtime_disabled = False
        self._disable_reason = ""

    @property
    def enabled(self) -> bool:
        return self.db is not None and not self._runtime_disabled

    def _disable_runtime(self, where: str, exc: Exception) -> None:
        if not self._runtime_disabled:
            logger.warning("Mongo disabled at %s due to error: %s", where, exc)
        self._runtime_disabled = True
        self._disable_reason = str(exc)

    def ensure_indexes(self) -> bool:
        if not self.enabled or self._indexes_ready:
            return self._indexes_ready

        try:
            self.db[self.USERS].create_index([("email", ASCENDING)], unique=True)
            self.db[self.USERS].create_index([("username", ASCENDING)], unique=True)
            self.db[self.USERS].create_index([("sql_user_id", ASCENDING)], unique=True)

            self.db[self.AUTH_SESSIONS].create_index([("token", ASCENDING)], unique=True)
            self.db[self.AUTH_SESSIONS].create_index([("sql_user_id", ASCENDING), ("expires_at", DESCENDING)])

            self.db[self.USER_PREFERENCES].create_index([("sql_user_id", ASCENDING)], unique=True)
            self.db[self.USER_PREFERENCES].create_index([("email", ASCENDING)])
            self.db[self.USER_PROFILES].create_index([("sql_user_id", ASCENDING)], unique=True)
            self.db[self.USER_PROFILES].create_index([("email", ASCENDING)])

            self.db[self.AUDIT_LOGS].create_index([("sql_user_id", ASCENDING), ("created_at", DESCENDING)])
            self.db[self.AUDIT_LOGS].create_index([("action", ASCENDING), ("created_at", DESCENDING)])

            self.db[self.MODEL_METRICS].create_index([("model_name", ASCENDING)], unique=True)
            self._indexes_ready = True
            return True
        except Exception as exc:
            self._disable_runtime("ensure_indexes", exc)
            return False

    def _mongo_ready(self) -> bool:
        if not self.enabled:
            return False
        return self.ensure_indexes()

    def upsert_user(self, *, user: Any, provider: str = "password") -> None:
        if not self._mongo_ready():
            return
        try:
            now = _utcnow()
            self.db[self.USERS].update_one(
                {"sql_user_id": int(user.id)},
                {
                    "$set": {
                        "username": _text(user.username).lower(),
                        "email": _text(user.email).lower(),
                        "provider": provider,
                        "updated_at": now,
                    },
                    "$setOnInsert": {
                        "sql_user_id": int(user.id),
                        "created_at": getattr(user, "created_at", None) or now,
                    },
                },
                upsert=True,
            )
            self.db[self.USER_PROFILES].update_one(
                {"sql_user_id": int(user.id)},
                {
                    "$set": {
                        "username": _text(user.username).lower(),
                        "email": _text(user.email).lower(),
                        "updated_at": now,
                    },
                    "$setOnInsert": {
                        "sql_user_id": int(user.id),
                        "full_name": "",
                        "phone": "",
                        "location": "",
                        "bio": "",
                        "created_at": now,
                    },
                },
                upsert=True,
            )
        except Exception as exc:
            self._disable_runtime("upsert_user", exc)

    def create_session(self, *, user: Any, token: str, expires_at: datetime) -> None:
        if not self._mongo_ready():
            return
        try:
            self.db[self.AUTH_SESSIONS].update_one(
                {"token": token},
                {
                    "$set": {
                        "token": token,
                        "sql_user_id": int(user.id),
                        "username": _text(user.username).lower(),
                        "email": _text(user.email).lower(),
                        "expires_at": expires_at,
                        "updated_at": _utcnow(),
                    },
                    "$setOnInsert": {
                        "created_at": _utcnow(),
                    },
                },
                upsert=True,
            )
        except Exception as exc:
            self._disable_runtime("create_session", exc)

    def touch_session(self, token: str) -> None:
        if not self._mongo_ready() or not token:
            return
        try:
            self.db[self.AUTH_SESSIONS].update_one(
                {"token": token},
                {"$set": {"last_validated_at": _utcnow()}},
            )
        except Exception as exc:
            self._disable_runtime("touch_session", exc)

    def revoke_user_sessions(self, sql_user_id: int) -> None:
        if not self._mongo_ready():
            return
        try:
            self.db[self.AUTH_SESSIONS].delete_many({"sql_user_id": int(sql_user_id)})
        except Exception as exc:
            self._disable_runtime("revoke_user_sessions", exc)

    def log_audit(self, *, action: str, user: Any | None = None, metadata: dict[str, Any] | None = None) -> None:
        if not self._mongo_ready():
            return
        try:
            payload = {
                "action": _text(action),
                "metadata": metadata or {},
                "created_at": _utcnow(),
            }
            if user is not None:
                payload.update(
                    {
                        "sql_user_id": int(user.id),
                        "username": _text(user.username).lower(),
                        "email": _text(user.email).lower(),
                    }
                )
            self.db[self.AUDIT_LOGS].insert_one(payload)
        except Exception as exc:
            self._disable_runtime("log_audit", exc)

    def get_preferences(self, *, user: Any) -> dict[str, Any]:
        if not self._mongo_ready():
            return {}
        try:
            doc = self.db[self.USER_PREFERENCES].find_one({"sql_user_id": int(user.id)}, {"_id": 0})
            return doc or {}
        except Exception as exc:
            self._disable_runtime("get_preferences", exc)
            return {}

    def get_user_profile(self, *, user: Any) -> dict[str, Any]:
        if not self._mongo_ready():
            return {
                "full_name": "",
                "phone": "",
                "location": "",
                "bio": "",
            }
        try:
            doc = self.db[self.USER_PROFILES].find_one({"sql_user_id": int(user.id)}, {"_id": 0})
            if not doc:
                return {
                    "full_name": "",
                    "phone": "",
                    "location": "",
                    "bio": "",
                }
            return {
                "full_name": _text(doc.get("full_name")),
                "phone": _text(doc.get("phone")),
                "location": _text(doc.get("location")),
                "bio": _text(doc.get("bio")),
            }
        except Exception as exc:
            self._disable_runtime("get_user_profile", exc)
            return {
                "full_name": "",
                "phone": "",
                "location": "",
                "bio": "",
            }

    def save_user_profile(
        self,
        *,
        user: Any,
        full_name: str | None = None,
        phone: str | None = None,
        location: str | None = None,
        bio: str | None = None,
    ) -> dict[str, Any]:
        incoming = {
            "full_name": full_name,
            "phone": phone,
            "location": location,
            "bio": bio,
        }
        if not self._mongo_ready():
            return {
                "full_name": _text(incoming["full_name"]),
                "phone": _text(incoming["phone"]),
                "location": _text(incoming["location"]),
                "bio": _text(incoming["bio"]),
            }

        try:
            now = _utcnow()
            current = self.db[self.USER_PROFILES].find_one({"sql_user_id": int(user.id)}, {"_id": 0}) or {}
            merged = {
                "full_name": _text(current.get("full_name")),
                "phone": _text(current.get("phone")),
                "location": _text(current.get("location")),
                "bio": _text(current.get("bio")),
            }
            for key, value in incoming.items():
                if value is not None:
                    merged[key] = _text(value)

            payload = {
                "sql_user_id": int(user.id),
                "username": _text(user.username).lower(),
                "email": _text(user.email).lower(),
                **merged,
                "updated_at": now,
            }
            self.db[self.USER_PROFILES].update_one(
                {"sql_user_id": int(user.id)},
                {"$set": payload, "$setOnInsert": {"created_at": now}},
                upsert=True,
            )
            return merged
        except Exception as exc:
            self._disable_runtime("save_user_profile", exc)
            return {
                "full_name": _text(incoming["full_name"]),
                "phone": _text(incoming["phone"]),
                "location": _text(incoming["location"]),
                "bio": _text(incoming["bio"]),
            }

    def save_preferences(
        self,
        *,
        user: Any,
        playback_mode: str,
        music_source: str,
        languages: list[str],
        theme: str,
    ) -> dict[str, Any]:
        if not self._mongo_ready():
            return {
                "playback_mode": playback_mode,
                "music_source": music_source,
                "languages": languages,
                "theme": theme,
            }
        try:
            payload = {
                "sql_user_id": int(user.id),
                "username": _text(user.username).lower(),
                "email": _text(user.email).lower(),
                "playback_mode": playback_mode,
                "music_source": music_source,
                "languages": languages,
                "theme": theme,
                "updated_at": _utcnow(),
            }
            self.db[self.USER_PREFERENCES].update_one(
                {"sql_user_id": int(user.id)},
                {"$set": payload, "$setOnInsert": {"created_at": _utcnow()}},
                upsert=True,
            )
            return payload
        except Exception as exc:
            self._disable_runtime("save_preferences", exc)
            return {
                "playback_mode": playback_mode,
                "music_source": music_source,
                "languages": languages,
                "theme": theme,
            }

    def sync_model_metrics(self) -> None:
        if not self._mongo_ready():
            return
        try:
            mapping = {
                "facial_emotion": self.settings.fer_model_path,
                "speech_emotion": self.settings.speech_model_path,
                "fusion_emotion": self.settings.fusion_model_path,
            }

            for model_name, checkpoint_path in mapping.items():
                metrics = self._load_metrics_for_checkpoint(checkpoint_path)
                self.db[self.MODEL_METRICS].update_one(
                    {"model_name": model_name},
                    {
                        "$set": {
                            "model_name": model_name,
                            "checkpoint_path": checkpoint_path,
                            "metrics": metrics,
                            "updated_at": _utcnow(),
                        },
                        "$setOnInsert": {"created_at": _utcnow()},
                    },
                    upsert=True,
                )
        except Exception as exc:
            self._disable_runtime("sync_model_metrics", exc)

    def _load_metrics_for_checkpoint(self, checkpoint_path: str) -> dict[str, Any]:
        checkpoint_name = Path(checkpoint_path).name
        mapping = {
            "fer_best.pt": "fer_test_metrics.json",
            "ser_best.pt": "ser_test_metrics.json",
            "ser_mel_bilstm_best.pt": "ser_mel_bilstm_test_metrics.json",
            "fusion_best.pt": "",
        }
        metrics_name = mapping.get(checkpoint_name, "")
        if not metrics_name:
            return {}

        candidates = [
            Path("backend/checkpoints") / metrics_name,
            Path("checkpoints") / metrics_name,
        ]
        for candidate in candidates:
            if candidate.exists():
                try:
                    return json.loads(candidate.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    return {}
        return {}


mongo_app = MongoAppService()
