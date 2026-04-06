from __future__ import annotations

from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, String, Text, Float, func
from sqlalchemy.orm import relationship

from app.database import Base


class AnalyticsEvent(Base):
    __tablename__ = "analytics_events"
    __table_args__ = (
        Index("ix_analytics_events_user_category_created", "user_id", "category", "created_at"),
        Index("ix_analytics_events_user_action_created", "user_id", "action", "created_at"),
        Index("ix_analytics_events_user_emotion_created", "user_id", "emotion", "created_at"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    category = Column(String(32), nullable=False, index=True)
    action = Column(String(64), nullable=False, index=True)
    emotion = Column(String(32), nullable=True, index=True)
    confidence = Column(Float, nullable=True)
    track_id = Column(String(255), nullable=True)
    track_name = Column(String(255), nullable=True)
    artist = Column(String(255), nullable=True)
    source = Column(String(64), nullable=True)
    transcript = Column(Text, nullable=True)
    metadata_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)

    user = relationship("User")
