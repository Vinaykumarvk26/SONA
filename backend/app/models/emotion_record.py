from __future__ import annotations

from sqlalchemy import Column, DateTime, Float, ForeignKey, Index, Integer, String, func
from sqlalchemy.orm import relationship

from app.database import Base


class EmotionRecord(Base):
    __tablename__ = "emotion_records"
    __table_args__ = (
        Index("ix_emotion_records_user_created", "user_id", "created_at"),
        Index("ix_emotion_records_fused_created", "fused_emotion", "created_at"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    face_emotion = Column(String(32), nullable=False, index=True)
    face_confidence = Column(Float, nullable=False)
    voice_emotion = Column(String(32), nullable=False, index=True)
    voice_confidence = Column(Float, nullable=False)
    fused_emotion = Column(String(32), nullable=False, index=True)
    fused_confidence = Column(Float, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)

    user = relationship("User", back_populates="emotion_records")
    recommended_songs = relationship(
        "RecommendedSong",
        back_populates="emotion_record",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
