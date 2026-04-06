from __future__ import annotations

from sqlalchemy import Column, DateTime, Enum, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.orm import relationship

from app.database import Base


class Feedback(Base):
    __tablename__ = "feedback"
    __table_args__ = (
        Index("ix_feedback_user_created", "user_id", "created_at"),
        Index("ix_feedback_song_action", "song_id", "action"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    song_id = Column(Integer, ForeignKey("recommended_songs.id", ondelete="CASCADE"), nullable=False, index=True)

    emotion_at_time = Column(String(32), nullable=False, index=True)
    comment = Column(Text, nullable=True)
    action = Column(
        Enum("like", "dislike", "skip", name="feedback_action", native_enum=False),
        nullable=False,
        index=True,
    )

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)

    user = relationship("User", back_populates="feedback_entries")
    song = relationship("RecommendedSong", back_populates="feedback_entries")
