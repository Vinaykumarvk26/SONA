from __future__ import annotations

from sqlalchemy import Column, DateTime, Float, ForeignKey, Index, Integer, String, func
from sqlalchemy.orm import relationship

from app.database import Base


class RecommendedSong(Base):
    __tablename__ = "recommended_songs"
    __table_args__ = (
        Index("ix_recommended_songs_record_created", "emotion_record_id", "created_at"),
        Index("ix_recommended_songs_artist_title", "artist_name", "song_title"),
    )

    id = Column(Integer, primary_key=True, index=True)
    emotion_record_id = Column(Integer, ForeignKey("emotion_records.id", ondelete="CASCADE"), nullable=False, index=True)

    song_title = Column(String(255), nullable=False, index=True)
    artist_name = Column(String(255), nullable=False, index=True)
    spotify_url = Column(String(1024), nullable=True)
    valence_score = Column(Float, nullable=True)
    energy_score = Column(Float, nullable=True)
    tempo = Column(Float, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)

    emotion_record = relationship("EmotionRecord", back_populates="recommended_songs")
    feedback_entries = relationship(
        "Feedback",
        back_populates="song",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
