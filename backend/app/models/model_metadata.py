from __future__ import annotations

from sqlalchemy import Column, DateTime, Enum, Float, Index, Integer, String, UniqueConstraint, func

from app.database import Base


class ModelMetadata(Base):
    __tablename__ = "model_metadata"
    __table_args__ = (
        UniqueConstraint("model_type", "model_version", name="uq_model_type_version"),
        Index("ix_model_metadata_type_trained", "model_type", "trained_at"),
    )

    id = Column(Integer, primary_key=True, index=True)
    model_type = Column(
        Enum("FER", "SER", "Fusion", name="model_type_enum", native_enum=False),
        nullable=False,
        index=True,
    )
    model_version = Column(String(64), nullable=False, index=True)

    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)

    trained_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
