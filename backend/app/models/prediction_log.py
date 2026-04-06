from __future__ import annotations

from sqlalchemy import Column, DateTime, Enum, Float, Index, Integer, String, func

from app.database import Base


class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    __table_args__ = (
        Index("ix_prediction_logs_input_run", "input_type", "run_tag"),
        Index("ix_prediction_logs_dataset_run", "dataset_name", "run_tag"),
        Index("ix_prediction_logs_timestamp", "timestamp"),
    )

    id = Column(Integer, primary_key=True, index=True)
    input_type = Column(
        Enum("face", "voice", name="prediction_input_type", native_enum=False),
        nullable=False,
        index=True,
    )
    dataset_name = Column(String(64), nullable=False, default="")
    actual_label = Column(String(32), nullable=False, index=True)
    predicted_label = Column(String(32), nullable=False, index=True)
    confidence_score = Column(Float, nullable=False, default=0.0)
    run_tag = Column(String(64), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
