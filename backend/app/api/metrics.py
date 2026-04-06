from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import EmotionFeedbackRequest, EvaluationRunRequest, MetricsEventRequest
from app.services.analytics_service import analytics
from app.services.model_evaluation_service import evaluation_service

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("")
def model_metrics(input_type: str = "all", db: Session = Depends(get_db)):
    return evaluation_service.get_metrics_summary(db, input_type=input_type)


@router.get("/overview")
def metrics_overview(user_id: str = "default-user", db: Session = Depends(get_db)):
    live_metrics = analytics.get_metrics_overview(user_id)
    evaluation_metrics = evaluation_service.get_metrics_summary(db)

    face_metrics = evaluation_metrics.get("by_input_type", {}).get("face") or live_metrics.get("facial_model", {})
    voice_metrics = evaluation_metrics.get("by_input_type", {}).get("voice") or live_metrics.get("speech_model", {})
    overall_metrics = {
        key: evaluation_metrics.get(key, 0.0) or live_metrics.get("overall_model", {}).get(key, 0.0)
        for key in ("accuracy", "precision", "recall", "f1_score")
    }

    return {
        **live_metrics,
        "facial_model": face_metrics,
        "speech_model": voice_metrics,
        "overall_model": overall_metrics,
        "evaluation": evaluation_metrics,
    }


@router.post("/evaluate")
def run_evaluation(payload: EvaluationRunRequest, db: Session = Depends(get_db)):
    return evaluation_service.run_evaluation(db, input_type=payload.input_type)


@router.post("/events")
def log_metrics_event(payload: MetricsEventRequest):
    category = (payload.category or "").strip().lower()
    if category == "playback":
        analytics.log_playback_event(
            user_id=payload.user_id,
            action=payload.action,
            track_id=payload.track_id,
            track_name=payload.track_name,
            artist=payload.artist,
            emotion=payload.emotion,
            source=payload.source,
        )
    elif category == "ui":
        analytics.log_ui_event(
            user_id=payload.user_id,
            action=payload.action,
            metadata=payload.metadata,
        )
    else:
        raise HTTPException(status_code=400, detail="Unsupported metrics event category")

    return {"ok": True}


@router.post("/emotion-feedback")
def log_emotion_feedback(payload: EmotionFeedbackRequest):
    analytics.log_feedback_event(
        user_id=payload.user_id,
        action="emotion_correct" if payload.is_correct else "emotion_incorrect",
        track_id=None,
        emotion=payload.predicted_label,
        value=1.0 if payload.is_correct else 0.0,
        comment=payload.corrected_label or payload.input_type,
    )
    return {"ok": True}
