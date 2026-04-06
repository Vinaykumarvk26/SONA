import io

import cv2
import numpy as np
import torch
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from app.database import get_db
from app.ml.constants import FER_EMOTIONS
from app.schemas import EmotionScores, MultimodalResult, TimelinePoint, TimelineResponse
from app.services.context_service import (
    get_emotion_timeline,
    get_recent_skip_rate,
    load_user_history_vector,
    save_emotion_timeline,
)
from app.services.analytics_service import analytics
from app.services.text_emotion_service import analyze_text_emotion, fuse_voice_and_text_emotion

router = APIRouter(tags=["emotion"])

inference_engine = None


def set_engine(engine):
    global inference_engine
    inference_engine = engine


def _ensure_engine():
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine unavailable")


def _neutral_prediction():
    scores = {e: 1.0 / len(FER_EMOTIONS) for e in FER_EMOTIONS}
    return {"emotion": "neutral", "confidence": scores["neutral"], "scores": scores}


def _prediction_to_scores(pred: dict):
    return EmotionScores(scores=pred["scores"], label=pred["emotion"], confidence=pred["confidence"])


async def _detect_face_impl(image: UploadFile, user_id: str = "default-user"):
    _ensure_engine()
    data = await image.read()
    try:
        pred = inference_engine.predict_face(data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    response = EmotionScores(scores=pred.scores, label=pred.emotion, confidence=pred.confidence)
    try:
        analytics.log_emotion_event(user_id=user_id, channel="face", label=response.label, confidence=response.confidence)
    except Exception:
        pass
    return response


@router.post("/emotion/face", response_model=EmotionScores)
async def infer_face(image: UploadFile = File(...), user_id: str = Form(default="default-user")):
    return await _detect_face_impl(image, user_id=user_id)


@router.post("/detect-face", response_model=EmotionScores)
async def detect_face(image: UploadFile = File(...), user_id: str = Form(default="default-user")):
    return await _detect_face_impl(image, user_id=user_id)


async def _detect_voice_impl(audio: UploadFile, transcript: str = "", user_id: str = "default-user"):
    _ensure_engine()
    data = await audio.read()
    try:
        pred = inference_engine.predict_voice(data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    text_analysis = analyze_text_emotion(transcript)
    fused = fuse_voice_and_text_emotion(pred.scores, text_analysis)
    response = EmotionScores(
        scores=fused["scores"],
        label=fused["label"],
        confidence=fused["confidence"],
        transcript=text_analysis["transcript"] if text_analysis else transcript or None,
        text_scores=text_analysis["scores"] if text_analysis else None,
        source_weights=fused["weights"],
    )
    try:
        analytics.log_emotion_event(
            user_id=user_id,
            channel="voice",
            label=response.label,
            confidence=response.confidence,
            transcript=response.transcript,
            source_weights=response.source_weights,
        )
    except Exception:
        pass
    return response


@router.post("/emotion/speech", response_model=EmotionScores)
async def infer_speech(audio: UploadFile = File(...), transcript: str = Form(default=""), user_id: str = Form(default="default-user")):
    return await _detect_voice_impl(audio, transcript=transcript, user_id=user_id)


@router.post("/detect-voice", response_model=EmotionScores)
async def detect_voice(audio: UploadFile = File(...), transcript: str = Form(default=""), user_id: str = Form(default="default-user")):
    return await _detect_voice_impl(audio, transcript=transcript, user_id=user_id)


@router.post("/emotion/multimodal", response_model=MultimodalResult)
@router.post("/detect-multimodal", response_model=MultimodalResult)
async def infer_multimodal(
    user_id: str = Form(default="default-user"),
    time_of_day: float = Form(default=0.5),
    skip_rate: float = Form(default=0.0),
    device_type: str = Form(default="desktop"),
    listening_history: str = Form(default=""),
    transcript: str = Form(default=""),
    image: UploadFile | None = File(default=None),
    audio: UploadFile | None = File(default=None),
    db: Session = Depends(get_db),
):
    _ensure_engine()
    has_face = image is not None
    has_voice = audio is not None

    if not has_face and not has_voice:
        raise HTTPException(status_code=400, detail="At least one modality file (image/audio) is required")

    face_pred_obj = None
    speech_pred_obj = None

    if image is not None:
        face_data = await image.read()
        try:
            face_pred_obj = inference_engine.predict_face(face_data)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        face_pred = {"emotion": face_pred_obj.emotion, "confidence": face_pred_obj.confidence, "scores": face_pred_obj.scores}
    else:
        face_pred = _neutral_prediction()

    if audio is not None:
        speech_data = await audio.read()
        try:
            speech_pred_obj = inference_engine.predict_voice(speech_data)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        text_analysis = analyze_text_emotion(transcript)
        speech_fused = fuse_voice_and_text_emotion(speech_pred_obj.scores, text_analysis)
        speech_pred = {
            "emotion": speech_fused["label"],
            "confidence": speech_fused["confidence"],
            "scores": speech_fused["scores"],
        }
    else:
        speech_pred = _neutral_prediction()

    history = []
    if listening_history.strip():
        try:
            history = [float(x.strip()) for x in listening_history.split(",") if x.strip()]
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid listening_history format") from exc

    context = {
        "time_of_day": max(0.0, min(1.0, float(time_of_day))),
        "skip_rate": max(0.0, min(1.0, float(skip_rate))),
        "device_type": device_type,
        "listening_history": history,
    }

    if not context["listening_history"]:
        context["listening_history"] = load_user_history_vector(db, user_id, size=20)
    context["skip_rate"] = max(context["skip_rate"], get_recent_skip_rate(db, user_id))

    # Build synthetic EmotionPrediction-like objects for fusion API.
    class _Pred:
        def __init__(self, p, embedding_dim: int):
            self.emotion = p["emotion"]
            self.confidence = p["confidence"]
            self.scores = p["scores"]
            self.embedding = torch.zeros(1, embedding_dim, device=inference_engine.device)

    if face_pred_obj is not None:
        face_pred_real = face_pred_obj
    else:
        face_pred_real = _Pred(face_pred, embedding_dim=256)

    if speech_pred_obj is not None:
        speech_pred_real = speech_pred_obj
    else:
        speech_pred_real = _Pred(speech_pred, embedding_dim=128)

    fused = inference_engine.predict_multimodal(
        face_pred_real,
        speech_pred_real,
        has_face=has_face,
        has_voice=has_voice,
    )

    save_emotion_timeline(
        db,
        user_identifier=user_id,
        face_emotion=face_pred["emotion"],
        face_confidence=face_pred["confidence"],
        voice_emotion=speech_pred["emotion"],
        voice_confidence=speech_pred["confidence"],
        fused_emotion=fused["emotion"],
        fused_confidence=fused["confidence"],
    )
    try:
        analytics.log_emotion_event(
            user_id=user_id,
            channel="fusion",
            label=fused["emotion"],
            confidence=fused["confidence"],
            transcript=transcript,
            source_weights=fused["weights"],
        )
    except Exception:
        pass

    return MultimodalResult(
        facial=EmotionScores(scores=face_pred["scores"], label=face_pred["emotion"], confidence=face_pred["confidence"]),
        speech=EmotionScores(scores=speech_pred["scores"], label=speech_pred["emotion"], confidence=speech_pred["confidence"]),
        fused=EmotionScores(scores=fused["scores"], label=fused["emotion"], confidence=fused["confidence"]),
        fusion_weights=fused["weights"],
    )


@router.post("/emotion/frame", response_model=EmotionScores)
async def infer_frame(image: UploadFile = File(...)):
    _ensure_engine()
    data = await image.read()
    nparr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid frame")

    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise HTTPException(status_code=400, detail="Frame encoding failed")

    try:
        pred = inference_engine.predict_face(encoded.tobytes())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return EmotionScores(scores=pred.scores, label=pred.emotion, confidence=pred.confidence)


@router.get("/emotion/timeline", response_model=TimelineResponse)
def timeline(user_id: str = Query("default-user"), db: Session = Depends(get_db)):
    rows = get_emotion_timeline(db, user_id)
    points = [
        TimelinePoint(
            timestamp=row.created_at.isoformat(),
            emotion=row.fused_emotion,
            confidence=row.fused_confidence,
        )
        for row in rows
    ]
    return TimelineResponse(user_id=user_id, points=points)
