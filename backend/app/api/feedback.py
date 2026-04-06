from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.feedback import Feedback
from app.models.recommended_song import RecommendedSong
from app.schemas import FeedbackRequest
from app.services.analytics_service import analytics
from app.services.feedback_service import FeedbackBandit
from app.services.user_lookup_service import get_or_create_user

router = APIRouter(prefix="/feedback", tags=["feedback"])
bandit = FeedbackBandit()


@router.post("")
def submit_feedback(payload: FeedbackRequest, db: Session = Depends(get_db)):
    action = (payload.event_type or "").strip().lower()
    if action not in {"like", "dislike", "skip"}:
        raise HTTPException(status_code=400, detail="Feedback action must be one of: like, dislike, skip")

    song_id = payload.song_id
    if song_id is None and payload.track_id is not None:
        try:
            song_id = int(payload.track_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="track_id must be a database song id") from exc

    if song_id is None:
        raise HTTPException(status_code=400, detail="song_id (or numeric track_id) is required")

    song = db.query(RecommendedSong).filter(RecommendedSong.id == song_id).one_or_none()
    if song is None:
        raise HTTPException(status_code=404, detail="Song not found")

    user = get_or_create_user(db, payload.user_id)
    emotion_at_time = (payload.emotion or "").strip().lower() or "neutral"

    event = Feedback(
        user_id=user.id,
        song_id=song.id,
        emotion_at_time=emotion_at_time,
        action=action,
        comment=payload.comment,
    )
    db.add(event)
    db.commit()

    reward = bandit.reward_from_event(action, payload.value)
    bandit.update(db, payload.user_id, emotion_at_time, reward)

    try:
        analytics.log_feedback_event(
            user_id=payload.user_id,
            action=action,
            song_id=song.id,
            track_id=str(song.id),
            emotion=emotion_at_time,
            value=payload.value,
            comment=payload.comment,
        )
    except Exception:
        pass

    return {"ok": True}
