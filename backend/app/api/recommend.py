from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.emotion_record import EmotionRecord
from app.models.recommended_song import RecommendedSong
from app.schemas import RecommendationResponse
from app.services.analytics_service import analytics
from app.services.context_service import get_latest_emotion_record, save_emotion_timeline
from app.services.feedback_service import FeedbackBandit
from app.services.spotify_service import SpotifyService
from app.services.user_lookup_service import resolve_user

router = APIRouter(prefix="/recommendations", tags=["recommend"])
legacy_router = APIRouter(tags=["recommend-legacy"])
spotify = SpotifyService()
bandit = FeedbackBandit()


def _track_key(title: str, artist: str) -> str:
    return f"{(title or '').strip().lower()}::{(artist or '').strip().lower()}"


def _recent_user_song_keys(db: Session, user_id: str, limit: int = 40) -> set[str]:
    user = resolve_user(db, user_id)
    if user is None:
        return set()

    rows = (
        db.query(RecommendedSong.song_title, RecommendedSong.artist_name)
        .join(EmotionRecord, RecommendedSong.emotion_record_id == EmotionRecord.id)
        .filter(EmotionRecord.user_id == user.id)
        .order_by(RecommendedSong.created_at.desc())
        .limit(limit)
        .all()
    )
    return {_track_key(title, artist) for title, artist in rows}


def _track_playability_rank(track: dict) -> int:
    if track.get("preview_url"):
        return 0
    if track.get("embed_url"):
        return 1
    if track.get("external_url"):
        return 2
    return 3


@router.get("", response_model=RecommendationResponse)
@legacy_router.get("/recommendations", response_model=RecommendationResponse)
@legacy_router.get("/recommend", response_model=RecommendationResponse)
def recommend(user_id: str, emotion: str, confidence: float = 0.5, db: Session = Depends(get_db)):
    if not emotion:
        raise HTTPException(status_code=400, detail="Emotion required")

    emotion = emotion.strip().lower()
    target = bandit.personalized_features(db, user_id=user_id, emotion=emotion)
    tracks = spotify.recommend_tracks(target, emotion=emotion, user_id=user_id, limit=20)
    recent_keys = _recent_user_song_keys(db, user_id, limit=40)
    if recent_keys:
        unseen = []
        seen = []
        for track in tracks:
            key = _track_key(track.get("name"), track.get("artist"))
            if key in recent_keys:
                seen.append(track)
            else:
                unseen.append(track)
        tracks = unseen + seen
    tracks.sort(key=_track_playability_rank)
    tracks = tracks[:10]

    emotion_record = get_latest_emotion_record(db, user_id)
    if emotion_record is None:
        # Backfill an emotion record so song recommendations can be traced in timeline/feedback.
        emotion_record = save_emotion_timeline(
            db,
            user_identifier=user_id,
            face_emotion=emotion,
            face_confidence=confidence,
            voice_emotion=emotion,
            voice_confidence=confidence,
            fused_emotion=emotion,
            fused_confidence=confidence,
        )

    persisted_tracks = []
    for track in tracks:
        song = RecommendedSong(
            emotion_record_id=emotion_record.id,
            song_title=track.get("name") or "Unknown",
            artist_name=track.get("artist") or "Unknown",
            spotify_url=track.get("external_url") or track.get("embed_url") or track.get("preview_url"),
            valence_score=float(target.get("target_valence", 0.5)),
            energy_score=float(target.get("target_energy", 0.5)),
            tempo=float(target.get("target_tempo", 110.0)),
        )
        db.add(song)
        db.flush()

        persisted_tracks.append(
            {
                "id": str(song.id),
                "name": song.song_title,
                "artist": song.artist_name,
                "preview_url": track.get("preview_url"),
                "external_url": track.get("external_url") or song.spotify_url,
                "image_url": track.get("image_url"),
                "embed_url": track.get("embed_url"),
            }
        )

    db.commit()

    try:
        analytics.log_recommendation_event(
            user_id=user_id,
            emotion=emotion,
            confidence=confidence,
            tracks=persisted_tracks,
            target_features=target,
        )
    except Exception:
        pass

    return RecommendationResponse(
        emotion=emotion,
        confidence=confidence,
        tracks=persisted_tracks,
        target_features=target,
    )
