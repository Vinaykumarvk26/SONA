from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import PreferenceResponse, PreferenceUpdateRequest
from app.services.auth_service import validate_token
from app.services.mongo_app_service import mongo_app

router = APIRouter(prefix="/preferences", tags=["preferences"])


def _extract_bearer(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization token")
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    return parts[1]


@router.get("/me", response_model=PreferenceResponse)
def get_preferences_me(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
):
    user = validate_token(db, _extract_bearer(authorization))
    data = mongo_app.get_preferences(user=user)
    return PreferenceResponse(
        playback_mode=data.get("playback_mode", "autoplay"),
        music_source=data.get("music_source", "Spotify"),
        languages=data.get("languages", ["Hindi", "English", "Telugu", "Tamil", "Kannada", "Malayalam"]),
        theme=data.get("theme", "dark"),
    )


@router.put("/me", response_model=PreferenceResponse)
def update_preferences_me(
    payload: PreferenceUpdateRequest,
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
):
    user = validate_token(db, _extract_bearer(authorization))
    saved = mongo_app.save_preferences(
        user=user,
        playback_mode=payload.playback_mode,
        music_source=payload.music_source,
        languages=payload.languages,
        theme=payload.theme,
    )
    mongo_app.log_audit(action="preferences_updated", user=user, metadata=saved)
    return PreferenceResponse(
        playback_mode=saved["playback_mode"],
        music_source=saved["music_source"],
        languages=saved["languages"],
        theme=saved["theme"],
    )
