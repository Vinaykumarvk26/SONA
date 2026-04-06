from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


EMOTIONS = ["happy", "sad", "angry", "fear", "disgust", "surprise", "neutral"]


class ContextFeatures(BaseModel):
    time_of_day: float = Field(ge=0.0, le=1.0)
    skip_rate: float = Field(ge=0.0, le=1.0)
    device_type: str
    listening_history: List[float] = Field(default_factory=list)


class InferenceRequest(BaseModel):
    user_id: str = "default-user"
    context: ContextFeatures


class EmotionScores(BaseModel):
    scores: Dict[str, float]
    confidence: float
    label: str
    transcript: Optional[str] = None
    text_scores: Optional[Dict[str, float]] = None
    source_weights: Optional[Dict[str, float]] = None


class MultimodalResult(BaseModel):
    facial: EmotionScores
    speech: EmotionScores
    fused: EmotionScores
    fusion_weights: Dict[str, float]


class TrackInfo(BaseModel):
    id: str
    name: str
    artist: str
    preview_url: Optional[str] = None
    external_url: Optional[str] = None
    image_url: Optional[str] = None
    embed_url: Optional[str] = None


class RecommendationResponse(BaseModel):
    emotion: str
    confidence: float
    tracks: List[TrackInfo]
    target_features: Dict[str, float]


class FeedbackRequest(BaseModel):
    user_id: str = "default-user"
    song_id: int | None = None
    track_id: str | None = None
    event_type: str
    value: float = 0.0
    emotion: Optional[str] = None
    comment: Optional[str] = None


class MetricsEventRequest(BaseModel):
    user_id: str = "default-user"
    category: str
    action: str
    track_id: Optional[str] = None
    track_name: Optional[str] = None
    artist: Optional[str] = None
    emotion: Optional[str] = None
    source: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluationRunRequest(BaseModel):
    input_type: str = "all"


class EmotionFeedbackRequest(BaseModel):
    user_id: str = "default-user"
    input_type: str
    predicted_label: str
    confidence_score: float = 0.0
    is_correct: bool
    corrected_label: Optional[str] = None


class PreferenceUpdateRequest(BaseModel):
    playback_mode: str = "autoplay"
    music_source: str = "Spotify"
    languages: List[str] = Field(default_factory=lambda: ["Hindi", "English", "Telugu", "Tamil", "Kannada", "Malayalam"])
    theme: str = "dark"


class PreferenceResponse(BaseModel):
    playback_mode: str
    music_source: str
    languages: List[str]
    theme: str


class TimelinePoint(BaseModel):
    timestamp: str
    emotion: str
    confidence: float


class TimelineResponse(BaseModel):
    user_id: str
    points: List[TimelinePoint]


class AuthSignupRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    email: str
    password: str = Field(min_length=8)


class AuthSigninRequest(BaseModel):
    identifier: Optional[str] = None
    email: Optional[str] = None
    password: str


class AuthGoogleRequest(BaseModel):
    access_token: str


class AuthResponse(BaseModel):
    token: str
    email: str
    username: Optional[str] = None


class UserProfileResponse(BaseModel):
    email: str
    username: str
    full_name: Optional[str] = ""
    phone: Optional[str] = ""
    location: Optional[str] = ""
    bio: Optional[str] = ""


class UserProfileUpdateRequest(BaseModel):
    username: Optional[str] = Field(default=None, min_length=3, max_length=64)
    full_name: Optional[str] = Field(default=None, max_length=120)
    phone: Optional[str] = Field(default=None, max_length=32)
    location: Optional[str] = Field(default=None, max_length=120)
    bio: Optional[str] = Field(default=None, max_length=400)


class ForgotPasswordRequest(BaseModel):
    identifier: str


class ForgotPasswordResponse(BaseModel):
    message: str
    reset_token: Optional[str] = None


class ResetPasswordRequest(BaseModel):
    reset_token: str
    new_password: str = Field(min_length=8)
