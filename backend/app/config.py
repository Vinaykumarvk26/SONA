from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Emotion-Aware Music Recommender"
    database_url: str = "sqlite:///./emotion_music.db"
    mongodb_url: str = ""
    mongodb_db_name: str = "sona"
    spotify_client_id: str = ""
    spotify_client_secret: str = ""
    spotify_market: str = "IN"
    fer_model_path: str = "./checkpoints/fer_best.pt"
    speech_model_path: str = "./checkpoints/ser_best.pt"
    fusion_model_path: str = "./checkpoints/fusion_best.pt"
    legacy_fer_model_path: str = "./checkpoints/fer_hybrid.pt"
    legacy_speech_model_path: str = "./checkpoints/speech_emotion.pt"
    fer_model_url: str = ""
    speech_model_url: str = ""
    fusion_model_url: str = ""
    auto_init_models: bool = True
    face_detector: str = "haar"
    fer_use_hf_model: bool = False
    hf_fer_model_id: str = "trpakov/vit-face-expression"
    hf_fer_local_files_only: bool = False
    ser_use_hf_model: bool = False
    hf_ser_model_id: str = "r-f/wav2vec-english-speech-emotion-recognition"
    hf_local_files_only: bool = False
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:5173,https://ai-sona.vercel.app"
    expose_reset_token_in_response: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
