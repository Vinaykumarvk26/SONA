from contextlib import asynccontextmanager
from pathlib import Path
import logging

import torch
from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import models  # noqa: F401
from app.api import auth, emotion, feedback, metrics, preferences, recommend
from app.config import get_settings
from app.database import init_db
from app.ml.inference import EmotionInferenceEngine
from app.ml.legacy_inference import LegacyInferenceEngine
from app.ml.models import FERCNNViT, FeatureFusionAttention, SERCNNLSTM
from app.models.fer_model import FERHybridNet
from app.models.speech_model import SpeechEmotionNet
from app.services.model_loader import ensure_model_checkpoint
from app.services.mongo_app_service import mongo_app
from app.utils.logging import setup_logging

logger = logging.getLogger("main")


def _load_checkpoint_or_warn(model, checkpoint_path: str, checkpoint_url: str = "") -> bool:
    try:
        ensure_model_checkpoint(model, checkpoint_path, checkpoint_url)
        return True
    except Exception as exc:  # noqa: BLE001 - keep API startup alive even if models are unavailable
        logger.exception("Failed to load checkpoint at %s: %s", checkpoint_path, exc)
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)
    init_db()
    try:
        mongo_app.ensure_indexes()
        mongo_app.sync_model_metrics()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Mongo app bootstrap failed: %s", exc)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fer_model = FERCNNViT(num_classes=7, dropout=0.5)
    ser_model = SERCNNLSTM(n_mfcc=40, num_classes=7, dropout=0.5)
    fusion_model = FeatureFusionAttention(face_dim=256, speech_dim=128, num_classes=7)

    fer_ready = _load_checkpoint_or_warn(fer_model, settings.fer_model_path, settings.fer_model_url)
    ser_ready = _load_checkpoint_or_warn(ser_model, settings.speech_model_path, settings.speech_model_url)
    fusion_ready = False
    hf_fer_ready = False
    hf_ser_ready = False
    hf_fer_model = None
    hf_ser_model = None

    fusion_path = Path(settings.fusion_model_path)
    if not fusion_path.exists() and settings.fusion_model_url:
        fusion_ready = _load_checkpoint_or_warn(fusion_model, settings.fusion_model_path, settings.fusion_model_url)
    elif fusion_path.exists():
        try:
            state = torch.load(fusion_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            fusion_model.load_state_dict(state)
            fusion_ready = True
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to load fusion checkpoint at %s: %s", fusion_path, exc)
    else:
        # Decision-level fusion remains active even without a trained feature-fusion checkpoint.
        logger.warning("Fusion checkpoint missing at %s; using decision-level dominant fusion.", fusion_path)

    if settings.fer_use_hf_model:
        try:
            from app.ml.hf_fer import HFFaceEmotionModel

            hf_fer_model = HFFaceEmotionModel(
                model_id=settings.hf_fer_model_id,
                device=device,
                local_files_only=settings.hf_fer_local_files_only,
            )
            hf_fer_ready = True
        except Exception as exc:  # noqa: BLE001 - do not block startup
            logger.exception("Failed to load HF FER model '%s': %s", settings.hf_fer_model_id, exc)

    if settings.ser_use_hf_model:
        try:
            from app.ml.hf_ser import HFWav2VecSER

            hf_ser_model = HFWav2VecSER(
                model_id=settings.hf_ser_model_id,
                device=device,
                local_files_only=settings.hf_local_files_only,
            )
            hf_ser_ready = True
        except Exception as exc:  # noqa: BLE001 - do not block startup
            logger.exception("Failed to load HF SER model '%s': %s", settings.hf_ser_model_id, exc)

    legacy_ready = False
    using_legacy_models = False
    using_legacy_face_model = False

    if fer_ready and ser_ready:
        engine = EmotionInferenceEngine(
            fer_model=fer_model,
            ser_model=ser_model,
            fusion_model=fusion_model,
            device=device,
            use_feature_fusion=fusion_ready,
            fer_hf_model=hf_fer_model,
            ser_hf_model=hf_ser_model,
        )
        emotion.set_engine(engine)
    else:
        # Compatibility fallback for older checkpoints in existing deployments.
        try:
            legacy_fer = FERHybridNet(num_classes=7)
            legacy_ser = SpeechEmotionNet(n_mfcc=40, num_classes=7)
            ensure_model_checkpoint(legacy_fer, settings.legacy_fer_model_path, "")
            ensure_model_checkpoint(legacy_ser, settings.legacy_speech_model_path, "")
            emotion.set_engine(LegacyInferenceEngine(fer_model=legacy_fer, ser_model=legacy_ser, device=device))
            legacy_ready = True
            using_legacy_models = True
            logger.warning("Using legacy FER/SER checkpoints for inference compatibility.")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Legacy checkpoint fallback failed: %s", exc)
            emotion.set_engine(None)

    app.state.active_fer_checkpoint = settings.legacy_fer_model_path if using_legacy_face_model else settings.fer_model_path
    app.state.model_status = {
        "fer_ready": fer_ready,
        "ser_ready": ser_ready,
        "fusion_ready": fusion_ready,
        "hf_fer_ready": hf_fer_ready,
        "hf_ser_ready": hf_ser_ready,
        "legacy_ready": legacy_ready,
        "using_legacy_models": using_legacy_models,
        "using_legacy_face_model": using_legacy_face_model,
        "fer_backend": "legacy_hybrid" if using_legacy_face_model else ("hybrid_hf+cnn_vit" if hf_fer_ready and fer_ready else "cnn_vit"),
        "ser_backend": "hybrid_hf+cnn_lstm" if hf_ser_ready and ser_ready else "cnn_lstm",
    }
    yield


settings = get_settings()
app = FastAPI(title=settings.app_name, version="2.0.0", lifespan=lifespan)

origins = [x.strip() for x in settings.cors_origins.split(",") if x.strip()]


def _expand_local_origins(values: list[str]) -> list[str]:
    expanded = set(values)
    for origin in values:
        if "://localhost" in origin:
            expanded.add(origin.replace("://localhost", "://127.0.0.1"))
        if "://127.0.0.1" in origin:
            expanded.add(origin.replace("://127.0.0.1", "://localhost"))
    return sorted(expanded)


origins = _expand_local_origins(origins)
lan_origin_regex = (
    r"^https?://("
    r"localhost|127\.0\.0\.1|"
    r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
    r"192\.168\.\d{1,3}\.\d{1,3}|"
    r"172\.(1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3}"
    r")(:\d+)?$"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=lan_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    status = getattr(app.state, "model_status", {"fer_ready": False, "ser_ready": False, "fusion_ready": False})
    return {
        "status": "ok",
        "version": "2.0.0",
        "fer_checkpoint": getattr(app.state, "active_fer_checkpoint", settings.fer_model_path),
        "ser_checkpoint": settings.speech_model_path,
        "fusion_checkpoint": settings.fusion_model_path,
        "models": status,
    }

@app.get("/")
def api_index():
    return {
        "name": settings.app_name,
        "version": "2.0.0",
        "docs": "/docs",
        "api_base": "/api/v1",
        "routes": {
            "health": "/api/v1/health",
            "auth": "/api/v1/auth",
            "emotion": "/api/v1/emotion",
            "recommendations": "/api/v1/recommendations",
            "feedback": "/api/v1/feedback",
            "metrics": "/api/v1/metrics",
            "preferences": "/api/v1/preferences",
        },
    }


@app.get("/api/v1/health")
def health_v1():
    return health()


api_v1 = APIRouter(prefix="/api/v1")
api_v1.include_router(emotion.router)
api_v1.include_router(recommend.router)
api_v1.include_router(feedback.router)
api_v1.include_router(metrics.router)
api_v1.include_router(preferences.router)
api_v1.include_router(auth.router)

app.include_router(api_v1)

# Legacy aliases kept so the current frontend and older clients continue to work.
app.include_router(emotion.router)
app.include_router(recommend.legacy_router)
app.include_router(feedback.router)
app.include_router(metrics.router)
app.include_router(preferences.router)
app.include_router(auth.router)
