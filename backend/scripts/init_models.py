import argparse

from app.config import get_settings
from app.ml.models import FERCNNViT, FeatureFusionAttention, SERCNNLSTM
from app.services.model_loader import ensure_model_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Download/validate model checkpoints for FER/SER/Fusion.")
    parser.add_argument("--fer-url", default="", help="Optional FER checkpoint URL")
    parser.add_argument("--speech-url", default="", help="Optional speech checkpoint URL")
    parser.add_argument("--fusion-url", default="", help="Optional fusion checkpoint URL")
    args = parser.parse_args()

    settings = get_settings()

    fer_model = FERCNNViT(num_classes=7, dropout=0.5)
    speech_model = SERCNNLSTM(n_mfcc=40, num_classes=7, dropout=0.5)
    fusion_model = FeatureFusionAttention(face_dim=256, speech_dim=128, num_classes=7)

    fer_path = ensure_model_checkpoint(fer_model, settings.fer_model_path, args.fer_url or settings.fer_model_url)
    speech_path = ensure_model_checkpoint(
        speech_model,
        settings.speech_model_path,
        args.speech_url or settings.speech_model_url,
    )

    fusion_url = args.fusion_url or settings.fusion_model_url
    if fusion_url:
        fusion_path = ensure_model_checkpoint(fusion_model, settings.fusion_model_path, fusion_url)
        print(f"Fusion model checkpoint ready: {fusion_path}")
    else:
        print("Fusion model URL not provided; skipping fusion checkpoint bootstrap.")

    print(f"FER model checkpoint ready: {fer_path}")
    print(f"Speech model checkpoint ready: {speech_path}")


if __name__ == "__main__":
    main()
