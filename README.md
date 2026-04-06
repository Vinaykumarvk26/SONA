# Emotion-Aware Music Recommendation System

Production-oriented FastAPI + React system with real FER (face), SER (speech), and multimodal fusion pipelines.

## Backend ML Architecture

### FER (Facial Emotion Recognition)
- Dataset: FER2013 (`fer2013.csv`)
- Preprocessing:
  - grayscale input
  - resize `48x48`
  - normalize to `[0, 1]`
  - split: train `70%`, val `15%`, test `15%`
- Model:
  - CNN feature extractor
  - ViT block on CNN patch tokens
  - dropout `0.5`
  - FC output: `7` classes
- Training:
  - loss: `CrossEntropyLoss`
  - optimizer: `Adam`
  - lr: `1e-4`
  - batch size: `32`
  - epochs: `50`
  - early stopping
  - best checkpoint by val accuracy

### SER (Speech Emotion Recognition)
- Dataset: RAVDESS
- Preprocessing:
  - MFCC: `40`
  - per-coefficient normalization
  - pad/truncate to fixed frame length
  - split: train `70%`, val `15%`, test `15%`
- Model:
  - CNN front-end
  - LSTM temporal model
  - FC + softmax logits (7 aligned classes)
- Training:
  - loss: `CrossEntropyLoss`
  - optimizer: `Adam`
  - lr: `1e-4`
  - epochs: `40`
  - best checkpoint by val accuracy

### Multimodal Fusion
- Feature-level fusion:
  - face embedding + speech embedding concatenation
  - attention-gated fusion head
- Decision-level fusion:
  - weighted average of probabilities
  - dynamic weights based on modality confidences
- Final prediction:
  - dynamic blend of feature-level and decision-level outputs

## New Backend Structure

```text
backend/
  app/
    ml/
      constants.py
      inference.py
      datasets/
        fer2013_dataset.py
        ravdess_dataset.py
        fusion_embedding_dataset.py
      models/
        fer_cnn_vit.py
        ser_cnn_lstm.py
        fusion_attention.py
      training/
        early_stopping.py
        metrics.py
    api/
      emotion.py
    main.py
    config.py
  train/
    train_fer.py
    train_speech.py
    train_fusion.py
    evaluate_fer.py
    evaluate_speech.py
```

## API Endpoints (Real Inference)

- `POST /detect-face`
  - file: `image`
  - returns: emotion + confidence + class scores

- `POST /detect-voice`
  - file: `audio`
  - returns: emotion + confidence + class scores

- `POST /detect-multimodal`
  - files: `image` and/or `audio`
  - form fields: `user_id`, `time_of_day`, `skip_rate`, `device_type`, `listening_history`
  - returns: face, speech, fused outputs + fusion weights

Compatibility routes kept:
- `/emotion/face`
- `/emotion/speech`
- `/emotion/multimodal`

## Training Commands

### FER Training
```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python train\train_fer.py --csv-path "C:\datasets\fer2013\fer2013.csv" --output-dir "./checkpoints"
```

### SER Training
```powershell
python train\train_speech.py --data-dir "C:\datasets\ravdess" --output-dir "./checkpoints"
```

### Fusion Head Training (optional, if you have prepared embeddings)
```powershell
python train\train_fusion.py --npz-path "C:\datasets\fusion_embeddings.npz" --output-dir "./checkpoints"
```

### Evaluation
```powershell
python train\evaluate_fer.py --csv-path "C:\datasets\fer2013\fer2013.csv" --checkpoint "./checkpoints/fer_best.pt"
python train\evaluate_speech.py --data-dir "C:\datasets\ravdess" --checkpoint "./checkpoints/ser_best.pt"
```

Metrics include:
- accuracy
- precision
- recall
- f1-score
- confusion matrix

## Runtime Model Loading

Configured in `backend/.env`:
- `FER_MODEL_PATH=./checkpoints/fer_best.pt`
- `SPEECH_MODEL_PATH=./checkpoints/ser_best.pt`
- `FUSION_MODEL_PATH=./checkpoints/fusion_best.pt`
- `FER_USE_HF_MODEL=true`
- `HF_FER_MODEL_ID=trpakov/vit-face-expression`
- `HF_FER_LOCAL_FILES_ONLY=false`
- `SER_USE_HF_MODEL=true`
- `HF_SER_MODEL_ID=r-f/wav2vec-english-speech-emotion-recognition`
- `HF_LOCAL_FILES_ONLY=false`

`FER` and `SER` checkpoints are expected for inference. If missing or invalid, API still starts and detection endpoints return `503` until checkpoints are provided.
Fusion checkpoint is optional; if missing, system remains decision-level dominant.
When enabled, pretrained HF FER/SER models are loaded once at startup and fused with local CNN models for stronger real-world robustness on webcam + microphone inputs.

## Run Backend

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Notes

- CUDA is used automatically if available.
- Models are loaded once at startup (not per request).
- Inference path includes logging and exception-safe API handling.
- If backend fails to start with `ModuleNotFoundError: torch`, install backend deps first: `pip install -r requirements.txt`.
