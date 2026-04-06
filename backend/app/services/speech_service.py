import io

import librosa
import numpy as np
import soundfile as sf
import torch

from app.schemas import EMOTIONS


class SpeechEmotionService:
    def __init__(self, model: torch.nn.Module, device: torch.device, sample_rate: int = 16000, n_mfcc: int = 40):
        self.model = model.to(device)
        self.device = device
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.model.eval()

    def _extract_mfcc(self, y: np.ndarray, sr: int) -> np.ndarray:
        if len(y) < sr:
            pad = sr - len(y)
            y = np.pad(y, (0, pad), mode="constant")
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        mfcc = librosa.util.normalize(mfcc)
        return mfcc.astype(np.float32)

    def predict_from_wav_bytes(self, wav_bytes: bytes):
        buffer = io.BytesIO(wav_bytes)
        y, sr = sf.read(buffer)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if sr != self.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate

        mfcc = self._extract_mfcc(y, sr)
        x = torch.from_numpy(mfcc).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        scores = {emotion: float(prob) for emotion, prob in zip(EMOTIONS, probs)}
        idx = int(np.argmax(probs))
        return scores, EMOTIONS[idx], float(probs[idx])
