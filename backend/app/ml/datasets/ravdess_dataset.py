from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from app.ml.constants import FER_EMOTIONS, RAVDESS_EMOTION_MAP

RAVDESS_KAGGLE_REF = "orvile/ravdess-dataset"
PREFERRED_SPEECH_DIRS = (
    "Audio_Speech_Actors_01-24_16k",
    "Audio_Speech_Actors_01-24",
)


def _extract_ravdess_code(file_name: str) -> str | None:
    m = re.match(r"\d{2}-\d{2}-(\d{2})-\d{2}-\d{2}-\d{2}-\d{2}\.wav", file_name)
    return m.group(1) if m else None


def _to_fer_index(emotion_name: str) -> int:
    return FER_EMOTIONS.index(emotion_name)


class RAVDESSDataset(Dataset):
    def __init__(
        self,
        files: list[str],
        labels: list[int],
        sample_rate: int = 16000,
        n_mfcc: int = 40,
        n_mels: int = 128,
        max_frames: int = 180,
        feature_type: str = "mfcc",
        augment: bool = False,
    ):
        self.files = files
        self.labels = labels
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.max_frames = max_frames
        self.feature_type = feature_type
        self.augment = augment

    def __len__(self) -> int:
        return len(self.files)

    def _augment_audio(self, y: np.ndarray) -> np.ndarray:
        if len(y) == 0:
            return y

        if np.random.rand() < 0.45:
            y = librosa.effects.time_stretch(y, rate=float(np.random.uniform(0.9, 1.1)))

        if np.random.rand() < 0.45:
            y = librosa.effects.pitch_shift(y, sr=self.sample_rate, n_steps=float(np.random.uniform(-2.0, 2.0)))

        if np.random.rand() < 0.55:
            noise_scale = float(np.random.uniform(0.001, 0.008))
            y = y + np.random.normal(0.0, noise_scale, size=y.shape).astype(np.float32)

        peak = np.max(np.abs(y)) if len(y) > 0 else 0.0
        if peak > 1e-6:
            y = y / peak
        return y.astype(np.float32, copy=False)

    def _extract_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        if self.feature_type == "mel":
            feat = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_mels=self.n_mels,
                n_fft=1024,
                hop_length=256,
                win_length=512,
                fmax=sr // 2,
            )
            feat = librosa.power_to_db(feat, ref=np.max)
        else:
            feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)

        feat = (feat - feat.mean(axis=1, keepdims=True)) / (feat.std(axis=1, keepdims=True) + 1e-6)

        if feat.shape[1] < self.max_frames:
            feat = np.pad(feat, ((0, 0), (0, self.max_frames - feat.shape[1])), mode="constant")
        else:
            feat = feat[:, : self.max_frames]
        return feat.astype(np.float32, copy=False)

    def __getitem__(self, idx: int):
        y, sr = librosa.load(self.files[idx], sr=self.sample_rate)
        if self.augment:
            y = self._augment_audio(y)

        features = self._extract_features(y, sr)
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


@dataclass
class SERDataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


def ensure_ravdess_root(root_dir: str | Path) -> Path:
    root_dir = Path(root_dir)
    if root_dir.exists():
        if any(root_dir.glob("*.wav")):
            return root_dir

        for name in PREFERRED_SPEECH_DIRS:
            preferred_dir = root_dir / name
            if preferred_dir.is_dir() and any(preferred_dir.rglob("*.wav")):
                return preferred_dir

        if any(root_dir.rglob("*.wav")):
            return root_dir

    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "RAVDESS dataset path is missing and kagglehub is not installed. "
            "Install backend requirements or provide a valid local RAVDESS path."
        ) from exc

    try:
        download_dir = Path(kagglehub.dataset_download(RAVDESS_KAGGLE_REF))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to download RAVDESS from Kaggle. Configure Kaggle credentials and retry."
        ) from exc

    wav_roots = [path for path in [download_dir, *download_dir.rglob("*")] if path.is_dir() and any(path.glob("*.wav"))]
    if not wav_roots:
        raise FileNotFoundError(f"Downloaded RAVDESS dataset at {download_dir} does not contain WAV files.")

    for name in PREFERRED_SPEECH_DIRS:
        preferred_matches = [
            path for path in [download_dir, *download_dir.rglob(name)] if path.is_dir() and path.name == name and any(path.rglob("*.wav"))
        ]
        if preferred_matches:
            return preferred_matches[0]

    # Prefer the deepest directory that directly contains the actor folders/audio files.
    return max(wav_roots, key=lambda path: len(path.parts))


def collect_ravdess(root_dir: str | Path):
    root_dir = ensure_ravdess_root(root_dir)
    files: list[str] = []
    labels: list[int] = []

    for wav_path in root_dir.rglob("*.wav"):
        code = _extract_ravdess_code(wav_path.name)
        if code is None:
            continue
        mapped = RAVDESS_EMOTION_MAP.get(code)
        if mapped is None:
            continue
        files.append(str(wav_path))
        labels.append(_to_fer_index(mapped))

    if not files:
        raise RuntimeError(f"No RAVDESS files found under {root_dir}")

    return files, labels


def build_ravdess_loaders(
    root_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
    feature_type: str = "mfcc",
    n_mels: int = 128,
    max_frames: int = 180,
) -> SERDataBundle:
    files, labels = collect_ravdess(root_dir)

    train_files, temp_files, train_labels, temp_labels = train_test_split(
        files, labels, test_size=0.30, random_state=seed, stratify=labels
    )
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels, test_size=0.50, random_state=seed, stratify=temp_labels
    )

    train_ds = RAVDESSDataset(
        train_files,
        train_labels,
        feature_type=feature_type,
        n_mels=n_mels,
        max_frames=max_frames,
        augment=True,
    )
    val_ds = RAVDESSDataset(
        val_files,
        val_labels,
        feature_type=feature_type,
        n_mels=n_mels,
        max_frames=max_frames,
        augment=False,
    )
    test_ds = RAVDESSDataset(
        test_files,
        test_labels,
        feature_type=feature_type,
        n_mels=n_mels,
        max_frames=max_frames,
        augment=False,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return SERDataBundle(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
