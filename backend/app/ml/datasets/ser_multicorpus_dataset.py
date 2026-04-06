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


def _to_fer_index(emotion_name: str) -> int:
    return FER_EMOTIONS.index(emotion_name)


class SERMultiCorpusDataset(Dataset):
    def __init__(self, files: list[str], labels: list[int], sample_rate: int = 16000, n_mfcc: int = 40, max_frames: int = 180):
        self.files = files
        self.labels = labels
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.max_frames = max_frames

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        y, sr = librosa.load(self.files[idx], sr=self.sample_rate)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)

        mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-6)
        if mfcc.shape[1] < self.max_frames:
            mfcc = np.pad(mfcc, ((0, 0), (0, self.max_frames - mfcc.shape[1])), mode="constant")
        else:
            mfcc = mfcc[:, : self.max_frames]

        x = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


@dataclass
class SERDataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_counts: dict[str, int]


def collect_ravdess(root_dir: str | Path) -> tuple[list[str], list[int]]:
    root = Path(root_dir)
    if not root.exists():
        return [], []

    files: list[str] = []
    labels: list[int] = []
    for wav_path in root.rglob("*.wav"):
        m = re.match(r"\d{2}-\d{2}-(\d{2})-\d{2}-\d{2}-\d{2}-\d{2}\.wav", wav_path.name)
        if not m:
            continue
        code = m.group(1)
        mapped = RAVDESS_EMOTION_MAP.get(code)
        if mapped is None:
            continue
        files.append(str(wav_path))
        labels.append(_to_fer_index(mapped))

    return files, labels


def collect_crema_d(root_dir: str | Path) -> tuple[list[str], list[int]]:
    # CREMA-D filename pattern example: 1001_DFA_ANG_XX.wav
    root = Path(root_dir)
    if not root.exists():
        return [], []

    code_to_emotion = {
        "ANG": "angry",
        "DIS": "disgust",
        "FEA": "fear",
        "HAP": "happy",
        "NEU": "neutral",
        "SAD": "sad",
    }

    files: list[str] = []
    labels: list[int] = []
    for wav_path in root.rglob("*.wav"):
        parts = wav_path.stem.split("_")
        if len(parts) < 3:
            continue
        code = parts[2].upper()
        emotion = code_to_emotion.get(code)
        if emotion is None:
            continue
        files.append(str(wav_path))
        labels.append(_to_fer_index(emotion))

    return files, labels


def collect_tess(root_dir: str | Path) -> tuple[list[str], list[int]]:
    # TESS filename examples: OAF_angry.wav, YAF_pleasant_surprise.wav
    root = Path(root_dir)
    if not root.exists():
        return [], []

    label_to_emotion = {
        "angry": "angry",
        "disgust": "disgust",
        "fear": "fear",
        "happy": "happy",
        "neutral": "neutral",
        "sad": "sad",
        "pleasant_surprise": "surprise",
    }

    files: list[str] = []
    labels: list[int] = []
    for wav_path in root.rglob("*.wav"):
        name = wav_path.stem.lower()
        emotion = None
        for key, mapped in label_to_emotion.items():
            if name.endswith(f"_{key}"):
                emotion = mapped
                break
        if emotion is None:
            continue
        files.append(str(wav_path))
        labels.append(_to_fer_index(emotion))

    return files, labels


def collect_savee(root_dir: str | Path) -> tuple[list[str], list[int]]:
    # SAVEE filename examples: DC_a01.wav, DC_d01.wav, DC_f01.wav, DC_h01.wav, DC_n01.wav, DC_sa01.wav, DC_su01.wav
    root = Path(root_dir)
    if not root.exists():
        return [], []

    code_to_emotion = {
        "a": "angry",
        "d": "disgust",
        "f": "fear",
        "h": "happy",
        "n": "neutral",
        "sa": "sad",
        "su": "surprise",
    }

    files: list[str] = []
    labels: list[int] = []
    for wav_path in root.rglob("*.wav"):
        m = re.search(r"_([a-z]{1,2})\d+", wav_path.stem.lower())
        if not m:
            continue
        code = m.group(1)
        emotion = code_to_emotion.get(code)
        if emotion is None:
            continue
        files.append(str(wav_path))
        labels.append(_to_fer_index(emotion))

    return files, labels


def build_ser_multicorpus_loaders(
    ravdess_dir: str | Path | None = None,
    crema_d_dir: str | Path | None = None,
    tess_dir: str | Path | None = None,
    savee_dir: str | Path | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
) -> SERDataBundle:
    all_files: list[str] = []
    all_labels: list[int] = []

    collectors = [
        (collect_ravdess, ravdess_dir),
        (collect_crema_d, crema_d_dir),
        (collect_tess, tess_dir),
        (collect_savee, savee_dir),
    ]
    for collector, src in collectors:
        if not src:
            continue
        files, labels = collector(src)
        all_files.extend(files)
        all_labels.extend(labels)

    if not all_files:
        raise RuntimeError("No SER audio files found. Provide at least one dataset path.")

    train_files, temp_files, train_labels, temp_labels = train_test_split(
        all_files, all_labels, test_size=0.30, random_state=seed, stratify=all_labels
    )
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels, test_size=0.50, random_state=seed, stratify=temp_labels
    )

    train_ds = SERMultiCorpusDataset(train_files, train_labels)
    val_ds = SERMultiCorpusDataset(val_files, val_labels)
    test_ds = SERMultiCorpusDataset(test_files, test_labels)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    class_counts: dict[str, int] = {name: 0 for name in FER_EMOTIONS}
    for idx in all_labels:
        class_counts[FER_EMOTIONS[idx]] += 1

    return SERDataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_counts=class_counts,
    )
