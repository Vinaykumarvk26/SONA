from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from app.ml.constants import FER_EMOTIONS


class FER2013Dataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.reset_index(drop=True)
        self.transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        pixels = np.fromstring(row["pixels"], dtype=np.uint8, sep=" ").reshape(48, 48)
        img = Image.fromarray(pixels, mode="L")
        x = self.transform(img)
        y = int(row["emotion"])
        return x, y


@dataclass
class FERDataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


def _split_dataframe(df: pd.DataFrame, seed: int):
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=seed, stratify=df["emotion"])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=seed, stratify=temp_df["emotion"])
    return train_df, val_df, test_df


def build_fer2013_loaders(csv_path: str | Path, batch_size: int = 32, num_workers: int = 0, seed: int = 42) -> FERDataBundle:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"FER2013 CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"emotion", "pixels"}
    if not required.issubset(df.columns):
        raise ValueError("FER2013 CSV must contain 'emotion' and 'pixels' columns")

    train_df, val_df, test_df = _split_dataframe(df, seed)

    train_ds = FER2013Dataset(train_df)
    val_ds = FER2013Dataset(val_df)
    test_ds = FER2013Dataset(test_df)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return FERDataBundle(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)


def fer_class_names() -> list[str]:
    return FER_EMOTIONS
