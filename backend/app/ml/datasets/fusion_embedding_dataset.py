from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class FusionEmbeddingDataset(Dataset):
    """Reads precomputed embeddings saved as .npz with keys: face_emb, speech_emb, labels."""

    def __init__(self, npz_path: str | Path):
        data = np.load(npz_path)
        self.face_emb = torch.tensor(data["face_emb"], dtype=torch.float32)
        self.speech_emb = torch.tensor(data["speech_emb"], dtype=torch.float32)
        self.labels = torch.tensor(data["labels"], dtype=torch.long)

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        return self.face_emb[idx], self.speech_emb[idx], self.labels[idx]
