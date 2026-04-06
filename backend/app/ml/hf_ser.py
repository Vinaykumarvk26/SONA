from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

from app.ml.constants import FER_EMOTIONS

try:
    from safetensors.torch import load_file as safe_load_file
except Exception:  # pragma: no cover - safetensors is optional at runtime
    safe_load_file = None


logger = logging.getLogger("hf_ser")

LABEL_MAP = {
    "ang": "angry",
    "anger": "angry",
    "angry": "angry",
    "calm": "neutral",
    "dis": "disgust",
    "disgust": "disgust",
    "fea": "fear",
    "fear": "fear",
    "fearful": "fear",
    "hap": "happy",
    "happy": "happy",
    "joy": "happy",
    "neu": "neutral",
    "neutral": "neutral",
    "sad": "sad",
    "sadness": "sad",
    "surprise": "surprise",
    "surprised": "surprise",
}


def _normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    z = np.array([max(1e-9, float(scores.get(e, 0.0))) for e in FER_EMOTIONS], dtype=np.float64)
    z = z / z.sum()
    return {e: float(v) for e, v in zip(FER_EMOTIONS, z.tolist())}


def _remap_checkpoint_keys(state_dict: dict) -> dict:
    # Many public wav2vec2 SER checkpoints were trained with older transformers heads.
    # This remaps legacy keys to current Wav2Vec2ForSequenceClassification names.
    remapped = dict(state_dict)
    key_map = {
        "classifier.dense.weight": "projector.weight",
        "classifier.dense.bias": "projector.bias",
        "classifier.out_proj.weight": "classifier.weight",
        "classifier.out_proj.bias": "classifier.bias",
        "classifier.output.weight": "classifier.weight",
        "classifier.output.bias": "classifier.bias",
        "wav2vec2.encoder.pos_conv_embed.conv.weight_g": "wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original0",
        "wav2vec2.encoder.pos_conv_embed.conv.weight_v": "wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original1",
    }
    for src, dst in key_map.items():
        if src in remapped:
            remapped[dst] = remapped.pop(src)
    return remapped


class HFWav2VecSER:
    def __init__(
        self,
        model_id: str,
        device: torch.device,
        local_files_only: bool = False,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.local_files_only = local_files_only

        self.feature_extractor = None
        self.model = None
        self.id2label: dict[int, str] = {}

        self._load()

    def _download_state_dict(self) -> dict:
        errors: list[str] = []

        try:
            bin_path = hf_hub_download(
                repo_id=self.model_id,
                filename="pytorch_model.bin",
                local_files_only=self.local_files_only,
            )
            return torch.load(bin_path, map_location="cpu")
        except Exception as exc:  # noqa: BLE001 - aggregate fallback errors
            errors.append(f"pytorch_model.bin: {exc}")

        try:
            if safe_load_file is None:
                raise RuntimeError("safetensors package is not installed")
            sf_path = hf_hub_download(
                repo_id=self.model_id,
                filename="model.safetensors",
                local_files_only=self.local_files_only,
            )
            return safe_load_file(sf_path, device="cpu")
        except Exception as exc:  # noqa: BLE001 - aggregate fallback errors
            errors.append(f"model.safetensors: {exc}")

        raise RuntimeError("Unable to load HF SER checkpoint. " + " | ".join(errors))

    def _load(self) -> None:
        config = AutoConfig.from_pretrained(self.model_id, local_files_only=self.local_files_only)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.model_id,
            local_files_only=self.local_files_only,
        )
        state_dict = self._download_state_dict()

        # Align classifier projection width with legacy checkpoint head if needed.
        if "classifier.dense.weight" in state_dict:
            proj_dim = int(state_dict["classifier.dense.weight"].shape[0])
            config.classifier_proj_size = proj_dim

        model = Wav2Vec2ForSequenceClassification(config)
        remapped = _remap_checkpoint_keys(state_dict)
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        if missing:
            logger.warning("HF SER missing keys while loading %s: %s", self.model_id, missing[:8])
        if unexpected:
            logger.warning("HF SER unexpected keys while loading %s: %s", self.model_id, unexpected[:8])

        model = model.to(self.device).eval()
        self.model = model

        raw_id2label = model.config.id2label or {}
        id2label: dict[int, str] = {}
        for key, value in raw_id2label.items():
            try:
                idx = int(key)
            except Exception:
                continue
            id2label[idx] = str(value)
        self.id2label = id2label
        logger.info("Loaded HF SER model '%s' with labels: %s", self.model_id, self.id2label)

    def predict_scores(self, audio: np.ndarray, sample_rate: int) -> Dict[str, float]:
        if self.model is None or self.feature_extractor is None:
            raise RuntimeError("HF SER model is not loaded")

        wave = np.asarray(audio, dtype=np.float32)
        if wave.ndim > 1:
            wave = wave.mean(axis=1)
        if wave.size == 0:
            raise ValueError("Voice input is empty")

        inputs = self.feature_extractor(
            wave,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        tensor_inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**tensor_inputs).logits.squeeze(0)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

        scores: Dict[str, float] = {emotion: 0.0 for emotion in FER_EMOTIONS}
        for idx, prob in enumerate(probs.tolist()):
            raw_label = self.id2label.get(idx, "neutral").lower().strip()
            mapped = LABEL_MAP.get(raw_label, raw_label)
            if mapped in scores:
                scores[mapped] += float(prob)

        return _normalize_scores(scores)
