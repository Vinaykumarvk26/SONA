import logging
from pathlib import Path

import requests
import torch


logger = logging.getLogger("model_loader")


def _download_file(url: str, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        with open(target_path, "wb") as file_obj:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file_obj.write(chunk)


def ensure_model_checkpoint(model: torch.nn.Module, checkpoint_path: str, download_url: str = "") -> str:
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        if download_url:
            logger.info("Downloading model checkpoint from %s", download_url)
            _download_file(download_url, path)
        else:
            raise FileNotFoundError(
                f"Required checkpoint is missing at {path}. "
                f"Provide the file or set a download URL."
            )

    state_dict = torch.load(path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)
    return str(path)
