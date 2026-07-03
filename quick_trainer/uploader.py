"""Upload fine-tuned models to Hugging Face Hub."""

from __future__ import annotations

import logging
from pathlib import Path

from huggingface_hub import HfApi, create_repo

from quick_trainer.config import HuggingFaceUploadConfig, QuickTrainerConfig
from quick_trainer.utils import resolve_hf_token

logger = logging.getLogger(__name__)


def _resolve_token(config: QuickTrainerConfig) -> str:
    token = resolve_hf_token(config)
    if not token:
        raise ValueError(
            "Hugging Face token required for upload. Set HF_TOKEN or pass hf_token in config."
        )
    return token


def upload_to_huggingface(
    model_dir: Path,
    config: QuickTrainerConfig,
    upload_cfg: HuggingFaceUploadConfig | None = None,
) -> str:
    """Push model directory to Hugging Face Hub. Returns repo URL."""
    upload_cfg = upload_cfg or config.huggingface
    if upload_cfg is None or not upload_cfg.enabled:
        logger.info("Hugging Face upload disabled; skipping.")
        return ""

    repo_id = upload_cfg.repo_id
    if not repo_id:
        logger.warning("Hugging Face upload enabled but repo_id is missing; skipping.")
        return ""

    token = _resolve_token(config)

    logger.info("Creating/updating Hugging Face repo: %s", repo_id)
    create_repo(repo_id=repo_id, token=token, private=upload_cfg.private, exist_ok=True)

    api = HfApi(token=token)
    logger.info("Uploading model from %s to %s", model_dir, repo_id)
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=upload_cfg.commit_message,
    )

    url = f"https://huggingface.co/{repo_id}"
    logger.info("Upload complete: %s", url)
    return url
