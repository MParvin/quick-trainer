"""Shared utilities."""

from __future__ import annotations

import os

from quick_trainer.config import QuickTrainerConfig


def resolve_hf_token(config: QuickTrainerConfig) -> str | None:
    return (
        config.hf_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )


def cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False
