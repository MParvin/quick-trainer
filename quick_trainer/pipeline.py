"""End-to-end training pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

from quick_trainer.config import QuickTrainerConfig
from quick_trainer.downloader import download_base_model
from quick_trainer.export import merge_lora_adapter
from quick_trainer.ollama import publish_to_ollama
from quick_trainer.trainer import fine_tune
from quick_trainer.uploader import upload_to_huggingface

logger = logging.getLogger(__name__)


def run_pipeline(config: QuickTrainerConfig, *, skip_download: bool = False) -> dict[str, str]:
    """Download, train, and publish according to config."""
    results: dict[str, str] = {"output_dir": config.training.output_dir}

    if not skip_download:
        download_base_model(config)
    else:
        logger.info("Skipping explicit model download (trainer loads on demand).")

    output_dir = fine_tune(config)
    results["output_dir"] = str(output_dir)

    export_dir = merge_lora_adapter(output_dir, config)
    results["export_dir"] = str(export_dir)

    hf_url = upload_to_huggingface(export_dir, config)
    if hf_url:
        results["huggingface_url"] = hf_url

    ollama_name = publish_to_ollama(export_dir, config)
    if ollama_name:
        results["ollama_model"] = ollama_name

    return results
