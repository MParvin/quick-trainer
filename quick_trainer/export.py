"""Merge LoRA adapters into a standalone model for export."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from quick_trainer.config import QuickTrainerConfig
from quick_trainer.utils import resolve_hf_token

logger = logging.getLogger(__name__)


def merge_lora_adapter(model_dir: Path, config: QuickTrainerConfig) -> Path:
    """Merge LoRA weights into the base model and write to `model_dir/merged`."""
    merged_dir = model_dir / "merged"
    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    merged_dir.mkdir(parents=True)

    if not config.training.use_lora:
        logger.info("LoRA disabled; copying training output for export.")
        for item in model_dir.iterdir():
            if item.name == "merged":
                continue
            dest = merged_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
        return merged_dir

    token = resolve_hf_token(config)
    logger.info("Merging LoRA adapter into base model...")

    base = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        token=token,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="cpu",
    )
    model = PeftModel.from_pretrained(base, str(model_dir))
    merged = model.merge_and_unload()

    merged.save_pretrained(str(merged_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    tokenizer.save_pretrained(str(merged_dir))

    logger.info("Merged model saved to %s", merged_dir)
    return merged_dir
