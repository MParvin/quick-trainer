"""Download models and datasets from Hugging Face."""

from __future__ import annotations

import logging
from pathlib import Path

from datasets import concatenate_datasets, load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from quick_trainer.config import DatasetConfig, QuickTrainerConfig
from quick_trainer.formatting import format_example
from quick_trainer.utils import cuda_available, resolve_hf_token

logger = logging.getLogger(__name__)


def download_base_model(config: QuickTrainerConfig, cache_dir: Path | None = None) -> Path:
    """Download base model weights and return local path."""
    token = resolve_hf_token(config)
    logger.info("Downloading base model: %s", config.base_model)
    local_path = snapshot_download(
        repo_id=config.base_model,
        token=token,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    logger.info("Base model cached at: %s", local_path)
    return Path(local_path)


def load_tokenizer(config: QuickTrainerConfig):
    token = resolve_hf_token(config)
    return AutoTokenizer.from_pretrained(
        config.base_model,
        token=token,
        trust_remote_code=config.trust_remote_code,
    )


def load_model_for_training(config: QuickTrainerConfig):
    """Load model with optional 4-bit quantization for LoRA training."""
    import torch
    from transformers import BitsAndBytesConfig

    token = resolve_hf_token(config)
    training = config.training
    use_4bit = training.use_lora and training.load_in_4bit and cuda_available()
    if training.load_in_4bit and not use_4bit:
        logger.warning("4-bit loading requires CUDA; falling back to full precision.")

    model_kwargs: dict = {
        "token": token,
        "trust_remote_code": config.trust_remote_code,
        "device_map": "auto" if cuda_available() else None,
    }

    if use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if training.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif training.bf16 and cuda_available():
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(config.base_model, **model_kwargs)
    return model


def _load_single_dataset(ds_cfg: DatasetConfig, token: str | None):
    if Path(ds_cfg.path).exists():
        return load_dataset("json", data_files=ds_cfg.path, split="train")
    if ds_cfg.dataset_config:
        return load_dataset(ds_cfg.path, ds_cfg.dataset_config, split=ds_cfg.split, token=token)
    return load_dataset(ds_cfg.path, split=ds_cfg.split, token=token)


def load_training_datasets(config: QuickTrainerConfig, tokenizer=None):
    """Load and merge all configured datasets."""
    token = resolve_hf_token(config)
    if tokenizer is None:
        tokenizer = load_tokenizer(config)
    parts = []

    for ds_cfg in config.datasets:
        label = ds_cfg.path
        if ds_cfg.dataset_config:
            label = f"{ds_cfg.path} ({ds_cfg.dataset_config})"
        logger.info("Loading dataset: %s (split=%s)", label, ds_cfg.split)

        dataset = _load_single_dataset(ds_cfg, token)

        if ds_cfg.max_samples is not None:
            dataset = dataset.select(range(min(ds_cfg.max_samples, len(dataset))))

        dataset = dataset.map(
            lambda row, cfg=ds_cfg: format_example(row, cfg, tokenizer),
            remove_columns=dataset.column_names,
            desc=f"Formatting {label}",
        )
        parts.append(dataset)

    if len(parts) == 1:
        merged = parts[0]
    else:
        merged = concatenate_datasets(parts)
        logger.info("Merged %d datasets -> %d samples", len(parts), len(merged))

    return merged
