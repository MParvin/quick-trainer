"""Download models and datasets from Hugging Face."""

from __future__ import annotations

import logging
from pathlib import Path

from datasets import concatenate_datasets, load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from quick_trainer.config import DatasetConfig, QuickTrainerConfig
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
    return AutoTokenizer.from_pretrained(config.base_model, token=token, trust_remote_code=True)


def _cuda_available() -> bool:
    return cuda_available()


def load_model_for_training(config: QuickTrainerConfig):
    """Load model with optional 4-bit quantization for LoRA training."""
    import torch
    from transformers import BitsAndBytesConfig

    token = resolve_hf_token(config)
    training = config.training
    use_4bit = training.use_lora and training.load_in_4bit and _cuda_available()
    if training.load_in_4bit and not use_4bit:
        logger.warning("4-bit loading requires CUDA; falling back to full precision.")

    model_kwargs: dict = {
        "token": token,
        "trust_remote_code": True,
        "device_map": "auto" if _cuda_available() else None,
    }

    if use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if training.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif training.bf16 and _cuda_available():
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(config.base_model, **model_kwargs)
    return model


def _format_example(row: dict, ds_cfg: DatasetConfig) -> dict[str, str]:
    if ds_cfg.instruction_field and ds_cfg.response_field:
        instruction = row.get(ds_cfg.instruction_field, "")
        response = row.get(ds_cfg.response_field, "")
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
    else:
        text = row.get(ds_cfg.text_field, "")
        if not isinstance(text, str):
            text = str(text)
    return {"text": text}


def load_training_datasets(config: QuickTrainerConfig):
    """Load and merge all configured datasets."""
    token = resolve_hf_token(config)
    parts = []

    for ds_cfg in config.datasets:
        logger.info("Loading dataset: %s (split=%s)", ds_cfg.path, ds_cfg.split)
        if Path(ds_cfg.path).exists():
            dataset = load_dataset("json", data_files=ds_cfg.path, split="train")
        else:
            dataset = load_dataset(ds_cfg.path, split=ds_cfg.split, token=token)

        if ds_cfg.max_samples is not None:
            dataset = dataset.select(range(min(ds_cfg.max_samples, len(dataset))))

        dataset = dataset.map(
            lambda row: _format_example(row, ds_cfg),
            remove_columns=dataset.column_names,
            desc=f"Formatting {ds_cfg.path}",
        )
        parts.append(dataset)

    if len(parts) == 1:
        merged = parts[0]
    else:
        merged = concatenate_datasets(parts)
        logger.info("Merged %d datasets -> %d samples", len(parts), len(merged))

    return merged
