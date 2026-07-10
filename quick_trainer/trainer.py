"""Fine-tune a base model with configured datasets."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments
from trl import SFTTrainer

from quick_trainer.config import QuickTrainerConfig, TrainingConfig
from quick_trainer.downloader import load_model_for_training, load_tokenizer, load_training_datasets
from quick_trainer.safety import sanitize_output_dir
from quick_trainer.utils import cuda_available

logger = logging.getLogger(__name__)


def _default_target_modules(model) -> list[str]:
    import torch.nn as nn

    names: set[str] = set()
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            names.add(module.__class__.__name__)
    # Prefer projection layer suffixes used by most causal LMs.
    preferred = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    found = preferred.intersection(
        {n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    )
    if found:
        return sorted(found)
    # GPT-2 style
    gpt2 = {"c_attn", "c_proj", "c_fc"}
    found_gpt2 = gpt2.intersection(
        {n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    )
    if found_gpt2:
        return sorted(found_gpt2)
    return sorted(names) or ["q_proj", "v_proj"]


def _build_lora_config(config: QuickTrainerConfig, model) -> LoraConfig:
    lora = config.training.lora
    target_modules = lora.target_modules or _default_target_modules(model)
    return LoraConfig(
        r=lora.r,
        lora_alpha=lora.lora_alpha,
        lora_dropout=lora.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )


def _common_training_kwargs(training_cfg: TrainingConfig, output_dir: Path) -> dict[str, Any]:
    return dict(
        output_dir=str(output_dir),
        num_train_epochs=training_cfg.num_train_epochs,
        per_device_train_batch_size=training_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        learning_rate=training_cfg.learning_rate,
        warmup_ratio=training_cfg.warmup_ratio,
        logging_steps=training_cfg.logging_steps,
        save_steps=training_cfg.save_steps,
        save_total_limit=2,
        optim="paged_adamw_8bit"
        if (training_cfg.load_in_4bit and cuda_available())
        else "adamw_torch",
        bf16=training_cfg.bf16 and cuda_available(),
        report_to="none",
        seed=training_cfg.seed,
        remove_unused_columns=False,
    )


def _build_training_args_and_sft_kwargs(
    training_cfg: TrainingConfig,
    output_dir: Path,
) -> tuple[TrainingArguments, dict[str, Any]]:
    """Build TRL training args, adapting to installed SFTConfig field names."""
    common_kwargs = _common_training_kwargs(training_cfg, output_dir)

    try:
        from trl import SFTConfig

        sft_fields = SFTConfig.__dataclass_fields__
        config_kwargs = dict(common_kwargs)

        if "dataset_text_field" in sft_fields:
            config_kwargs["dataset_text_field"] = "text"
        if "max_length" in sft_fields:
            config_kwargs["max_length"] = training_cfg.max_seq_length
        elif "max_seq_length" in sft_fields:
            config_kwargs["max_seq_length"] = training_cfg.max_seq_length

        return SFTConfig(**config_kwargs), {}
    except ImportError:
        return TrainingArguments(**common_kwargs), {
            "dataset_text_field": "text",
            "max_seq_length": training_cfg.max_seq_length,
        }


def fine_tune(config: QuickTrainerConfig) -> Path:
    """Run LoRA (or full) SFT training and return output directory."""
    output_dir = sanitize_output_dir(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading datasets...")
    dataset: Dataset = load_training_datasets(config)

    logger.info("Loading tokenizer and model...")
    tokenizer = load_tokenizer(config)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model_for_training(config)
    training_cfg = config.training

    if training_cfg.use_lora:
        if training_cfg.load_in_4bit and cuda_available():
            model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, _build_lora_config(config, model))
        model.print_trainable_parameters()

    training_args, extra_sft_kwargs = _build_training_args_and_sft_kwargs(training_cfg, output_dir)

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=dataset,
        **extra_sft_kwargs,
    )
    try:
        trainer = SFTTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = SFTTrainer(tokenizer=tokenizer, **trainer_kwargs)

    logger.info("Starting training...")
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    logger.info("Training complete. Artifacts saved to %s", output_dir)
    return output_dir
