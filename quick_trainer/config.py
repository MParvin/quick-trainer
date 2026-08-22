"""Configuration models for Quick Trainer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from quick_trainer.safety import (
    sanitize_gguf_path,
    sanitize_modelfile_text,
    sanitize_output_dir,
    validate_ollama_model_name,
)


class DatasetConfig(BaseModel):
    """A single training dataset source."""

    path: str = Field(description="Hugging Face dataset id or local path")
    split: str = Field(default="train", description="Dataset split to use")
    dataset_config: str | None = Field(
        default=None,
        description="Hugging Face dataset subset/config name (e.g. 'go' for CodeXGLUE)",
    )
    text_field: str = Field(
        default="text",
        description="Column containing a single flat training string per row",
    )
    messages_field: str | None = Field(
        default=None,
        description="Column with chat messages [{role, content}, ...]; uses model chat template",
    )
    instruction_field: str | None = Field(
        default=None,
        description="Optional instruction column for instruction-tuning format",
    )
    response_field: str | None = Field(
        default=None,
        description="Optional response column paired with instruction_field",
    )
    code_field: str | None = Field(
        default=None,
        description="Code column for code/docstring pairs (use with docstring_field)",
    )
    docstring_field: str | None = Field(
        default=None,
        description="Docstring column for code/docstring pairs (use with code_field)",
    )
    code_language: str = Field(
        default="code",
        description="Language label used in code/docstring formatting (e.g. go, python)",
    )
    max_samples: int | None = Field(
        default=None,
        description="Limit number of samples from this dataset",
    )


class LoRAConfig(BaseModel):
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] | None = None


class TrainingConfig(BaseModel):
    output_dir: str = "./output"

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, value: str) -> str:
        sanitize_output_dir(value)
        return value

    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 512
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 200
    use_lora: bool = True
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    load_in_4bit: bool = True
    bf16: bool = True
    seed: int = 42
    merge_adapter: bool = Field(
        default=True,
        description="Merge LoRA adapters into full weights before publish",
    )


class HuggingFaceUploadConfig(BaseModel):
    enabled: bool = True
    repo_id: str | None = Field(
        default=None,
        description="Target repo, e.g. username/my-finetuned-model",
    )
    private: bool = False
    commit_message: str = "Upload fine-tuned model via Quick Trainer"
    upload_adapter_only: bool = Field(
        default=False,
        description="Upload adapter/checkpoint directory instead of merged full weights",
    )


class OllamaConfig(BaseModel):
    enabled: bool = False
    model_name: str = Field(
        default="quick-trainer-model",
        description="Local Ollama model name",
    )

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        return validate_ollama_model_name(value)

    base_template: str = Field(
        default="{{ .Prompt }}",
        description="Modelfile template line (Go template syntax)",
    )
    system_prompt: str | None = Field(
        default=None,
        description="Optional system prompt for the Modelfile",
    )
    push: bool = Field(
        default=False,
        description="Run `ollama push` after create (requires Ollama account)",
    )
    gguf_path: str | None = Field(
        default=None,
        description=(
            "Optional pre-built GGUF path under the workspace; convert merged weights "
            "externally (e.g. llama.cpp) before setting this"
        ),
    )

    @field_validator("system_prompt")
    @classmethod
    def validate_system_prompt(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return sanitize_modelfile_text(value, field_name="system_prompt")

    @field_validator("base_template")
    @classmethod
    def validate_base_template(cls, value: str) -> str:
        return sanitize_modelfile_text(value, field_name="base_template")

    @field_validator("gguf_path")
    @classmethod
    def validate_gguf_path(cls, value: str | None) -> str | None:
        if value is None:
            return value
        sanitize_gguf_path(value)
        return value


class EvaluationConfig(BaseModel):
    """Optional post-train smoke generation."""

    enabled: bool = False
    prompt: str = Field(
        default="### Instruction:\nSay hello in one short sentence.\n\n### Response:\n",
        description="Prompt used for a short smoke generation after training",
    )
    max_new_tokens: int = 32


class QuickTrainerConfig(BaseModel):
    """Root configuration for a training run."""

    base_model: str = Field(description="Hugging Face model id or local path")
    datasets: list[DatasetConfig] = Field(min_length=1)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    huggingface: HuggingFaceUploadConfig | None = Field(default_factory=HuggingFaceUploadConfig)
    ollama: OllamaConfig | None = Field(default_factory=OllamaConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    trust_remote_code: bool = Field(
        default=False,
        description="Allow executing remote code from the Hub model repo (unsafe; opt-in)",
    )
    hf_token: str | None = Field(
        default=None,
        description="Hugging Face token (prefer HF_TOKEN env var)",
    )

    @field_validator("datasets")
    @classmethod
    def require_at_least_one_dataset(cls, value: list[DatasetConfig]) -> list[DatasetConfig]:
        if not value:
            raise ValueError("At least one dataset is required")
        return value


def load_config(path: Path) -> QuickTrainerConfig:
    """Load and validate a YAML config file."""
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must be a YAML mapping: {path}")
    return QuickTrainerConfig.model_validate(raw)


def config_for_display(config: QuickTrainerConfig) -> dict[str, Any]:
    """Serialize config for logs/CLI with secrets redacted."""
    data = config.model_dump(mode="json")
    if data.get("hf_token"):
        data["hf_token"] = "***REDACTED***"
    return data
