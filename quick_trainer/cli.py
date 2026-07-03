"""Command-line interface for Quick Trainer."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Optional

import typer
import yaml

from quick_trainer.config import (
    DatasetConfig,
    HuggingFaceUploadConfig,
    OllamaConfig,
    QuickTrainerConfig,
    TrainingConfig,
    load_config,
)
from quick_trainer.pipeline import run_pipeline

app = typer.Typer(
    name="quick-trainer",
    help="Fine-tune Hugging Face models and publish to HF Hub and/or Ollama.",
    no_args_is_help=True,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def _parse_dataset_specs(specs: list[str]) -> list[DatasetConfig]:
    """Parse dataset specs like 'databricks/databricks-dolly-15k' or 'path:file.jsonl'."""
    datasets: list[DatasetConfig] = []
    for spec in specs:
        if ":" in spec and not spec.startswith("http"):
            path, split = spec.rsplit(":", 1)
            datasets.append(DatasetConfig(path=path, split=split))
        else:
            datasets.append(DatasetConfig(path=spec))
    return datasets


def _build_config_from_args(
    *,
    config_path: Path | None,
    base_model: str | None,
    datasets: list[str] | None,
    output_dir: str | None,
    epochs: int | None,
    batch_size: int | None,
    learning_rate: float | None,
    max_seq_length: int | None,
    hf_repo: str | None,
    hf_private: bool,
    no_hf_upload: bool,
    ollama_model: str | None,
    ollama_push: bool,
    no_ollama: bool,
    hf_token: str | None,
) -> QuickTrainerConfig:
    if config_path:
        config = load_config(config_path)
    else:
        if not base_model:
            raise typer.BadParameter("Provide --config or --base-model.")
        if not datasets:
            raise typer.BadParameter("Provide --config or at least one --dataset.")
        config = QuickTrainerConfig(
            base_model=base_model,
            datasets=_parse_dataset_specs(datasets),
        )

    updates: dict = {}
    if base_model:
        updates["base_model"] = base_model
    if datasets:
        updates["datasets"] = _parse_dataset_specs(datasets)
    if hf_token:
        updates["hf_token"] = hf_token

    training_updates: dict = {}
    if output_dir:
        training_updates["output_dir"] = output_dir
    if epochs is not None:
        training_updates["num_train_epochs"] = epochs
    if batch_size is not None:
        training_updates["per_device_train_batch_size"] = batch_size
    if learning_rate is not None:
        training_updates["learning_rate"] = learning_rate
    if max_seq_length is not None:
        training_updates["max_seq_length"] = max_seq_length

    if training_updates:
        merged_training = config.training.model_dump()
        merged_training.update(training_updates)
        updates["training"] = TrainingConfig.model_validate(merged_training)

    if no_hf_upload:
        updates["huggingface"] = HuggingFaceUploadConfig(enabled=False)
    elif hf_repo:
        hf = config.huggingface.model_dump() if config.huggingface else {}
        hf.update({"enabled": True, "repo_id": hf_repo, "private": hf_private})
        updates["huggingface"] = HuggingFaceUploadConfig.model_validate(hf)

    if no_ollama:
        updates["ollama"] = OllamaConfig(enabled=False)
    elif ollama_model or ollama_push:
        ol = config.ollama.model_dump() if config.ollama else {}
        ol["enabled"] = True
        if ollama_model:
            ol["model_name"] = ollama_model
        if ollama_push:
            ol["push"] = True
        updates["ollama"] = OllamaConfig.model_validate(ol)

    if updates:
        merged = config.model_dump()
        for key, value in updates.items():
            if hasattr(value, "model_dump"):
                merged[key] = value.model_dump()
            else:
                merged[key] = value
        config = QuickTrainerConfig.model_validate(merged)

    return config


@app.command("train")
def train(
    config: Annotated[
        Optional[Path],
        typer.Option("--config", "-c", help="Path to YAML config file"),
    ] = None,
    base_model: Annotated[
        Optional[str],
        typer.Option("--base-model", "-m", help="Hugging Face base model id"),
    ] = None,
    dataset: Annotated[
        Optional[list[str]],
        typer.Option("--dataset", "-d", help="Dataset id or local path (repeatable)"),
    ] = None,
    output_dir: Annotated[
        Optional[str],
        typer.Option("--output-dir", "-o", help="Training output directory"),
    ] = None,
    epochs: Annotated[Optional[int], typer.Option("--epochs", help="Training epochs")] = None,
    batch_size: Annotated[
        Optional[int], typer.Option("--batch-size", help="Per-device batch size")
    ] = None,
    learning_rate: Annotated[
        Optional[float], typer.Option("--learning-rate", help="Learning rate")
    ] = None,
    max_seq_length: Annotated[
        Optional[int], typer.Option("--max-seq-length", help="Max sequence length")
    ] = None,
    hf_repo: Annotated[
        Optional[str],
        typer.Option("--hf-repo", help="Hugging Face target repo id"),
    ] = None,
    hf_private: Annotated[
        bool, typer.Option("--hf-private", help="Create private HF repo")
    ] = False,
    no_hf_upload: Annotated[
        bool, typer.Option("--no-hf-upload", help="Skip Hugging Face upload")
    ] = False,
    ollama_model: Annotated[
        Optional[str],
        typer.Option("--ollama-model", help="Ollama model name to create"),
    ] = None,
    ollama_push: Annotated[
        bool, typer.Option("--ollama-push", help="Push model to Ollama registry after create")
    ] = False,
    no_ollama: Annotated[bool, typer.Option("--no-ollama", help="Skip Ollama publish")] = False,
    hf_token: Annotated[
        Optional[str],
        typer.Option("--hf-token", help="Hugging Face token (prefer HF_TOKEN env)"),
    ] = None,
    skip_download: Annotated[
        bool,
        typer.Option(
            "--skip-download",
            help="Skip pre-download step (model still loads during training)",
        ),
    ] = False,
) -> None:
    """Run the full fine-tuning pipeline."""
    cfg = _build_config_from_args(
        config_path=config,
        base_model=base_model,
        datasets=dataset,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_seq_length=max_seq_length,
        hf_repo=hf_repo,
        hf_private=hf_private,
        no_hf_upload=no_hf_upload,
        ollama_model=ollama_model,
        ollama_push=ollama_push,
        no_ollama=no_ollama,
        hf_token=hf_token,
    )

    typer.echo(f"Base model: {cfg.base_model}")
    typer.echo(f"Datasets: {[d.path for d in cfg.datasets]}")
    results = run_pipeline(cfg, skip_download=skip_download)
    typer.echo(json.dumps(results, indent=2))


@app.command("validate")
def validate_config(
    config: Annotated[Path, typer.Argument(help="Path to YAML config file")],
) -> None:
    """Validate a config file without running training."""
    cfg = load_config(config)
    typer.echo("Config is valid.")
    typer.echo(yaml.dump(cfg.model_dump(mode="json"), default_flow_style=False, sort_keys=False))


if __name__ == "__main__":
    app()
