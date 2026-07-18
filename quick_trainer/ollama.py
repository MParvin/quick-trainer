"""Publish fine-tuned models to Ollama."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from quick_trainer.config import OllamaConfig, QuickTrainerConfig
from quick_trainer.safety import (
    OLLAMA_MODEL_NAME_RE,
    sanitize_gguf_path,
    sanitize_modelfile_text,
    validate_ollama_model_name,
)

logger = logging.getLogger(__name__)

_ALLOWED_OLLAMA_COMMANDS = frozenset({"create", "push"})


def _run_ollama(ollama_bin: str, subcommand: str, model_name: str, *extra: str) -> None:
    """Run a validated Ollama CLI command without shell interpolation."""
    if subcommand not in _ALLOWED_OLLAMA_COMMANDS:
        raise ValueError(f"Unsupported Ollama subcommand: {subcommand!r}")
    if not OLLAMA_MODEL_NAME_RE.fullmatch(model_name):
        raise ValueError(f"Invalid Ollama model name: {model_name!r}")

    cmd = [ollama_bin, subcommand, model_name, *extra]
    logger.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )


def _ensure_ollama() -> str:
    path = shutil.which("ollama")
    if not path:
        raise RuntimeError(
            "Ollama CLI not found. Install from https://ollama.com and ensure `ollama` is on PATH."
        )
    return path


def _write_modelfile(
    model_dir: Path,
    cfg: OllamaConfig,
    *,
    allow_root: Path | None = None,
) -> Path:
    modelfile = model_dir / "Modelfile"
    lines: list[str] = []

    if cfg.gguf_path:
        gguf = sanitize_gguf_path(cfg.gguf_path, allow_root=allow_root)
        lines.append(f'FROM "{gguf}"')
    else:
        lines.append(f'FROM "{model_dir.resolve()}"')

    if cfg.system_prompt:
        system_prompt = sanitize_modelfile_text(cfg.system_prompt, field_name="system_prompt")
        lines.append(f'SYSTEM """{system_prompt}"""')

    template = sanitize_modelfile_text(cfg.base_template, field_name="base_template")
    lines.append(f'TEMPLATE """{template}"""')
    lines.append('PARAMETER stop "### Instruction:"')
    lines.append('PARAMETER stop "### Response:"')

    modelfile.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote Modelfile at %s", modelfile)
    return modelfile


def publish_to_ollama(
    model_dir: Path,
    config: QuickTrainerConfig,
    ollama_cfg: OllamaConfig | None = None,
) -> str:
    """Create (and optionally push) an Ollama model from training output."""
    ollama_cfg = ollama_cfg or config.ollama
    if ollama_cfg is None or not ollama_cfg.enabled:
        logger.info("Ollama publish disabled; skipping.")
        return ""

    ollama_bin = _ensure_ollama()
    model_name = validate_ollama_model_name(ollama_cfg.model_name)
    allow_root = Path.cwd()
    _write_modelfile(model_dir, ollama_cfg, allow_root=allow_root)

    logger.info("Creating Ollama model: %s", model_name)
    _run_ollama(ollama_bin, "create", model_name, "-f", str(model_dir / "Modelfile"))

    if ollama_cfg.push:
        logger.info("Pushing Ollama model: %s", model_name)
        _run_ollama(ollama_bin, "push", model_name)

    logger.info("Ollama model ready: %s", model_name)
    return model_name
