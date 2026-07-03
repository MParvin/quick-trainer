"""Publish fine-tuned models to Ollama."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from quick_trainer.config import OllamaConfig, QuickTrainerConfig

logger = logging.getLogger(__name__)


def _run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess:
    logger.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return result


def _ensure_ollama() -> str:
    path = shutil.which("ollama")
    if not path:
        raise RuntimeError(
            "Ollama CLI not found. Install from https://ollama.com and ensure `ollama` is on PATH."
        )
    return path


def _write_modelfile(model_dir: Path, cfg: OllamaConfig) -> Path:
    modelfile = model_dir / "Modelfile"
    lines: list[str] = []

    if cfg.gguf_path:
        lines.append(f'FROM "{Path(cfg.gguf_path).resolve()}"')
    else:
        lines.append(f'FROM "{model_dir.resolve()}"')

    if cfg.system_prompt:
        lines.append(f'SYSTEM """{cfg.system_prompt}"""')

    lines.append(f'TEMPLATE """{cfg.base_template}"""')
    lines.append(f'PARAMETER stop "### Instruction:"')
    lines.append(f'PARAMETER stop "### Response:"')

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

    _ensure_ollama()
    _write_modelfile(model_dir, ollama_cfg)

    logger.info("Creating Ollama model: %s", ollama_cfg.model_name)
    _run(["ollama", "create", ollama_cfg.model_name, "-f", str(model_dir / "Modelfile")])

    if ollama_cfg.push:
        logger.info("Pushing Ollama model: %s", ollama_cfg.model_name)
        _run(["ollama", "push", ollama_cfg.model_name])

    logger.info("Ollama model ready: %s", ollama_cfg.model_name)
    return ollama_cfg.model_name
