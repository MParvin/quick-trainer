"""Input validation helpers for config and filesystem paths."""

from __future__ import annotations

import re
from pathlib import Path

OLLAMA_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")


def validate_ollama_model_name(name: str) -> str:
    """Validate an Ollama model name against a safe allowlist."""
    if not OLLAMA_MODEL_NAME_RE.fullmatch(name):
        raise ValueError(
            f"Invalid Ollama model name {name!r}. "
            "Use letters, digits, and . _ : / - only; must start with a letter or digit."
        )
    return name


def sanitize_output_dir(path_str: str) -> Path:
    """Reject unsafe path components and return a resolved absolute path."""
    if "\0" in path_str:
        raise ValueError("output_dir must not contain null bytes")
    path = Path(path_str)
    if ".." in path.parts:
        raise ValueError("output_dir must not contain '..' components")
    return path.resolve()
