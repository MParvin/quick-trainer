"""Input validation helpers for config and filesystem paths."""

from __future__ import annotations

import re
from pathlib import Path

OLLAMA_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")
CONFIG_PATH_RE = re.compile(r"^configs/[A-Za-z0-9][A-Za-z0-9._/-]*\.ya?ml$")
_UNSAFE_MODELFILE_CHARS = re.compile(r'["\n\r\x00]')


def validate_ollama_model_name(name: str) -> str:
    """Validate an Ollama model name against a safe allowlist."""
    if not OLLAMA_MODEL_NAME_RE.fullmatch(name):
        raise ValueError(
            f"Invalid Ollama model name {name!r}. "
            "Use letters, digits, and . _ : / - only; must start with a letter or digit."
        )
    return name


def validate_config_path(path_str: str) -> str:
    """Allow only repo-relative YAML paths under configs/."""
    if not CONFIG_PATH_RE.fullmatch(path_str):
        raise ValueError(
            f"Invalid config path {path_str!r}. "
            "Expected a path like configs/example.yaml under configs/."
        )
    return path_str


def sanitize_output_dir(path_str: str) -> Path:
    """Reject unsafe path components and return a resolved absolute path."""
    if "\0" in path_str:
        raise ValueError("output_dir must not contain null bytes")
    path = Path(path_str)
    if ".." in path.parts:
        raise ValueError("output_dir must not contain '..' components")
    return path.resolve()


def sanitize_modelfile_text(value: str, *, field_name: str) -> str:
    """Reject characters that can break Ollama Modelfile triple-quoted blocks."""
    if '"""' in value or _UNSAFE_MODELFILE_CHARS.search(value):
        raise ValueError(
            f"{field_name} contains characters that are unsafe in an Ollama Modelfile "
            "(disallowed: triple quotes, double quotes, newlines, null bytes)."
        )
    return value


def sanitize_gguf_path(path_str: str, *, allow_root: Path | None = None) -> Path:
    """Resolve a GGUF path and optionally require it under allow_root."""
    if "\0" in path_str:
        raise ValueError("gguf_path must not contain null bytes")
    path = Path(path_str)
    if ".." in path.parts:
        raise ValueError("gguf_path must not contain '..' components")
    resolved = path.resolve()
    if allow_root is not None:
        root = allow_root.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"gguf_path {resolved} is outside allowed directory {root}")
    return resolved
