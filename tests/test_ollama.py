"""Tests for Ollama Modelfile publishing helpers."""

from pathlib import Path

import pytest

from quick_trainer.config import OllamaConfig
from quick_trainer.ollama import _write_modelfile


def test_write_modelfile_basic(tmp_path: Path):
    cfg = OllamaConfig(
        enabled=True,
        model_name="demo-model",
        system_prompt="You are helpful.",
        base_template="{{ .Prompt }}",
    )
    modelfile = _write_modelfile(tmp_path, cfg, allow_root=tmp_path)
    text = modelfile.read_text(encoding="utf-8")
    assert f'FROM "{tmp_path.resolve()}"' in text
    assert 'SYSTEM """You are helpful."""' in text
    assert 'TEMPLATE """{{ .Prompt }}"""' in text


def test_write_modelfile_rejects_gguf_outside_root(tmp_path: Path):
    cfg = OllamaConfig(enabled=True, model_name="demo-model", gguf_path="/etc/passwd")
    with pytest.raises(ValueError, match="outside allowed directory"):
        _write_modelfile(tmp_path, cfg, allow_root=tmp_path)


def test_write_modelfile_accepts_gguf_under_root(tmp_path: Path):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"gguf")
    cfg = OllamaConfig(enabled=True, model_name="demo-model", gguf_path=str(gguf))
    modelfile = _write_modelfile(tmp_path, cfg, allow_root=tmp_path)
    assert f'FROM "{gguf.resolve()}"' in modelfile.read_text(encoding="utf-8")
