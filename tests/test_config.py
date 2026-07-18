"""Tests for Quick Trainer config and safety helpers."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from quick_trainer.config import (
    OllamaConfig,
    QuickTrainerConfig,
    TrainingConfig,
    config_for_display,
    load_config,
)
from quick_trainer.safety import (
    sanitize_gguf_path,
    sanitize_modelfile_text,
    sanitize_output_dir,
    validate_config_path,
    validate_ollama_model_name,
)

try:
    from quick_trainer.trainer import _build_training_args_and_sft_kwargs

    HAS_TRL = True
except ImportError:
    HAS_TRL = False


def test_load_example_config():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "example.yaml")
    assert config.base_model
    assert len(config.datasets) >= 1
    assert config.training.use_lora is True
    assert config.trust_remote_code is False


def test_cli_style_minimal_config():
    raw = {
        "base_model": "gpt2",
        "datasets": [{"path": "imdb", "split": "train", "text_field": "text"}],
        "huggingface": {"enabled": False},
        "ollama": {"enabled": False},
    }
    config = QuickTrainerConfig.model_validate(raw)
    assert config.base_model == "gpt2"


def test_config_roundtrip():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "example.yaml")
    dumped = yaml.safe_load(yaml.dump(config.model_dump(mode="json")))
    restored = QuickTrainerConfig.model_validate(dumped)
    assert restored.base_model == config.base_model


def test_load_golang_dev_config():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "golang-dev.yaml")
    assert config.base_model == "Qwen/Qwen3-8B"
    assert config.datasets[0].messages_field == "messages"
    assert config.datasets[1].dataset_config == "go"
    assert config.datasets[1].code_field == "code"
    assert config.datasets[1].code_language == "go"


def test_load_ci_smoke_config():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "ci-smoke.yaml")
    assert config.base_model == "gpt2"
    assert config.huggingface.enabled is False


def test_rejects_path_traversal_in_output_dir():
    with pytest.raises(ValidationError, match="output_dir must not contain"):
        TrainingConfig(output_dir="../escape")


def test_rejects_invalid_ollama_model_name():
    with pytest.raises(ValidationError, match="Invalid Ollama model name"):
        OllamaConfig(model_name="; rm -rf /")


def test_validate_ollama_model_name_accepts_namespace():
    assert validate_ollama_model_name("user/my-model:tag") == "user/my-model:tag"


def test_sanitize_output_dir_rejects_null_bytes():
    with pytest.raises(ValueError, match="null bytes"):
        sanitize_output_dir("output\0/evil")


def test_config_for_display_redacts_token():
    config = QuickTrainerConfig.model_validate(
        {
            "base_model": "gpt2",
            "datasets": [{"path": "x"}],
            "hf_token": "hf_SECRETTOKEN123",
            "huggingface": {"enabled": False},
            "ollama": {"enabled": False},
        }
    )
    displayed = config_for_display(config)
    assert displayed["hf_token"] == "***REDACTED***"
    assert "hf_SECRETTOKEN123" not in str(displayed)


def test_rejects_modelfile_injection_in_system_prompt():
    with pytest.raises(ValidationError, match="unsafe in an Ollama Modelfile"):
        OllamaConfig(system_prompt='evil"""\nFROM /etc/passwd')


def test_rejects_gguf_path_with_dotdot():
    with pytest.raises(ValidationError, match="gguf_path"):
        OllamaConfig(gguf_path="../escape.gguf")


def test_sanitize_gguf_path_enforces_allow_root(tmp_path: Path):
    allowed = tmp_path / "model.gguf"
    allowed.write_text("x", encoding="utf-8")
    assert sanitize_gguf_path(str(allowed), allow_root=tmp_path) == allowed.resolve()
    with pytest.raises(ValueError, match="outside allowed directory"):
        sanitize_gguf_path("/etc/passwd", allow_root=tmp_path)


def test_sanitize_modelfile_text_rejects_newlines():
    with pytest.raises(ValueError, match="unsafe"):
        sanitize_modelfile_text("line1\nline2", field_name="system_prompt")


def test_validate_config_path_allowlist():
    assert validate_config_path("configs/ci-smoke.yaml") == "configs/ci-smoke.yaml"
    with pytest.raises(ValueError, match="Invalid config path"):
        validate_config_path("../secrets.yaml")
    with pytest.raises(ValueError, match="Invalid config path"):
        validate_config_path("configs/../../etc/passwd")


def test_build_training_args_adapts_to_installed_trl():
    if not HAS_TRL:
        pytest.skip("trl not installed")

    training_cfg = TrainingConfig(max_seq_length=128)
    output_dir = Path("/tmp/quick-trainer-test-output")
    args, extra_kwargs = _build_training_args_and_sft_kwargs(training_cfg, output_dir)

    assert args.output_dir == str(output_dir)
    seq_len = getattr(args, "max_length", None) or getattr(args, "max_seq_length", None)
    if seq_len is not None:
        assert seq_len == 128
    else:
        assert extra_kwargs.get("max_seq_length") == 128
    assert extra_kwargs == {} or extra_kwargs.get("dataset_text_field") == "text"
