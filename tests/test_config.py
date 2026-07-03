"""Tests for Quick Trainer."""

from pathlib import Path

import yaml

from quick_trainer.config import QuickTrainerConfig, load_config


def test_load_example_config():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "example.yaml")
    assert config.base_model
    assert len(config.datasets) >= 1
    assert config.training.use_lora is True


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
