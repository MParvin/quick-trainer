"""CLI-focused tests."""

import os
from pathlib import Path

from typer.testing import CliRunner

from quick_trainer.cli import app

runner = CliRunner()
ROOT = Path(__file__).resolve().parents[1]


def test_validate_redacts_hf_token(tmp_path: Path):
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        """
base_model: gpt2
datasets:
  - path: dummy
hf_token: hf_SECRETTOKEN123
huggingface:
  enabled: false
ollama:
  enabled: false
""",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["validate", str(config_path)])
    assert result.exit_code == 0, result.output
    assert "Config is valid." in result.output
    assert "***REDACTED***" in result.output
    assert "hf_SECRETTOKEN123" not in result.output


def test_validate_require_repo_path_rejects_escape(tmp_path: Path):
    evil = tmp_path / "evil.yaml"
    evil.write_text(
        """
base_model: gpt2
datasets:
  - path: dummy
huggingface:
  enabled: false
ollama:
  enabled: false
""",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["validate", "--require-repo-path", str(evil)])
    assert result.exit_code != 0
    assert "Invalid config path" in result.output or "Invalid config path" in str(result.exception)


def test_validate_example_config():
    previous = Path.cwd()
    os.chdir(ROOT)
    try:
        result = runner.invoke(app, ["validate", "--require-repo-path", "configs/example.yaml"])
    finally:
        os.chdir(previous)
    assert result.exit_code == 0, result.output
    assert "Config is valid." in result.output
