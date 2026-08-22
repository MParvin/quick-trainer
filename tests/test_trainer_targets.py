"""Tests for LoRA target-module discovery."""

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from quick_trainer.trainer import _default_target_modules  # noqa: E402


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.v_proj = nn.Linear(4, 4)
        self.other = nn.Linear(4, 4)


def test_default_target_modules_prefers_projection_suffixes():
    modules = _default_target_modules(_TinyModel())
    assert modules == ["q_proj", "v_proj"]
    assert "Linear" not in modules


class _OddModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.custom_fc = nn.Linear(4, 4)


def test_default_target_modules_falls_back_to_suffixes_not_class_name():
    modules = _default_target_modules(_OddModel())
    assert modules == ["custom_fc"]
    assert "Linear" not in modules
