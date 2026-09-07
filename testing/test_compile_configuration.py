import pytest

from jobs.process.BaseSDTrainProcess import _use_block_compile
from toolkit.config_modules import ModelConfig


def test_gradient_checkpointing_forces_whole_model_compile():
    assert not _use_block_compile(block_compile=True, gradient_checkpointing=True)


def test_block_compile_remains_available_without_checkpointing():
    assert _use_block_compile(block_compile=True, gradient_checkpointing=False)


def test_invalid_compile_mode_is_rejected_before_training():
    with pytest.raises(ValueError, match="Invalid compile_mode 'fastest'"):
        ModelConfig(name_or_path="test", compile=True, compile_mode="fastest")


def test_stale_compile_mode_is_ignored_when_compile_is_disabled():
    config = ModelConfig(name_or_path="test", compile=False, compile_mode="fastest")

    assert config.compile_mode == "fastest"
