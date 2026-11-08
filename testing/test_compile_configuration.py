import pytest

from jobs.process.BaseSDTrainProcess import _use_block_compile
from toolkit.config_modules import (
    ModelConfig,
    validate_block_compile_under_checkpointing,
)


def test_gradient_checkpointing_forces_whole_model_compile():
    assert not _use_block_compile(block_compile=True, gradient_checkpointing=True)


def test_block_compile_remains_available_without_checkpointing():
    assert _use_block_compile(block_compile=True, gradient_checkpointing=False)


def test_opt_in_key_lifts_the_checkpointing_guard():
    assert _use_block_compile(
        block_compile=True,
        gradient_checkpointing=True,
        allow_under_checkpointing=True,
    )
    # the opt-in flag alone does nothing when checkpointing is off
    assert _use_block_compile(
        block_compile=True, gradient_checkpointing=False, allow_under_checkpointing=False
    )


def test_invalid_compile_mode_is_rejected_before_training():
    with pytest.raises(ValueError, match="Invalid compile_mode 'fastest'"):
        ModelConfig(name_or_path="test", compile=True, compile_mode="fastest")


def test_stale_compile_mode_is_ignored_when_compile_is_disabled():
    config = ModelConfig(name_or_path="test", compile=False, compile_mode="fastest")

    assert config.compile_mode == "fastest"


def _native_krea2_opt_in(**overrides):
    # minimal valid Krea 2 ConvRot8 config with the opt-in key engaged
    kwargs = dict(
        name_or_path="test",
        arch="krea2",
        quantize=True,
        qtype="convrot8",
        compile=True,
        block_compile=True,
        block_compile_with_checkpointing=True,
    )
    kwargs.update(overrides)
    return ModelConfig(**kwargs)


def test_opt_in_key_parses():
    config = _native_krea2_opt_in()
    assert config.block_compile_with_checkpointing is True


def test_opt_in_key_requires_the_full_block_compile_setup():
    with pytest.raises(ValueError, match="requires compile: true and block_compile: true"):
        ModelConfig(name_or_path="test", block_compile_with_checkpointing=True)
    with pytest.raises(ValueError, match="requires compile: true and block_compile: true"):
        ModelConfig(
            name_or_path="test", compile=True, block_compile_with_checkpointing=True
        )


def test_opt_in_accepts_native_convrot8_under_checkpointing():
    validate_block_compile_under_checkpointing(_native_krea2_opt_in(), True)


def test_opt_in_rejects_comfy_kitchen_under_checkpointing():
    config = _native_krea2_opt_in(convrot8_backend="comfy_kitchen")
    with pytest.raises(ValueError, match="comfy_kitchen.*CheckpointError"):
        validate_block_compile_under_checkpointing(config, True)
    # without checkpointing the combo keeps its old behavior (block compile
    # runs normally; only the under-checkpointing path is guarded)
    validate_block_compile_under_checkpointing(config, False)


def test_opt_in_rejects_w8a8_backward_under_checkpointing():
    config = _native_krea2_opt_in(convrot8_backward="w8a8")
    with pytest.raises(ValueError, match="w8a8.*segfault"):
        validate_block_compile_under_checkpointing(config, True)


def test_broken_backends_pass_with_guard_explicitly_off():
    # the combos only matter when the guard lift would engage; since krea2
    # compile now defaults the lift on, opting out requires the explicit key
    kitchen = ModelConfig(
        name_or_path="test",
        arch="krea2",
        quantize=True,
        qtype="convrot8",
        compile=True,
        block_compile=True,
        block_compile_with_checkpointing=False,
        convrot8_backend="comfy_kitchen",
    )
    validate_block_compile_under_checkpointing(kitchen, True)


def test_krea2_compile_defaults_to_the_block_path():
    # compile: true alone on krea2 now selects per-block compile and lifts
    # the checkpointing guard (whole-model compile measured ~null here)
    config = ModelConfig(
        name_or_path="test", arch="krea2", quantize=True, qtype="convrot8",
        compile=True,
    )
    assert config.block_compile is True
    assert config.block_compile_with_checkpointing is True


def test_krea2_compile_defaults_respect_explicit_overrides():
    config = ModelConfig(
        name_or_path="test", arch="krea2", quantize=True, qtype="convrot8",
        compile=True, block_compile=False,
    )
    assert config.block_compile is False
    assert config.block_compile_with_checkpointing is False
    config = ModelConfig(
        name_or_path="test", arch="krea2", quantize=True, qtype="convrot8",
        compile=True, block_compile_with_checkpointing=False,
    )
    assert config.block_compile is True
    assert config.block_compile_with_checkpointing is False


def test_other_arches_keep_block_compile_opt_in():
    config = ModelConfig(name_or_path="test", arch="flux", compile=True)
    assert config.block_compile is False
    assert config.block_compile_with_checkpointing is False


def test_krea2_compile_w8a8_rejected_by_default():
    # with the guard lift now defaulting on, a krea2 compile user who also
    # sets w8a8 gets the fail-fast rejection instead of the segfault
    config = ModelConfig(
        name_or_path="test", arch="krea2", quantize=True, qtype="convrot8",
        compile=True, convrot8_backward="w8a8",
    )
    with pytest.raises(ValueError, match="w8a8"):
        validate_block_compile_under_checkpointing(config, True)
