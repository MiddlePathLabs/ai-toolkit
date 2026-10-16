"""resolve_h3_training_adapter + maybe_default_h3_training_adapter: named
H3 training-adapter shortcuts and the trainer-only Circlestone default."""

from types import SimpleNamespace

from extensions_built_in.diffusion_models.minimax_h3.minimax_h3 import (
    maybe_default_h3_training_adapter,
    resolve_h3_training_adapter,
)


# ------------------------------------------------- resolve_h3_training_adapter

def test_circlestone_named_choice():
    assert resolve_h3_training_adapter("circlestone") == (
        "circlestone-labs/MiniMax-H3-Image-Training-Adapter/"
        "minimax_h3_image_training_adapter.safetensors"
    )


def test_off_and_none_mean_no_adapter():
    for value in (None, "", "off", "OFF", "none", "disabled"):
        assert resolve_h3_training_adapter(value) is None


def test_ostris_maps_by_partition():
    assert "minimax_h3_training_adapter_v1" in resolve_h3_training_adapter("ostris")
    assert "minimax_h3_training_adapter_v1" in resolve_h3_training_adapter(
        "ostris", "fl2va_pruned"
    )
    assert "minimax_h3_ref2va_training_adapter_v1" in resolve_h3_training_adapter(
        "ostris", "ref2va_pruned"
    )


def test_explicit_variant_names():
    assert "minimax_h3_training_adapter_v1" in resolve_h3_training_adapter("ostris_fl2va")
    assert "ref2va_training_adapter" in resolve_h3_training_adapter("ostris_ref2va")


def test_other_values_pass_through_as_paths():
    assert resolve_h3_training_adapter("C:/models/adapter.safetensors") == (
        "C:/models/adapter.safetensors"
    )
    assert resolve_h3_training_adapter("user/repo/file.safetensors") == (
        "user/repo/file.safetensors"
    )
    assert resolve_h3_training_adapter("Circlestone ") == resolve_h3_training_adapter(
        "circlestone"
    )


# --------------------------------------- maybe_default_h3_training_adapter

def _mc(arch="minimax_h3", assistant=None, model_kwargs=None):
    return SimpleNamespace(
        arch=arch, assistant_lora_path=assistant, model_kwargs=model_kwargs or {}
    )


def test_default_injected_for_h3_training():
    mc = _mc()
    assert maybe_default_h3_training_adapter(mc) is True
    assert mc.model_kwargs["training_adapter"] == "circlestone"


def test_explicit_choice_not_overridden():
    mc = _mc(model_kwargs={"training_adapter": "ostris"})
    assert maybe_default_h3_training_adapter(mc) is False
    assert mc.model_kwargs["training_adapter"] == "ostris"


def test_off_opt_out_preserved():
    mc = _mc(model_kwargs={"training_adapter": "off"})
    assert maybe_default_h3_training_adapter(mc) is False
    assert mc.model_kwargs["training_adapter"] == "off"


def test_explicit_assistant_lora_path_wins():
    mc = _mc(assistant="user/repo/file.safetensors")
    assert maybe_default_h3_training_adapter(mc) is False
    assert "training_adapter" not in mc.model_kwargs


def test_non_h3_arch_untouched():
    for arch in ("krea2", "qwen_image_2", "flux3", None, ""):
        mc = _mc(arch=arch)
        assert maybe_default_h3_training_adapter(mc) is False
        assert mc.model_kwargs == {}


def test_none_config_is_safe():
    assert maybe_default_h3_training_adapter(None) is False
