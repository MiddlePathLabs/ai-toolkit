"""D-OPSD self-ref teacher recipe from musubi-tuner's `ref` teacher."""
from types import SimpleNamespace

import pytest

from extensions_built_in.diffusion_models.minimax_h3.minimax_h3 import MinimaxH3Ref2VAModel
from toolkit.config_modules import ModelConfig
from toolkit.h3_dopsd import (
    parse_dopsd_settings,
    teacher_step_conditioned,
    validate_dopsd,
    wrap_ref_teacher_caption,
)


def _cfg(**kw):
    return ModelConfig(name_or_path="x", arch="minimax_h3_ref2va", model_kwargs=kw)


def _model():
    return SimpleNamespace(get_base_model_version=lambda: "minimax_h3_ref2va")


def test_defaults_are_always_conditioned():
    s = parse_dopsd_settings(_cfg(dopsd=True))
    assert not s.teacher_gated
    assert s.preservation_weight == 1.0 and not s.copy_declaration
    assert teacher_step_conditioned(s, 1.0) and teacher_step_conditioned(s, 0.0)


def test_sigma_band_gates_the_teacher():
    s = parse_dopsd_settings(_cfg(
        dopsd=True, dopsd_teacher_sigma_max=0.75, dopsd_teacher_sigma_min=0.15,
        dopsd_preservation_weight=2.0,
    ))
    assert s.teacher_gated and s.preservation_weight == 2.0
    assert teacher_step_conditioned(s, 0.5)
    assert not teacher_step_conditioned(s, 0.9)   # high-sigma anchor
    assert not teacher_step_conditioned(s, 0.1)   # low-sigma anchor


def test_gating_validation():
    with pytest.raises(ValueError):
        parse_dopsd_settings(_cfg(dopsd=True, dopsd_teacher_sigma_max=1.5))
    bad = parse_dopsd_settings(_cfg(
        dopsd=True, dopsd_teacher_sigma_max=0.3, dopsd_teacher_sigma_min=0.5))
    with pytest.raises(ValueError, match="below"):
        validate_dopsd(bad, arch="minimax_h3_ref2va", model=_model())
    gated = parse_dopsd_settings(_cfg(dopsd=True, dopsd_teacher_sigma_max=0.75))
    with pytest.raises(ValueError, match="batch_size"):
        validate_dopsd(gated, arch="minimax_h3_ref2va", model=_model(),
                       train_config=SimpleNamespace(batch_size=2, loss_type="mse"))
    validate_dopsd(gated, arch="minimax_h3_ref2va", model=_model(),
                   train_config=SimpleNamespace(batch_size=1, loss_type="mse"))


def test_copy_declaration_wrap():
    with_audio = wrap_ref_teacher_caption("<Video 1> waves.", with_audio=True)
    assert with_audio.endswith("detailed_description:\n<Video 1> waves.")
    assert "<Video 1> (all shots): fully_preserved" in with_audio
    assert "<Audio 1>: fully_copy" in with_audio
    assert "[video editing + audio reuse]" in with_audio
    silent = wrap_ref_teacher_caption("<Video 1> waves.", with_audio=False)
    assert "<Audio 1>" not in silent
    assert "[video editing]" in silent


def _ref2va(**kw):
    model = object.__new__(MinimaxH3Ref2VAModel)
    model.model_config = SimpleNamespace(model_kwargs=kw)
    model.arch = "minimax_h3_ref2va"
    return model


def test_fl2va_base_is_allowed_only_for_dopsd():
    assert _ref2va(partition="ref2va")._dit_component() == "dit_ref2va"
    with pytest.raises(ValueError):
        _ref2va(partition="fl2va_pruned")._dit_component()
    m = _ref2va(partition="fl2va_pruned", dopsd=True)
    assert m._dit_component() == "dit_fl2va_pruned"
    assert m.get_base_model_version() == "minimax_h3_fl2va"
    # D-OPSD validation accepts the FL2VA base in the Ref2VA model class
    validate_dopsd(parse_dopsd_settings(_cfg(dopsd=True)), arch="minimax_h3_ref2va", model=m)


def test_teacher_audio_probe_runs_once_per_file(monkeypatch):
    import av
    from toolkit.dataloader_mixins import CaptionProcessingDTOMixin

    calls = []

    class _Container:
        streams = SimpleNamespace(audio=[object()])

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(av, "open", lambda path: calls.append(path) or _Container())
    item = SimpleNamespace(path="clip.mp4", dataset_config=SimpleNamespace(do_audio=True))
    probe = CaptionProcessingDTOMixin._dopsd_teacher_has_audio
    assert probe(item) is True
    assert probe(item) is True
    assert calls == ["clip.mp4"]
