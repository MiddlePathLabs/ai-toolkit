import os
from types import SimpleNamespace

import pytest
import torch

from toolkit.config_modules import TrainConfig
from toolkit.h3_dopsd import (
    IDENTITY_FIRST_AUTO_STEPS,
    IDENTITY_FIRST_LR_SCALE,
    assign_other_photo_pairs,
    dopsd_teacher_wanted,
    format_dopsd_error_log,
    identity_first_step_scale,
    identity_first_teacher_active,
    parse_dopsd_settings,
    pick_slot,
    resolve_identity_first_steps,
    rotation_partners,
    source_path,
    unweighted_errors,
    validate_dopsd,
)
from toolkit.optimizer import get_optimizer
from toolkit.optimizer_runtime import OptimizerRuntimeAdapter


def _model_config(**kwargs):
    return SimpleNamespace(model_kwargs=dict(kwargs), arch="minimax_h3")


def _photo(path, folder=None, *, video=False, audio=False, flip_x=False):
    if folder is None:
        folder = path.rsplit("/", 1)[0] if "/" in path or "\\" in path else "g"
    full = f"{folder}/{path}" if folder not in path else path
    return SimpleNamespace(
        path=full,
        is_video=video,
        is_audio_only=audio,
        flip_x=flip_x,
        flip_y=False,
    )


def _ref2va():
    return SimpleNamespace(
        arch="minimax_h3",
        get_base_model_version=lambda: "minimax_h3_ref2va",
    )


def test_default_off():
    settings = parse_dopsd_settings(_model_config())
    assert not settings.enabled
    assert settings.ref_mode == "self"
    assert not settings.identity_first


def test_self_ref_is_existing_dopsd_default():
    settings = parse_dopsd_settings(_model_config(dopsd=True))
    assert settings.enabled
    assert settings.self_ref
    assert not settings.other_ref
    assert settings.bleed_strength == 1.0


def test_other_mode_requires_dopsd():
    with pytest.raises(ValueError, match="dopsd_ref_mode='other' requires"):
        parse_dopsd_settings(_model_config(dopsd_ref_mode="other"))


def test_identity_first_requires_dopsd():
    with pytest.raises(ValueError, match="dopsd_identity_first requires"):
        parse_dopsd_settings(_model_config(dopsd_identity_first=True))


def test_invalid_ref_mode_fails_closed():
    with pytest.raises(ValueError, match="self' or 'other"):
        parse_dopsd_settings(_model_config(dopsd=True, dopsd_ref_mode="teacher"))


def test_rotation_never_includes_self():
    keys = ["a.jpg", "b.jpg", "c.jpg"]
    partners = rotation_partners(keys, 2)
    for key, refs in partners.items():
        assert key not in refs
        assert len(refs) == 2
        assert len(set(refs)) == 2


def test_rotation_singleton_fails():
    with pytest.raises(ValueError, match="at least 2 photos"):
        rotation_partners(["only.jpg"], 1)


def test_folders_never_cross_subjects():
    settings = parse_dopsd_settings(
        _model_config(dopsd=True, dopsd_ref_mode="other", dopsd_ref_count=1)
    )
    items = [
        _photo("a.jpg", r"E:\subjA"),
        _photo("b.jpg", r"E:\subjA"),
        _photo("c.jpg", r"E:\subjB"),
        _photo("d.jpg", r"E:\subjB"),
    ]
    assign_other_photo_pairs(items, settings)
    a_partners = {slot["path"] for slot in items[0].dopsd_ref_slots}
    assert any(p.endswith("b.jpg") for p in a_partners)
    assert not any(p.endswith("c.jpg") or p.endswith("d.jpg") for p in a_partners)


def test_singleton_folder_fails_closed():
    settings = parse_dopsd_settings(
        _model_config(dopsd=True, dopsd_ref_mode="other")
    )
    items = [_photo("only.jpg", r"E:\alone"), _photo("x.jpg", r"E:\pair"), _photo("y.jpg", r"E:\pair")]
    with pytest.raises(ValueError, match="never pairs an item with itself"):
        assign_other_photo_pairs(items, settings)


def test_own_flip_is_not_a_partner():
    settings = parse_dopsd_settings(
        _model_config(dopsd=True, dopsd_ref_mode="other", dopsd_ref_count=2)
    )
    orig = _photo("a.jpg", r"E:\s")
    flip = _photo("a.jpg", r"E:\s", flip_x=True)
    other = _photo("b.jpg", r"E:\s")
    assign_other_photo_pairs([orig, flip, other], settings)
    for item in (orig, flip):
        assert item.dopsd_ref_slots
        for slot in item.dopsd_ref_slots:
            assert os.path.normcase(os.path.abspath(slot["path"])) != source_path(item)
            assert slot["path"].endswith("b.jpg")


def test_one_still_and_its_flip_fails_closed():
    settings = parse_dopsd_settings(
        _model_config(dopsd=True, dopsd_ref_mode="other")
    )
    orig = _photo("only.jpg", r"E:\alone")
    flip = _photo("only.jpg", r"E:\alone", flip_x=True)
    with pytest.raises(ValueError, match="own flip"):
        assign_other_photo_pairs([orig, flip], settings)



def test_clips_and_voice_sit_out_of_pairing():
    settings = parse_dopsd_settings(
        _model_config(dopsd=True, dopsd_ref_mode="other")
    )
    photos = [_photo("a.jpg", r"E:\s"), _photo("b.jpg", r"E:\s")]
    clip = _photo("c.mp4", r"E:\s", video=True)
    voice = _photo("d.wav", r"E:\s", audio=True)
    assign_other_photo_pairs(photos + [clip, voice], settings)
    assert photos[0].dopsd_ref_slots
    assert clip.dopsd_ref_slots == []
    assert voice.dopsd_ref_slots == []


def test_pick_slot_is_deterministic_and_changes_with_epoch():
    k = 4
    a = pick_slot("E:/s/a.jpg|0|0", epoch=0, seed=7, k=k)
    b = pick_slot("E:/s/a.jpg|0|0", epoch=0, seed=7, k=k)
    assert a == b
    assert 0 <= a < k
    other = {pick_slot("E:/s/a.jpg|0|0", epoch=e, seed=7, k=k) for e in range(40)}
    assert len(other) > 1


def test_identity_first_auto_is_650_steps_not_epochs():
    assert resolve_identity_first_steps(-1, 5000) == IDENTITY_FIRST_AUTO_STEPS
    assert resolve_identity_first_steps(-1, 100) == 100
    assert resolve_identity_first_steps(12, 5000) == 12


def test_identity_first_phase_switch():
    assert identity_first_teacher_active(
        0, 10, identity_first=True, dopsd_enabled=True
    )
    assert not identity_first_teacher_active(
        10, 10, identity_first=True, dopsd_enabled=True
    )
    assert identity_first_teacher_active(
        999, 10, identity_first=False, dopsd_enabled=True
    )
    assert identity_first_step_scale(True, identity_first=True) == IDENTITY_FIRST_LR_SCALE
    assert identity_first_step_scale(False, identity_first=True) == 1.0


def test_phase2_skips_teacher_and_other_photo_batch():
    settings = parse_dopsd_settings(
        _model_config(
            dopsd=True,
            dopsd_ref_mode="other",
            dopsd_identity_first=True,
            dopsd_identity_first_steps=5,
        )
    )
    settings = settings.__class__(
        **{**settings.__dict__, "identity_first_steps": 5}
    )
    photo_batch = SimpleNamespace(file_items=[_photo("a.jpg", r"E:\s")])
    clip_batch = SimpleNamespace(file_items=[_photo("c.mp4", r"E:\s", video=True)])
    assert dopsd_teacher_wanted(settings, step_num=0, batch=photo_batch)
    assert not dopsd_teacher_wanted(settings, step_num=5, batch=photo_batch)
    assert not dopsd_teacher_wanted(settings, step_num=0, batch=clip_batch)


def test_unweighted_errors_are_raw():
    parts = unweighted_errors(0.08, 0.20)
    assert parts["teacher"] == pytest.approx(0.08)
    assert parts["photo"] == pytest.approx(0.20)
    text = format_dopsd_error_log(0.08, 0.20, teacher_weight=0.8)
    assert "teacher err 0.0800" in text
    assert "photo err 0.2000" in text


def test_other_photo_rejects_batch_size_over_one():
    settings = parse_dopsd_settings(
        _model_config(dopsd=True, dopsd_ref_mode="other")
    )
    with pytest.raises(ValueError, match="batch_size: 1"):
        validate_dopsd(
            settings,
            arch="minimax_h3",
            train_config=TrainConfig(batch_size=2),
            model=_ref2va(),
        )


def test_dopsd_rejects_non_h3():
    settings = parse_dopsd_settings(_model_config(dopsd=True))
    with pytest.raises(ValueError, match="MiniMax-H3 only"):
        validate_dopsd(settings, arch="flux", model=_ref2va())


def test_dopsd_rejects_non_ref2va_backbone():
    settings = parse_dopsd_settings(_model_config(dopsd=True))
    fl2va = SimpleNamespace(
        arch="minimax_h3",
        get_base_model_version=lambda: "minimax_h3_fl2va",
    )
    with pytest.raises(ValueError, match="ref2va"):
        validate_dopsd(settings, arch="minimax_h3", model=fl2va)


def test_identity_first_without_step_scale_fails_closed():
    settings = parse_dopsd_settings(
        _model_config(dopsd=True, dopsd_identity_first=True, dopsd_identity_first_steps=10)
    )
    runtime = SimpleNamespace(
        supports_step_scale=False,
        optimizer_label="prodigy",
        unsupported_reason="controller uses group lr",
    )
    with pytest.raises(ValueError, match="supports_step_scale"):
        validate_dopsd(
            settings,
            arch="minimax_h3",
            optimizer_runtime=runtime,
            train_config=TrainConfig(steps=1000),
            model=_ref2va(),
        )


def _smoke_identity_first_scale(optimizer_type: str):
    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.ones(4, 4))
    opt = get_optimizer([param], optimizer_type, learning_rate=1e-3)
    adapter = OptimizerRuntimeAdapter.inspect(opt, optimizer_type=optimizer_type)
    assert adapter.supports_step_scale
    param.grad = torch.ones_like(param)
    before = param.detach().clone()
    adapter.begin_window(opt, IDENTITY_FIRST_LR_SCALE)
    if optimizer_type != "automagic2":
        opt.step()
        assert not torch.equal(param.detach(), before)
    adapter.end_window(opt)
    if adapter.step_scale_strategy == "group_lr":
        assert opt.param_groups[0]["lr"] == pytest.approx(1e-3)



def test_step_time_optimizer_survives_identity_first_scale():
    _smoke_identity_first_scale("adam")


def test_fused_backward_optimizer_survives_identity_first_scale():
    _smoke_identity_first_scale("automagic2")


def test_other_photo_cache_rebuilds_embeds_from_one_read(tmp_path):
    from safetensors.torch import save_file
    from toolkit.h3_dopsd import load_other_photo_teacher_cache

    path = tmp_path / "slot.safetensors"
    save_file(
        {
            "text_embed": torch.ones(2, 4),
            "dopsd_ref_tensor": torch.zeros(1, 3, 8, 8),
        },
        str(path),
    )
    embeds, ref = load_other_photo_teacher_cache(str(path))
    assert tuple(embeds.text_embeds.shape) == (2, 4)
    assert tuple(ref.shape) == (1, 3, 8, 8)



def _prior_trainer(*, unload_text_encoder=False, predict_error=None):
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer

    class _UNet:
        def __init__(self):
            self.training = True

        def eval(self):
            self.training = False

        def train(self):
            self.training = True

    def _predict_noise(**kwargs):
        if predict_error is not None:
            raise predict_error
        return torch.zeros(1, 4, 8, 8)

    trainer = SDTrainer.__new__(SDTrainer)
    trainer.network = SimpleNamespace(is_active=True)
    trainer.adapter = None
    trainer.embedding = None
    trainer.train_config = SimpleNamespace(
        unload_text_encoder=unload_text_encoder,
        dtype="fp32",
        cfg_scale=1.0,
        do_guidance_loss=False,
        cfg_rescale=1.0,
    )
    trainer.sd = SimpleNamespace(
        unet=_UNet(),
        device_torch=torch.device("cpu"),
        torch_dtype=torch.float32,
        encode_control_in_text_embeddings=False,
        is_flux=False,
        is_lumina2=False,
        predict_noise=_predict_noise,
    )
    trainer.device_torch = torch.device("cpu")
    return trainer


def _call_prior(trainer):
    from toolkit.prompt_utils import PromptEmbeds

    noisy = torch.zeros(1, 4, 8, 8)
    return trainer.get_prior_prediction(
        noisy,
        PromptEmbeds(torch.zeros(1, 4, 8)),
        False,
        [1.0],
        torch.zeros(1),
        {},
        SimpleNamespace(),
        noisy,
    )


def test_prior_prediction_restores_network_after_oom():
    trainer = _prior_trainer(predict_error=torch.cuda.OutOfMemoryError("cuda OOM"))
    with pytest.raises(torch.cuda.OutOfMemoryError):
        _call_prior(trainer)
    assert trainer.network.is_active is True
    assert trainer.sd.unet.training is True


def test_prior_prediction_restores_network_after_unload_text_encoder():
    trainer = _prior_trainer(unload_text_encoder=True)
    trainer.adapter = SimpleNamespace(is_active=True)
    with pytest.raises(ValueError, match="unloading text encoder"):
        _call_prior(trainer)
    assert trainer.network.is_active is True
    assert trainer.sd.unet.training is True


def test_prior_prediction_restores_network_on_success():
    trainer = _prior_trainer()
    out = _call_prior(trainer)
    assert tuple(out.shape) == (1, 4, 8, 8)
    assert trainer.network.is_active is True
    assert trainer.sd.unet.training is True
