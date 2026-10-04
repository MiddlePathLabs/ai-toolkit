from types import SimpleNamespace

import pytest
import torch

from toolkit.category_stop import ANCHOR_SCALE, CategoryStop, bind_category_stop
from toolkit.config_modules import EMAConfig, TrainConfig
from toolkit.ema import ExponentialMovingAverage


def _batch(*kinds):
    items = []
    for kind in kinds:
        items.append(
            SimpleNamespace(is_video=kind == "clip", is_audio_only=kind == "voice")
        )
    return SimpleNamespace(file_items=items)


def _runtime(step_scale=True):
    return SimpleNamespace(
        supports_step_scale=step_scale,
        optimizer_label="Fake",
        unsupported_reason=None if step_scale else "no scale",
    )


def test_config_default_off_and_validation():
    assert TrainConfig().category_stop.enabled is False
    cfg = TrainConfig(category_stop={"voice_step": 1500})
    assert cfg.category_stop.steps == {"voice": 1500}
    assert cfg.category_stop.mode == "anchor"
    for bad in ({"voice_step": 0}, {"voice_step": 1.5}, {"voice_step": True}, {"mode": "pause"}):
        with pytest.raises(ValueError):
            TrainConfig(category_stop=bad)


def test_anchor_scales_only_retired_windows():
    stop = CategoryStop({"voice": 100}, "anchor")
    assert stop.step_scale({"voice"}, 99) == 1.0
    assert stop.step_scale({"voice"}, 100) == ANCHOR_SCALE
    assert stop.step_scale({"photo"}, 500) == 1.0
    assert stop.step_scale(set(), 500) == 1.0
    assert stop.should_skip(_batch("voice"), 500) is False  # anchor never skips


def test_stop_skips_only_retired_batches():
    stop = CategoryStop({"voice": 100}, "stop")
    assert stop.should_skip(_batch("voice"), 99) is False
    assert stop.should_skip(_batch("voice"), 100) is True
    assert stop.should_skip(_batch("photo"), 100) is False
    assert stop.should_skip(_batch("clip"), 100) is False
    assert stop.step_scale({"voice"}, 100) == 1.0  # stop mode never scales


def test_bind_requires_one_item_per_window():
    with pytest.raises(ValueError, match="batch_size"):
        bind_category_stop(TrainConfig(batch_size=2, category_stop={"voice_step": 10}), _runtime())
    with pytest.raises(ValueError, match="gradient_accumulation"):
        bind_category_stop(
            TrainConfig(gradient_accumulation=2, category_stop={"voice_step": 10}), _runtime()
        )


def test_bind_rejects_dataset_level_batch_size():
    cfg = TrainConfig(category_stop={"voice_step": 10})
    ok = [SimpleNamespace(batch_size=None, folder_path="a"), SimpleNamespace(batch_size=1, folder_path="b")]
    assert bind_category_stop(cfg, _runtime(), datasets=ok) is not None
    bad = ok + [SimpleNamespace(batch_size=4, folder_path="clips")]
    with pytest.raises(ValueError, match=r"datasets\[clips\]\.batch_size=4"):
        bind_category_stop(cfg, _runtime(), datasets=bad)


def test_mode_must_be_a_string():
    # unquoted YAML off / no / false load as bools; never fall back to anchor
    for bad in (False, True, 0):
        with pytest.raises(ValueError, match="as a string"):
            TrainConfig(category_stop={"voice_step": 10, "mode": bad})
    with pytest.raises(ValueError, match="one of"):
        TrainConfig(category_stop={"voice_step": 10, "mode": "off"})
    assert TrainConfig(category_stop={"voice_step": 10, "mode": None}).category_stop.mode == "anchor"
    assert TrainConfig(category_stop={"voice_step": 10, "mode": " Stop "}).category_stop.mode == "stop"


def test_ema_warmup_rejects_non_bool():
    for bad in ("false", "no", "0", 1, 0):
        with pytest.raises(ValueError, match="warmup"):
            EMAConfig(warmup=bad)
    assert EMAConfig(warmup=None).warmup is False
    assert EMAConfig(warmup=False).warmup is False


def test_bind_anchor_needs_step_scale_but_stop_does_not():
    with pytest.raises(ValueError, match="anchor"):
        bind_category_stop(TrainConfig(category_stop={"voice_step": 10}), _runtime(False))
    bound = bind_category_stop(
        TrainConfig(category_stop={"voice_step": 10, "mode": "stop"}), _runtime(False)
    )
    assert isinstance(bound, CategoryStop)
    assert bind_category_stop(TrainConfig(), _runtime()) is None


def test_reg_items_never_count_toward_a_category():
    from toolkit.category_stop import category_kinds, window_category_kinds

    reg_photo = SimpleNamespace(
        file_items=[SimpleNamespace(is_video=False, is_audio_only=False, is_reg=True)]
    )
    assert category_kinds(reg_photo) == set()
    assert window_category_kinds([reg_photo, _batch("voice"), None]) == {"voice"}
    stop = CategoryStop({"photo": 10}, "stop")
    assert stop.should_skip(reg_photo, 100) is False
    anchor = CategoryStop({"photo": 10}, "anchor")
    assert anchor.step_scale(category_kinds(reg_photo), 100) == 1.0


def test_bind_refuses_anchor_on_trainer_without_runtime_window():
    with pytest.raises(ValueError, match="runtime window"):
        bind_category_stop(
            TrainConfig(category_stop={"voice_step": 10}), _runtime(), supports_anchor=False
        )
    assert bind_category_stop(
        TrainConfig(category_stop={"voice_step": 10, "mode": "stop"}),
        _runtime(),
        supports_anchor=False,
    ) is not None




def _ema_host(step_num, warmup):
    from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess

    param = torch.nn.Parameter(torch.zeros(4))
    host = SimpleNamespace(
        train_config=TrainConfig(
            ema_config={"use_ema": True, "ema_decay": 0.98, "warmup": warmup}
        ),
        optimizer=SimpleNamespace(param_groups=[{"params": [param]}]),
        sd=SimpleNamespace(),
        step_num=step_num,
        completed_update_id=step_num,
        _resume_state=None,
    )
    BaseSDTrainProcess.setup_ema(host)
    return host


def test_ema_warmup_does_not_restart_on_resume():
    assert _ema_host(0, True).ema.num_updates == 0
    assert _ema_host(1200, True).ema.num_updates == 1200  # ramp saturated
    assert _ema_host(1200, False).ema.num_updates is None  # warmup off: unchanged


def test_ema_warmup_flag_ramps_decay():
    assert EMAConfig().warmup is False
    assert EMAConfig(warmup=True).warmup is True
    param = torch.nn.Parameter(torch.zeros(4))
    plain = ExponentialMovingAverage([param], decay=0.98)
    ramped = ExponentialMovingAverage([param], decay=0.98, use_num_updates=True)
    with torch.no_grad():
        param.fill_(1.0)
    plain.update()
    ramped.update()
    # constant decay moves the shadow 2%; the ramp's first decay is 2/11
    assert float(plain.shadow_params[0][0]) == pytest.approx(0.02)
    assert float(ramped.shadow_params[0][0]) == pytest.approx(1 - 2 / 11)
