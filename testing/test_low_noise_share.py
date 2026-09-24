from types import SimpleNamespace

import pytest
import torch

from toolkit.config_modules import TrainConfig
from toolkit.low_noise_share import (
    bind_low_noise_share,
    low_noise_share_of_shift,
    shift_for_low_noise_share,
)
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler


def _h3_scheduler():
    # minimax_h3.scheduler_config
    return CustomFlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000, shift=12.0, use_dynamic_shifting=False
    )


def _share_below_half(scheduler) -> float:
    sigmas = scheduler.timesteps.float() / 1000.0
    return float((sigmas < 0.5).float().mean())


def test_config_default_off_and_validation():
    assert TrainConfig().low_noise_share is None
    assert TrainConfig(low_noise_share=0.6).low_noise_share == 0.6
    for bad in (0, 1, 1.5, -0.1, "x", True):
        with pytest.raises(ValueError):
            TrainConfig(low_noise_share=bad)


def test_shift_math_roundtrips():
    shift = shift_for_low_noise_share(0.6, 0.0, 1.0)
    assert shift == pytest.approx(2.0 / 3.0)
    assert low_noise_share_of_shift(shift, 0.0, 1.0) == pytest.approx(0.6)
    assert low_noise_share_of_shift(12.0, 0.0, 1.0) == pytest.approx(1.0 / 13.0)


def test_unset_override_is_the_model_default_grid():
    a, b = _h3_scheduler(), _h3_scheduler()
    a.set_train_timesteps(1000, device="cpu", timestep_type="shift")
    b.set_train_timesteps(1000, device="cpu", timestep_type="shift", shift_override=None)
    assert torch.equal(a.timesteps, b.timesteps)
    # H3's own draw: ~6.6% of the grid below sigma 0.5
    assert _share_below_half(a) == pytest.approx(0.066, abs=0.003)


def test_override_hits_the_requested_share_and_leaves_scheduler_shift():
    scheduler = _h3_scheduler()
    shift = shift_for_low_noise_share(0.6, float(scheduler.sigma_min), float(scheduler.sigma_max))
    scheduler.set_train_timesteps(1000, device="cpu", timestep_type="shift", shift_override=shift)
    assert _share_below_half(scheduler) == pytest.approx(0.6, abs=0.002)
    assert scheduler.shift == 12.0
    # the next un-overridden call is back to the model default
    scheduler.set_train_timesteps(1000, device="cpu", timestep_type="shift")
    assert _share_below_half(scheduler) == pytest.approx(0.066, abs=0.003)


def test_override_refused_on_dynamic_shifting_and_non_shift_types():
    dynamic = CustomFlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=True
    )
    latents = torch.zeros(1, 4, 16, 16)
    with pytest.raises(ValueError, match="dynamic"):
        dynamic.set_train_timesteps(
            1000, device="cpu", timestep_type="shift", latents=latents, shift_override=0.7
        )
    with pytest.raises(ValueError, match="shift timestep type"):
        _h3_scheduler().set_train_timesteps(
            1000, device="cpu", timestep_type="sigmoid", shift_override=0.7
        )


def _train_config(**overrides):
    base = dict(low_noise_share=0.6, timestep_type="shift", noise_scheduler="flowmatch")
    base.update(overrides)
    return TrainConfig(**base)


def test_bind_off_returns_none():
    assert bind_low_noise_share(TrainConfig(), SimpleNamespace()) is None


def test_bind_returns_shift_for_h3():
    sd = SimpleNamespace(noise_scheduler=_h3_scheduler(), is_multistage=False)
    shift = bind_low_noise_share(_train_config(), sd)
    assert 0.6 < shift < 0.7


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"timestep_type": "sigmoid"}, "timestep_type"),
        ({"content_or_style": "style"}, "content_or_style"),
        ({"first_timestep_chance": 0.1}, "first_timestep_chance"),
        ({"min_denoising_steps": 100}, "denoising"),
        ({"noise_scheduler": "ddpm"}, "noise_scheduler"),
    ],
)
def test_bind_fails_closed_on_draws_it_cannot_describe(overrides, match):
    sd = SimpleNamespace(noise_scheduler=_h3_scheduler(), is_multistage=False)
    with pytest.raises(ValueError, match=match):
        bind_low_noise_share(_train_config(**overrides), sd)


def test_h3_audio_share_is_reported_not_rescaled():
    from toolkit.low_noise_share import h3_audio_share

    scheduler = _h3_scheduler()
    u_min, u_max = float(scheduler.sigma_min), float(scheduler.sigma_max)
    shift = shift_for_low_noise_share(0.6, u_min, u_max)
    assert h3_audio_share(12.0, u_min, u_max) == pytest.approx(0.241, abs=0.003)
    assert h3_audio_share(shift, u_min, u_max) == pytest.approx(0.858, abs=0.003)


def test_training_pairs_audio_to_video_the_way_inference_does():
    # get_noise_prediction pairs sigma_a = remap_sigma(sigma_v) (default shifts
    # 12 -> 3). The pipeline pairs every step the same way, so a training draw
    # at any video sigma lands on a pair inference produces. Remapping from the
    # training shift would break exactly this.
    from extensions_built_in.diffusion_models.minimax_h3.src.packing import (
        AUDIO_SIGMA_SHIFT,
        VIDEO_SIGMA_SHIFT,
        build_sigma_schedule,
        remap_sigma,
    )

    sigmas_v = build_sigma_schedule(50).double()
    inference_a = remap_sigma(sigmas_v, VIDEO_SIGMA_SHIFT, AUDIO_SIGMA_SHIFT)
    assert torch.allclose(remap_sigma(sigmas_v), inference_a)
    sv = torch.tensor(0.2178, dtype=torch.float64)
    assert float(remap_sigma(sv)) == pytest.approx(0.065, abs=0.002)
    assert float(remap_sigma(sv, from_shift=0.6536)) > 0.5  # off the inference pairing


def test_bind_logs_h3_audio_share():
    sd = SimpleNamespace(noise_scheduler=_h3_scheduler(), is_multistage=False, arch="minimax_h3")
    assert bind_low_noise_share(_train_config(), sd) is not None


def test_bind_fails_on_dynamic_scheduler():
    sd = SimpleNamespace(
        noise_scheduler=CustomFlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=True
        ),
        is_multistage=False,
    )
    with pytest.raises(ValueError, match="dynamic"):
        bind_low_noise_share(_train_config(), sd)
