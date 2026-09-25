from types import SimpleNamespace

import pytest
import torch

from extensions_built_in.diffusion_models.minimax_h3.minimax_h3 import MinimaxH3Model
from extensions_built_in.diffusion_models.minimax_h3.src.packing import shift_sigma
from toolkit.config_modules import TrainConfig
from toolkit.timestep_focus import (
    TimestepFocus,
    apply_timestep_focus,
    band_base_weights,
    focus_band_indices,
    validate_timestep_focus,
)


class _H3:
    timestep_to_base_sigma = MinimaxH3Model.timestep_to_base_sigma


def _h3_grid(n=1000):
    # the 'shift' training grid: uniform base u from 1 down, shifted by 12
    base = torch.linspace(1.0, 1.0 / n, n)
    return shift_sigma(base, 12.0) * 1000.0, base


def test_config_defaults_off_and_validates():
    cfg = TrainConfig()
    assert cfg.timestep_focus_prob == 0.0
    assert (cfg.timestep_focus_min, cfg.timestep_focus_max) == (0.4, 0.8)
    with pytest.raises(ValueError):
        TrainConfig(timestep_focus_prob=1.5)
    with pytest.raises(ValueError):
        TrainConfig(timestep_focus_min=0.8, timestep_focus_max=0.4)


def test_band_is_selected_in_base_space_for_h3():
    grid, base = _h3_grid()
    focus = TimestepFocus(SimpleNamespace(timestep_focus_prob=1.0), _H3())
    out = focus(torch.zeros(4000, dtype=torch.long), grid, 0, 999)
    picked = base[out]
    assert picked.min() >= 0.4 - 1e-6 and picked.max() < 0.8
    # the video sigmas of that band are 0.89-0.98, not 0.4-0.8
    assert (grid[out] / 1000.0).min() > 0.88


def test_band_density_matches_musubi_formula():
    torch.manual_seed(0)
    _, base = _h3_grid()
    band = focus_band_indices(base, 0.4, 0.8, 0, 999)
    uniform = torch.randint(0, 1000, (200_000,))
    out = apply_timestep_focus(uniform, band, 0.5)
    in_band = ((base[out] >= 0.4) & (base[out] < 0.8)).float().mean().item()
    assert in_band == pytest.approx(0.5 + 0.5 * 0.4, abs=0.01)


def test_band_respects_denoising_range_and_fails_closed():
    _, base = _h3_grid()
    band = focus_band_indices(base, 0.4, 0.8, 300, 999)
    assert band.min() >= 300
    with pytest.raises(ValueError):
        focus_band_indices(base, 0.4, 0.8, 900, 999)


def test_off_is_identity_and_bias_types_fail_closed():
    idx = torch.arange(10)
    focus = TimestepFocus(SimpleNamespace(timestep_focus_prob=0.0), _H3())
    assert focus(idx, torch.ones(10), 0, 9) is idx
    with pytest.raises(ValueError):
        validate_timestep_focus(SimpleNamespace(
            timestep_focus_prob=0.5, timestep_type="shift", content_or_style="content"))
    validate_timestep_focus(SimpleNamespace(
        timestep_focus_prob=0.5, timestep_type="shift", content_or_style="balanced"))


def test_real_h3_training_grid():
    from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler

    scheduler = CustomFlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000, shift=12.0, use_dynamic_shifting=False
    )
    scheduler.set_train_timesteps(1000, device="cpu", timestep_type="shift")
    base = _H3().timestep_to_base_sigma(scheduler.timesteps.float() / 1000.0)
    band = focus_band_indices(base, 0.4, 0.8, 0, 999)
    # uniform base grid on [sigma_min, 1]: the band is 0.4 / (1 - sigma_min)
    # of the grid points, contiguous
    expected = 0.4 / (1.0 - float(scheduler.sigma_min)) * 1000
    assert band.numel() == pytest.approx(expected, abs=2)
    assert int(band.max() - band.min()) + 1 == band.numel()


def _real_grid(shift_override=None):
    from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler

    scheduler = CustomFlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000, shift=12.0, use_dynamic_shifting=False
    )
    kw = {} if shift_override is None else {"shift_override": shift_override}
    scheduler.set_train_timesteps(1000, device="cpu", timestep_type="shift", **kw)
    return scheduler


def test_default_grid_needs_no_weights():
    scheduler = _real_grid()
    base = _H3().timestep_to_base_sigma(scheduler.timesteps.float() / 1000.0)
    band = focus_band_indices(base, 0.4, 0.8, 0, 999)
    assert band_base_weights(base, band) is None


def test_low_noise_share_grid_keeps_the_band_and_stays_uniform_in_base():
    from toolkit.low_noise_share import shift_for_low_noise_share

    probe = _real_grid()
    shift = shift_for_low_noise_share(0.6, float(probe.sigma_min), float(probe.sigma_max))
    scheduler = _real_grid(shift)
    grid = scheduler.timesteps
    base = _H3().timestep_to_base_sigma(grid.float() / 1000.0)
    focus = TimestepFocus(SimpleNamespace(timestep_focus_prob=1.0), _H3())
    torch.manual_seed(0)
    out = focus(torch.zeros(200_000, dtype=torch.long), grid, 0, 999)
    picked = base[out]
    # same band of the model's schedule as without low_noise_share
    assert picked.min() >= 0.4 - 0.01 and picked.max() < 0.8
    assert focus._weights is not None
    # uniform in base sigma: each half of the band gets ~half the draws
    lower_half = ((picked >= 0.4) & (picked < 0.6)).float().mean().item()
    assert lower_half == pytest.approx(0.5, abs=0.03)


def test_generator_on_its_own_device():
    band = torch.arange(10, 20)
    g = torch.Generator(device="cpu").manual_seed(0)
    out = apply_timestep_focus(torch.zeros(8, dtype=torch.long), band, 1.0, generator=g)
    assert out.min() >= 10 and out.max() < 20
    if torch.cuda.is_available():
        g = torch.Generator(device="cuda").manual_seed(0)
        out = apply_timestep_focus(torch.zeros(8, dtype=torch.long), band, 1.0, generator=g,
                                   weights=torch.ones(10))
        assert out.device.type == "cpu" and out.min() >= 10
