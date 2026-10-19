"""'trunc_logit_normal' train timesteps: MOSS-Image's pretraining sigma draw.

The branch must reproduce the report's final pretraining stage (technical
report Appendix A.2, Table 6): sigma = sigmoid(-N(loc, scale^2)) truncated at
cutoff by inverse CDF, rescaled by / cutoff onto the full grid, snapped up to
the j/num_train_timesteps cell, with a uniform_frac share of uniform draws.
Each grid cell j (sigma = j/1000) therefore has analytic mass

    uniform_frac / T + (1 - uniform_frac) * (F(min(j*c/T, c)) - F((j-1)*c/T)) / F(c)

with F the untruncated sigmoid(-N(loc, scale^2)) CDF, and the tests compare a
large sampled grid against it cell by cell.
"""
import math

import pytest
import torch

from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)

N_SAMPLES = 200_000
T = 1000


def _scheduler():
    return CustomFlowMatchEulerDiscreteScheduler(
        shift=5.0, num_train_timesteps=T, use_dynamic_shifting=False
    )


def _cdf(x, loc, scale):
    # survival-function form: stable in the far tail (matches the sampler)
    return 0.5 * math.erfc(
        (math.log((1.0 - x) / x) - loc) / (scale * math.sqrt(2.0))
    )


def _cell_mass(j, loc, scale, cutoff, uniform_frac):
    upper = min(j * cutoff / T, cutoff)
    lower = (j - 1) * cutoff / T
    return uniform_frac / T + (1.0 - uniform_frac) * (
        _cdf(upper, loc, scale) - _cdf(lower, loc, scale)
    ) / _cdf(cutoff, loc, scale)


def _sample(loc=-1.09, scale=0.95, cutoff=0.95, uniform_frac=0.01, n=N_SAMPLES):
    torch.manual_seed(42)
    sched = _scheduler()
    ts = sched.set_train_timesteps(
        n,
        device="cpu",
        timestep_type="trunc_logit_normal",
        logit_loc=loc,
        logit_scale=scale,
        sigma_cutoff=cutoff,
        uniform_frac=uniform_frac,
    )
    return sched, ts


def test_report_defaults_match_analytic_cell_masses():
    sched, ts = _sample()
    assert ts is sched.timesteps
    assert len(ts) == N_SAMPLES
    for j in (50, 100, 300, 500, 800, 950, T):
        emp = (ts == j).float().mean().item()
        anal = _cell_mass(j, -1.09, 0.95, 0.95, 0.01)
        assert abs(emp - anal) < 0.004, f"cell {j}: empirical {emp:.5f} vs analytic {anal:.5f}"
    # the distribution is noise-tilted: median sigma ~0.78 under report defaults
    median = ts.float().median().item() / T
    assert 0.76 < median < 0.80


def test_grid_values_are_snapped_and_sorted():
    _, ts = _sample(n=5000)
    assert ts.dtype == torch.int
    assert int(ts.min()) >= 1
    assert int(ts.max()) <= T
    assert bool((ts[:-1] >= ts[1:]).all())  # sorted descending


def test_params_flow_through():
    # loc=0, no uniform: symmetric around sigma 0.5 (cutoff ~1 keeps rescale tiny)
    _, ts = _sample(loc=0.0, scale=1.0, cutoff=0.9999, uniform_frac=0.0)
    median = ts.float().median().item() / T
    assert 0.48 < median < 0.52
    # s = sigmoid(-Z): negative loc tilts toward noise, positive toward clean
    _, ts = _sample(loc=-2.0, scale=1.0, cutoff=0.95, uniform_frac=0.0)
    assert ts.float().median().item() / T > 0.9
    _, ts = _sample(loc=2.0, scale=1.0, cutoff=0.95, uniform_frac=0.0)
    assert ts.float().median().item() / T < 0.2


def test_validation_rejects_bad_params():
    sched = _scheduler()
    with pytest.raises(ValueError):
        sched.set_train_timesteps(10, device="cpu", timestep_type="trunc_logit_normal",
                                  sigma_cutoff=1.0)
    with pytest.raises(ValueError):
        sched.set_train_timesteps(10, device="cpu", timestep_type="trunc_logit_normal",
                                  logit_scale=0.0)
    with pytest.raises(ValueError):
        sched.set_train_timesteps(10, device="cpu", timestep_type="trunc_logit_normal",
                                  uniform_frac=1.0)
    with pytest.raises(ValueError):
        # shift_override is only valid with shift timestep types
        sched.set_train_timesteps(10, device="cpu", timestep_type="trunc_logit_normal",
                                  shift_override=3.0)


def test_extreme_tail_params_do_not_collapse():
    # F(cutoff) ~ 7.6e-24 here: 1 - Phi(...) rounds to 0 in float64, which
    # used to send every draw to inf and clamp the whole grid to timestep 1.
    # The conditional distribution concentrates just below the cutoff, i.e.
    # sigma = s / cutoff near 1 (noise).
    _, ts = _sample(loc=-10.0, scale=1.0, cutoff=0.5, uniform_frac=0.0)
    assert (ts == 1).float().mean().item() < 0.01
    assert ts.float().median().item() / T > 0.9


def test_scheduler_get_sigmas_resolves_duplicate_pool_values():
    torch.manual_seed(7)
    sched = _scheduler()
    pool = sched.set_train_timesteps(T, device="cpu", timestep_type="trunc_logit_normal")
    vals, counts = pool.unique(return_counts=True)
    duplicated = vals[counts > 1]
    assert len(duplicated) > 0  # a 1000-draw stochastic grid repeats values
    dup = int(duplicated[0])
    sigma = sched.get_sigmas(torch.tensor([dup, dup]), n_dim=4,
                             dtype=torch.float32, device="cpu")
    # flow-match sigma is the timestep on the training grid, by construction
    assert sigma.shape == (2, 1, 1, 1)
    assert torch.allclose(sigma.squeeze(), torch.full((2,), dup / T))


def test_process_get_sigmas_flow_match_direct():
    from types import SimpleNamespace

    from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess

    torch.manual_seed(7)
    sched = _scheduler()
    sched.set_train_timesteps(T, device="cpu", timestep_type="trunc_logit_normal")
    proc = BaseSDTrainProcess.__new__(BaseSDTrainProcess)
    proc.sd = SimpleNamespace(is_flow_matching=True, noise_scheduler=sched)
    proc.device = "cpu"
    timesteps = torch.tensor([1, 500, 1000, 500])
    sigma = proc.get_sigmas(timesteps, n_dim=4, dtype=torch.float32)
    assert sigma.shape == (4, 1, 1, 1)
    assert torch.allclose(sigma.squeeze(), timesteps.float() / T)


def test_ming_scheduler_forwards_new_kwargs():
    from extensions_built_in.diffusion_models.ming_image.ming_image import (
        MingImageFlowMatchScheduler,
        scheduler_config as ming_scheduler_config,
    )

    sched = MingImageFlowMatchScheduler(**ming_scheduler_config)
    # without latents (no dynamic-shift branch) and with them (register_to_config)
    for latents in (None, torch.zeros(1, 128, 8, 8)):
        ts = sched.set_train_timesteps(
            100,
            device="cpu",
            timestep_type="trunc_logit_normal",
            latents=latents,
            logit_loc=-1.09,
            logit_scale=0.95,
            sigma_cutoff=0.95,
            uniform_frac=0.01,
        )
        assert len(ts) == 100 and int(ts.min()) >= 1 and int(ts.max()) <= T
