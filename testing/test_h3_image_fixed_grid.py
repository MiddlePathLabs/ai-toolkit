"""MiniMax-H3-Image trains on a fixed 6-sigma grid; features written for the
1000-step grid must either describe it correctly or refuse it."""
import ast
import inspect
import textwrap
import types

import pytest
import torch

from extensions_built_in.diffusion_models.minimax_h3.minimax_h3 import MinimaxH3Model
from extensions_built_in.diffusion_models.minimax_h3.minimax_h3_image import (
    MinimaxH3ImageModel,
)
from extensions_built_in.diffusion_models.minimax_h3.src import pipeline as h3_pipeline
from toolkit.low_noise_share import bind_low_noise_share


def _train_config(**overrides):
    values = dict(
        low_noise_share=0.6,
        noise_scheduler="flowmatch",
        timestep_type="shift",
        linear_timesteps=False,
        linear_timesteps2=False,
        content_or_style="balanced",
        content_or_style_reg="balanced",
        num_train_timesteps=1000,
        min_denoising_steps=0,
        max_denoising_steps=999,
        first_timestep_chance=0.0,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


def test_low_noise_share_refuses_fixed_grid():
    scheduler = MinimaxH3ImageModel.get_train_scheduler()
    sd = types.SimpleNamespace(
        noise_scheduler=scheduler, arch="minimax_h3_image", is_multistage=False
    )
    with pytest.raises(ValueError, match="fixed sigma grid"):
        bind_low_noise_share(_train_config(), sd)


def test_low_noise_share_still_binds_on_h3_video():
    scheduler = MinimaxH3Model.get_train_scheduler()
    sd = types.SimpleNamespace(
        noise_scheduler=scheduler, arch="minimax_h3", is_multistage=False
    )
    assert bind_low_noise_share(_train_config(), sd) is not None


def test_shift_override_rejected_on_fixed_grid():
    scheduler = MinimaxH3ImageModel.get_train_scheduler()
    with pytest.raises(ValueError, match="fixed training sigma grid"):
        scheduler.set_train_timesteps(
            1000, "cpu", timestep_type="shift", shift_override=0.66
        )


@pytest.mark.parametrize("v2", [False, True])
def test_fixed_grid_weights_follow_sigma_not_grid_index(v2):
    fixed = MinimaxH3ImageModel.get_train_scheduler()
    timesteps = fixed.set_train_timesteps(1000, "cpu", timestep_type="shift")
    got = fixed.get_weights_for_timesteps(timesteps, v2=v2, timestep_type="shift")

    # the same timesteps on the 1000-step linear grid they were written for
    table = fixed.linear_timesteps_weights2 if v2 else fixed.linear_timesteps_weights
    expected = torch.stack(
        [table[(fixed.linear_timesteps - t).abs().argmin()] for t in timesteps]
    )
    assert torch.allclose(got, expected, atol=1e-2)
    # mid-schedule sigmas carry real weight (grid-index lookup gave < 0.03)
    assert float(got.max()) > 0.5


def test_weighted_scheme_reads_fixed_grid_by_timestep():
    from toolkit.timestep_weighing.default_weighing_scheme import (
        default_weighing_scheme,
    )

    fixed = MinimaxH3ImageModel.get_train_scheduler()
    timesteps = fixed.set_train_timesteps(1000, "cpu", timestep_type="shift")
    got = fixed.get_weights_for_timesteps(timesteps, timestep_type="weighted")
    expected = [
        default_weighing_scheme[min(max(int(round(1000 - float(t))), 0), 999)]
        for t in timesteps
    ]
    assert torch.allclose(got, torch.tensor(expected, dtype=got.dtype))


def test_sampler_reference_loop_keeps_target_audio_length():
    """The reference loop must not rebind the target's a_lat."""
    tree = ast.parse(
        textwrap.dedent(inspect.getsource(h3_pipeline.MiniMaxH3Pipeline.__call__))
    )
    assigned = [
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        for inner in ast.walk(node)
        if isinstance(inner, ast.Assign)
        for target in inner.targets
        if isinstance(target, ast.Name)
    ]
    assert "a_lat" not in assigned
