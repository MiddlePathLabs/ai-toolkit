"""Guidance-loss sigma gate works in H3's pre-shift base space, not video sigma."""
from types import SimpleNamespace

import pytest
import torch

from extensions_built_in.diffusion_models.minimax_h3.minimax_h3 import MinimaxH3Model
from extensions_built_in.diffusion_models.minimax_h3.src.packing import shift_sigma
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer


class _H3Stub:
    timestep_to_base_sigma = MinimaxH3Model.timestep_to_base_sigma


def _trainer(sd, sigma_min):
    trainer = object.__new__(SDTrainer)
    trainer.sd = sd
    trainer.train_config = SimpleNamespace(guidance_loss_sigma_min=sigma_min)
    trainer.additional_logs = {}
    return trainer


def test_base_sigma_inverts_the_video_shift():
    base = torch.tensor([0.05, 0.15, 0.4, 0.8, 1.0])
    video = shift_sigma(base, 12.0)
    # musubi's table: base 0.15 -> video 0.68, base 0.8 -> video 0.98
    assert video[1].item() == pytest.approx(0.68, abs=0.005)
    assert video[3].item() == pytest.approx(0.98, abs=0.005)
    back = _trainer(_H3Stub(), 0.0)._base_sigma(video * 1000.0)
    assert torch.allclose(back, base, atol=1e-6)


def test_h3_gate_skips_the_cleanest_fifteen_percent_of_base_draws():
    base = torch.linspace(0.0005, 0.9995, 1000)
    timesteps = shift_sigma(base, 12.0) * 1000.0
    trainer = _trainer(_H3Stub(), 0.15)
    gate = trainer._log_guidance_sigma_gate(timesteps)
    assert gate.float().mean().item() == pytest.approx(0.85, abs=0.002)
    assert trainer.additional_logs['guidance/applied'] == pytest.approx(0.85, abs=0.002)


def test_models_without_the_hook_keep_timesteps_over_1000():
    timesteps = torch.tensor([100.0, 200.0, 900.0])
    trainer = _trainer(SimpleNamespace(), 0.15)
    assert trainer._log_guidance_sigma_gate(timesteps).tolist() == [False, True, True]
