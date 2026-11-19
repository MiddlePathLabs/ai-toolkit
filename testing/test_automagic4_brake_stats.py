"""Automagic v4 working-rate telemetry (Fizgig 5b0fbca4 / 882da7cc port):
the voted rate alone reads as its flat ceiling, so the reported lr is the
working rate — voted rate x mean brake — and the brake parts are exposed via
``brake_stats()`` for the progress bar. brake = min(1, ref / sensitivity):
elements whose sensitivity is above the group reference get clamped below 1."""

import pytest
import torch

from toolkit.optimizers.automagic3 import Automagic3
from toolkit.optimizers.automagic4 import Automagic4


def _step(opt, p, grad):
    p.grad = grad.clone()
    opt.step()
    opt.zero_grad(set_to_none=True)


def test_brake_stats_measurement_then_braked():
    p = torch.nn.Parameter(torch.zeros(8))
    # fused=False: manual grads (the fused variant rides backward hooks
    # and exercises the same _update_param / _accumulate_brake path)
    opt = Automagic4([p], lr=1e-4, fused=False)

    # bimodal constant grads: 4 elements insensitive (|g|=1), 4 sensitive
    # (|g|=100). Per-tensor geomean sensitivity = 10 = the group reference,
    # so exactly the sensitive half brakes, at balance = 10/100 = 0.1
    g = torch.tensor([1.0] * 4 + [100.0] * 4)

    # first update only measures (no reference yet): no stats, raw rate
    _step(opt, p, g)
    assert opt.brake_stats() == {"mean_brake": None, "braked_fraction": None}
    raw = Automagic3.get_avg_learning_rate(opt)
    assert opt.get_avg_learning_rate() == pytest.approx(raw)

    # second update brakes
    _step(opt, p, g)
    stats = opt.brake_stats()
    assert stats["braked_fraction"] == pytest.approx(0.5, rel=0.05)
    assert stats["mean_brake"] == pytest.approx((4 * 1.0 + 4 * 0.1) / 8, rel=0.05)

    # the reported rate is the working rate: raw average x mean brake
    raw = Automagic3.get_avg_learning_rate(opt)
    assert opt.get_avg_learning_rate() == pytest.approx(
        raw * stats["mean_brake"], rel=1e-6
    )
    assert opt.get_avg_learning_rate() < raw

    # the accumulator window rolls with the updates
    _step(opt, p, g)
    assert opt.brake_stats()["mean_brake"] is not None


def test_brake_stats_none_without_any_braked_step():
    p = torch.nn.Parameter(torch.zeros(4))
    opt = Automagic4([p], lr=1e-4, fused=False)
    assert opt.brake_stats() == {"mean_brake": None, "braked_fraction": None}
    assert opt.get_avg_learning_rate() == pytest.approx(
        Automagic3.get_avg_learning_rate(opt)
    )


def test_brake_stats_survive_param_that_misses_an_update():
    """A param with no grad this update (conditional module, truncated
    backward) lags one state step forever. The accumulator window is keyed by
    the update cycle, not per-param step, so the laggard must not split the
    window and shrink the reported stats to its own bucket."""
    hot = torch.nn.Parameter(torch.zeros(8))    # bimodal: half brakes
    cold = torch.nn.Parameter(torch.zeros(8))   # uniform: never brakes
    opt = Automagic4([hot, cold], lr=1e-4, fused=False)
    g_hot = torch.tensor([1.0] * 4 + [100.0] * 4)
    g_cold = torch.ones(8)

    # two full updates: measurement, then a braked one covering both params
    for _ in range(2):
        hot.grad, cold.grad = g_hot.clone(), g_cold.clone()
        opt.step()
        opt.zero_grad(set_to_none=True)
    both = opt.brake_stats()
    # group ref = 10 (numel-weighted 0.9 quantile of geomeans {1, 10}):
    # hot gives 4x1.0 + 4x0.1, cold gives 8x1.0 -> 12.4/16, 4/16 braked
    assert both["mean_brake"] == pytest.approx(12.4 / 16, rel=0.05)
    assert both["braked_fraction"] == pytest.approx(0.25, rel=0.05)

    # hot misses an update: the window covers only what was updated
    hot.grad = g_hot.clone()
    opt.step()
    opt.zero_grad(set_to_none=True)
    solo = opt.brake_stats()
    assert solo["mean_brake"] == pytest.approx(4.4 / 8, rel=0.05)

    # both resume: the laggard (one state step behind forever) must rejoin
    # the same window, not reset it midway through the update
    for _ in range(2):
        hot.grad, cold.grad = g_hot.clone(), g_cold.clone()
        opt.step()
        opt.zero_grad(set_to_none=True)
    rejoined = opt.brake_stats()
    assert rejoined["mean_brake"] == pytest.approx(both["mean_brake"], rel=0.05)
    assert rejoined["braked_fraction"] == pytest.approx(both["braked_fraction"], rel=0.05)

    # and the working rate is the all-param mean lr x that same brake
    raw = Automagic3.get_avg_learning_rate(opt)
    assert opt.get_avg_learning_rate() == pytest.approx(
        raw * rejoined["mean_brake"], rel=1e-6
    )
