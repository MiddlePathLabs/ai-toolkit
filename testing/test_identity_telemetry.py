"""Canonical telemetry tests: payload map, t-bins, MetricBuffer.

Backed by the perceptual fork's metrics overhaul (SDTrainer._bin_update /
_canonical_telemetry_payload, extensions_built_in/sd_trainer/metric_buffer.py).
No GPU, model, or dataset needed -- exercises the static helpers only,
mirroring the fork's scripts/metrics_overhaul_smoke.py expectations.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from extensions_built_in.sd_trainer.metric_buffer import MetricBuffer, MetricValue


# ---------------------------------------------------------------------------
# t-bin accumulation
# ---------------------------------------------------------------------------

def test_bin_update_accumulates_and_finalizes():
    bins = {}
    SDTrainer._bin_update(bins, 'id_sim_t40', 0.10)
    SDTrainer._bin_update(bins, 'id_sim_t40', 0.20)
    SDTrainer._bin_update(bins, 'id_sim_t40', 0.30)
    SDTrainer._bin_update(bins, 'id_sim_t70', 0.50)
    SDTrainer._bin_update(bins, 'id_sim_t70', 0.70)
    out = SDTrainer._bin_finalize(bins)
    assert out['id_sim_t40'] == pytest.approx(0.20)
    assert out['id_sim_t70'] == pytest.approx(0.60)
    assert SDTrainer._bin_finalize({}) == {}


def test_bin_key_formatting_matches_fork():
    # fork: bin_start = int(t * 10) / 10.0 -> f'<prefix>_t{int(bin_start*100):02d}'
    for t_val, expected in [
        (0.0, 'id_sim_t00'),
        (0.04, 'id_sim_t00'),
        (0.47, 'id_sim_t40'),
        (0.5, 'id_sim_t50'),
        (0.999, 'id_sim_t90'),
        (1.0, 'id_sim_t100'),  # edge: fork pattern accepts 2-3 digits
    ]:
        bin_start = int(t_val * 10) / 10.0
        assert f'id_sim_t{int(bin_start * 100):02d}' == expected


# ---------------------------------------------------------------------------
# canonical payload
# ---------------------------------------------------------------------------

def test_canonical_payload_maps_and_skips():
    loss_dict = {'loss': 0.1, 'id_sim': 0.72, 'identity_loss': 0.28, 'id_clean_target': 0.70,
                 'body_shape_gated_pct': 0.9, 'weight_norm': 12.5}
    bins = {'id_sim': {}}
    SDTrainer._bin_update(bins['id_sim'], 'id_sim_t40', 0.5)
    SDTrainer._bin_update(bins['id_sim'], 'id_sim_t40', 0.7)
    payload = SDTrainer._canonical_telemetry_payload(loss_dict, {}, bins)
    assert payload['identity/sim'] == 0.72
    assert payload['identity/loss_raw'] == 0.28
    assert payload['identity/clean_target'] == 0.70
    assert payload['body_shape/gated_pct'] == 0.9
    assert payload['core/weight_norm'] == 12.5
    # absent legacy keys are skipped, not emitted as None
    assert 'identity/loss_applied' not in payload
    assert 'identity/clean_delta' not in payload
    assert payload['identity/sim/t40'] == pytest.approx(0.6)


def test_canonical_payload_prefers_buffer_mean():
    loss_dict = {'id_sim': 0.50}
    payload = SDTrainer._canonical_telemetry_payload(loss_dict, {'id_sim': 0.61}, {})
    assert payload['identity/sim'] == 0.61


def test_canonical_payload_bin_prefixes():
    bins = {
        'depth_loss': {'depth_loss_t80': {'sum': 0.4, 'count': 2}},
        'bsh_sim': {'bsh_sim_t10': {'sum': 0.9, 'count': 1}},
        'diffusion_loss': {'diffusion_loss_t00': {'sum': 0.2, 'count': 2}},
    }
    payload = SDTrainer._canonical_telemetry_payload({}, {}, bins)
    assert payload['depth/loss/t80'] == pytest.approx(0.2)
    assert payload['body_shape/sim/t10'] == pytest.approx(0.9)
    assert payload['diffusion/loss/t00'] == pytest.approx(0.1)


def test_canonical_payload_empty_is_empty():
    assert SDTrainer._canonical_telemetry_payload({'loss': 1.0}, {}, {}) == {}


# ---------------------------------------------------------------------------
# MetricBuffer (GA-safe means)
# ---------------------------------------------------------------------------

def test_metric_buffer_weighted_mean():
    buf = MetricBuffer()
    buf.add_scalar('id_sim', 0.6, weight=1)
    buf.add_scalar('id_sim', 0.8, weight=3)
    out = buf.flush_scalars()
    assert out['id_sim'] == pytest.approx((0.6 * 1 + 0.8 * 3) / 4)
    assert buf.flush_scalars() == {}  # flush clears state


def test_metric_buffer_drops_none_and_nonfinite():
    buf = MetricBuffer()
    buf.add_scalar('x', None)
    buf.add_scalar('x', float('nan'))
    buf.add_scalar('x', 1.0, weight=0.0)
    buf.add_scalar('x', 2.0)
    assert buf.flush_scalars() == {'x': 2.0}


def test_metric_value_is_float():
    mv = MetricValue(0.5, {"samples": [], "n": 0})
    assert mv == 0.5
    assert f"{mv:.2f}" == "0.50"
