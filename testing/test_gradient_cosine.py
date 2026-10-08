"""Gradient-cosine diagnostic tests (perceptual-fork port, 2026-10-08).

Covers the ported `SDTrainer._record_grad_cosine` / `_iter_trainable_params`
math against an independent autograd oracle, the `_grad_cosine_should_fire`
gate (incl. the max_loss-clamp exclusion), and the `_log_grad_cosine`
cross-microbatch accumulator. The dual-grad wiring in
`train_single_accumulation` mirrors the depth/* telemetry drain (canonical
keys `grad/norm/diffusion`, `grad/norm/depth`, `grad/cos/diff_depth`,
committed at the firing step); its live smoke belongs to a GPU run, not here.

Design pinned here (review fixes, 2026-10-08): BOTH gradient sides come from
pre-backward autograd.grad, never from p.grad — optimizer post-accumulate
hooks (stochastic _accum_grad accumulation, automagic2's fused step) move or
clear p.grad during backward, and offloaded params stage grads as async D2H
copies. The real backward + hooks below proves the diagnostic's values are
unaffected by grad clearing.
"""
import pytest
import torch

from toolkit.config_modules import TrainConfig
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer


class _FakeTrainer:
    """Minimal stand-in carrying the attrs the bound methods read/write."""


class _RecordingLogger:
    def __init__(self):
        self.calls = []

    def log(self, log_dict):
        self.calls.append(dict(log_dict))


class _FakeTrainConfig:
    def __init__(self, every):
        self.gradient_cosine_log_every = every


def _bind(tr):
    # Mirror SDTrainer.__init__ for the diagnostic attrs, then bind the
    # production methods exactly like a real instance would.
    tr._last_grad_norm_diffusion = None
    tr._last_grad_norm_depth = None
    tr._last_grad_cos_diff_depth = None
    tr._grad_cos_acc = None
    tr._dc_applied_for_grad = None
    tr._last_loss_clamped = False
    tr._iter_trainable_params = SDTrainer._iter_trainable_params.__get__(tr)
    tr._record_grad_cosine = SDTrainer._record_grad_cosine.__get__(tr)
    tr._log_grad_cosine = SDTrainer._log_grad_cosine.__get__(tr)
    tr._grad_cosine_should_fire = SDTrainer._grad_cosine_should_fire.__get__(tr)
    return tr


def _oracle(g_depth, g_diff):
    norm_dc_sq = norm_diff_sq = dot = 0.0
    for gd, gf in zip(g_depth, g_diff):
        if gd is None and gf is None:
            continue
        # a term one side never touched contributes nothing to that side
        gd = gd if gd is not None else torch.zeros_like(gf)
        gf = gf if gf is not None else torch.zeros_like(gd)
        norm_dc_sq += float((gd ** 2).sum())
        norm_diff_sq += float((gf ** 2).sum())
        dot += float((gd * gf).sum())
    norm_dc = norm_dc_sq ** 0.5
    norm_diff = norm_diff_sq ** 0.5
    cos = dot / (norm_dc * norm_diff) if norm_dc * norm_diff > 1e-12 else 0.0
    return norm_dc, norm_diff, cos


def _capture_and_record(tr, params, total, depth):
    """Production order: full-loss grad, depth-only grad, join, record."""
    full_grads = torch.autograd.grad(total, params, retain_graph=True,
                                     allow_unused=True)
    dc_grads = torch.autograd.grad(depth, params, retain_graph=True,
                                   allow_unused=True)
    tr._record_grad_cosine(full_grads, dc_grads)
    return full_grads, dc_grads


def test_record_grad_cosine_matches_independent_oracle():
    torch.manual_seed(0)
    w1 = torch.nn.Parameter(torch.randn(4))
    w2 = torch.nn.Parameter(torch.randn(4))
    params = [w1, w2]
    diff_loss = (3.0 * w1).pow(2).sum() + 0.5 * w2.pow(4).mean()
    depth_loss = 0.1 * (w1 * w2).sum()
    total = diff_loss + depth_loss

    tr = _bind(_FakeTrainer())
    g_dc = torch.autograd.grad(depth_loss, params, retain_graph=True)
    g_diff = torch.autograd.grad(diff_loss, params, retain_graph=True)
    norm_dc, norm_diff, cos = _oracle(g_dc, g_diff)
    _capture_and_record(tr, params, total, depth_loss)

    assert tr._last_grad_norm_depth == pytest.approx(norm_dc, rel=1e-5)
    assert tr._last_grad_norm_diffusion == pytest.approx(norm_diff, rel=1e-5)
    assert tr._last_grad_cos_diff_depth == pytest.approx(cos, rel=1e-5)
    assert -1.0 <= tr._last_grad_cos_diff_depth <= 1.0


def test_record_grad_cosine_immune_to_grad_clearing_hooks():
    # Regression (review 🔴): post-accumulate optimizer hooks move the grad to
    # _accum_grad and delete p.grad (stochastic accumulation), or null p.grad
    # outright (automagic2 fused step). The diagnostic must produce identical
    # values whether or not the real backward ran and cleared every p.grad.
    torch.manual_seed(1)
    w1 = torch.nn.Parameter(torch.randn(4))
    w2 = torch.nn.Parameter(torch.randn(4))
    params = [w1, w2]
    diff_loss = (2.0 * w1).pow(2).sum() + 0.5 * w2.pow(4).mean()
    depth_loss = 0.1 * (w1 * w2).sum()
    total = diff_loss + depth_loss

    # mimic toolkit stochastic_grad_accummulation exactly
    def _stash_and_delete(p):
        if hasattr(p, "_accum_grad"):
            p._accum_grad = p._accum_grad + p.grad
        else:
            p._accum_grad = p.grad.clone()
        del p.grad

    handles = [p.register_post_accumulate_grad_hook(_stash_and_delete) for p in params]

    tr = _bind(_FakeTrainer())
    full_grads, dc_grads = _capture_and_record(tr, params, total, depth_loss)
    expected = (tr._last_grad_norm_depth, tr._last_grad_norm_diffusion,
                tr._last_grad_cos_diff_depth)

    total.backward()  # hooks fire: every p.grad is now None
    for p in params:
        assert p.grad is None
        assert p._accum_grad is not None

    # re-record from the SAME pre-backward captures: identical — the reducer
    # is a pure function of the captured lists and never consults p.grad
    tr2 = _bind(_FakeTrainer())
    tr2._record_grad_cosine(full_grads, dc_grads)
    assert tr2._last_grad_norm_depth == pytest.approx(expected[0], rel=1e-6)
    assert tr2._last_grad_norm_diffusion == pytest.approx(expected[1], rel=1e-6)
    assert tr2._last_grad_cos_diff_depth == pytest.approx(expected[2], rel=1e-6)

    for h in handles:
        h.remove()


def test_record_grad_cosine_unused_param_counts_toward_diffusion_only():
    # allow_unused=True: a param the depth loss never touches gets g_dc=None;
    # its full gradient belongs to the diffusion side alone.
    torch.manual_seed(2)
    w1 = torch.nn.Parameter(torch.randn(4))
    w2 = torch.nn.Parameter(torch.randn(4))
    params = [w1, w2]
    diff_loss = w1.pow(2).sum() + w2.pow(2).sum()
    depth_loss = 0.5 * w1.sum()  # w2 unused by depth

    dc_grads = torch.autograd.grad(depth_loss, params, retain_graph=True,
                                   allow_unused=True)
    assert dc_grads[1] is None  # precondition for this branch

    tr = _bind(_FakeTrainer())
    _capture_and_record(tr, params, diff_loss + depth_loss, depth_loss)

    g_dc = torch.autograd.grad(depth_loss, params, retain_graph=True,
                               allow_unused=True)
    g_diff = torch.autograd.grad(diff_loss, params, retain_graph=True)
    norm_dc, norm_diff, cos = _oracle(g_dc, g_diff)

    assert tr._last_grad_norm_depth == pytest.approx(norm_dc, rel=1e-5)
    assert tr._last_grad_norm_diffusion == pytest.approx(norm_diff, rel=1e-5)
    assert tr._last_grad_cos_diff_depth == pytest.approx(cos, rel=1e-5)


def test_record_grad_cosine_zero_depth_grad_reports_zero_cos():
    # A zero depth gradient (denominator 0) must yield cos 0.0, never NaN.
    w1 = torch.nn.Parameter(torch.randn(4))
    params = [w1]
    diff_loss = w1.pow(2).sum()
    depth_loss = 0.0 * w1.sum()

    tr = _bind(_FakeTrainer())
    _capture_and_record(tr, params, diff_loss + depth_loss, depth_loss)
    assert tr._last_grad_norm_depth == 0.0
    assert tr._last_grad_norm_diffusion > 0.0
    assert tr._last_grad_cos_diff_depth == 0.0


def test_grad_cosine_should_fire_gate():
    # fires on matching parity with a stashed depth graph and no clamp
    tr = _bind(_FakeTrainer())
    tr.train_config = _FakeTrainConfig(50)
    tr._dc_applied_for_grad = torch.tensor(1.0)
    tr.step_num = 100
    assert tr._grad_cosine_should_fire() is True

    # off-parity step
    tr.step_num = 101
    assert tr._grad_cosine_should_fire() is False

    # depth did not contribute this microbatch
    tr.step_num = 100
    tr._dc_applied_for_grad = None
    assert tr._grad_cosine_should_fire() is False

    # max_loss-saturated microbatch: clamp zeroes the optimizer gradient,
    # so the diagnostic must not fire (review 🟡)
    tr._dc_applied_for_grad = torch.tensor(1.0)
    tr._last_loss_clamped = True
    assert tr._grad_cosine_should_fire() is False
    tr._last_loss_clamped = False
    assert tr._grad_cosine_should_fire() is True

    # disabled
    tr.train_config = _FakeTrainConfig(0)
    assert tr._grad_cosine_should_fire() is False


def test_log_grad_cosine_accumulates_across_microbatches():
    # Regression (review 🟡): logger.log is per-key last-write-wins and
    # step_num is constant across an accumulation window, so naive logging
    # keeps only the LAST microbatch. _log_grad_cosine logs the running
    # cross-microbatch MEAN; the step's final write is the full aggregate.
    tr = _bind(_FakeTrainer())
    tr.logger = _RecordingLogger()
    tr.step_num = 100

    tr._last_grad_norm_depth, tr._last_grad_norm_diffusion, tr._last_grad_cos_diff_depth = 1.0, 2.0, 0.5
    tr._log_grad_cosine()
    assert len(tr.logger.calls) == 1
    assert tr.logger.calls[0] == {
        'grad/norm/depth': 1.0, 'grad/norm/diffusion': 2.0, 'grad/cos/diff_depth': 0.5,
    }

    # second microbatch of the SAME optimizer step: running mean committed
    tr._last_grad_norm_depth, tr._last_grad_norm_diffusion, tr._last_grad_cos_diff_depth = 3.0, 4.0, -0.1
    tr._log_grad_cosine()
    assert tr.logger.calls[-1] == {
        'grad/norm/depth': pytest.approx(2.0),
        'grad/norm/diffusion': pytest.approx(3.0),
        'grad/cos/diff_depth': pytest.approx(0.2),
    }

    # a later firing step starts a fresh accumulator
    tr.step_num = 150
    tr._last_grad_norm_depth, tr._last_grad_norm_diffusion, tr._last_grad_cos_diff_depth = 5.0, 6.0, 0.9
    tr._log_grad_cosine()
    assert tr.logger.calls[-1] == {
        'grad/norm/depth': 5.0, 'grad/norm/diffusion': 6.0, 'grad/cos/diff_depth': 0.9,
    }


def test_iter_trainable_params_flattens_param_groups():
    p1 = torch.nn.Parameter(torch.zeros(1))
    p2 = torch.nn.Parameter(torch.zeros(1))
    p3 = torch.nn.Parameter(torch.zeros(1))
    tr = _bind(_FakeTrainer())
    tr.params = [{"params": [p1, p2]}, p3]
    assert list(tr._iter_trainable_params()) == [p1, p2, p3]

    tr.params = [p1, p2]
    assert list(tr._iter_trainable_params()) == [p1, p2]

    tr.params = []
    assert list(tr._iter_trainable_params()) == []


def test_train_config_gradient_cosine_log_every():
    assert TrainConfig().gradient_cosine_log_every == 0
    assert TrainConfig(gradient_cosine_log_every=50).gradient_cosine_log_every == 50
    assert TrainConfig(gradient_cosine_log_every="25").gradient_cosine_log_every == 25
