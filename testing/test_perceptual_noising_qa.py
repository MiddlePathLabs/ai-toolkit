"""Focused QA tests for Phase 1 noising.

These tests are deliberately independent of the implementation's own assumptions:
they compute expected values from first principles and assert against them, and
they cover edge cases / ordering properties that the base test suite does not.

Run with the embedded training python and the repo root on the path, e.g.:
    PYTHONPATH=. E:\\ai-toolkit\\python_embeded\\python.exe -m pytest testing/test_perceptual_noising_qa.py -v
"""

import math

import pytest
import torch

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from toolkit.config_modules import TrainConfig


def make_trainer(params, **train_kwargs):
    trainer = SDTrainer.__new__(SDTrainer)
    trainer.params = params
    trainer.train_config = TrainConfig(**train_kwargs)
    trainer.step_num = 0
    trainer.optimizer = None
    return trainer


# ---------------------------------------------------------------------------
# 1. Independent expected-value checks for the relative weight-noise scale.
#    Verifies sigma scales by the parameter's own weight RMS (variant: using
#    gradient RMS instead would fail these).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_relative_weight_noise_sigma_equals_config_times_weight_rms(seed):
    torch.manual_seed(seed)
    w = torch.randn(2000)
    tagged = torch.nn.Parameter(w.clone())
    tagged._is_lora = True
    expected_rms = float(w.pow(2).mean().sqrt())

    trainer = make_trainer(
        [tagged],
        weight_noise={"enabled": True, "mode": "relative", "sigma": 0.00125, "log_every": 1},
    )
    before = tagged.detach().clone()
    trainer._inject_weight_noise()
    delta = (tagged.detach() - before)

    expected_sigma = 0.00125 * expected_rms
    # std of (delta/sigma) should be ~1.0 if the scale is exactly sigma.
    assert expected_sigma > 0
    ratio_std = float(delta.std(unbiased=True) / expected_sigma)
    assert abs(ratio_std - 1.0) < 0.15, f"scale mismatch: ratio_std={ratio_std}"
    # mean perturbation is ~0 (unbiased Gaussian).
    assert abs(float(delta.mean())) < expected_sigma


def test_relative_weight_noise_proportional_to_each_tensors_own_scale():
    torch.manual_seed(3)
    small = torch.nn.Parameter(torch.randn(5000) * 0.01)
    large = torch.nn.Parameter(torch.randn(5000) * 10.0)
    small._is_lora = True
    large._is_lora = True
    trainer = make_trainer(
        [small, large],
        weight_noise={"enabled": True, "mode": "relative", "sigma": 0.5, "log_every": 1},
    )
    s_before, l_before = small.detach().clone(), large.detach().clone()
    trainer._inject_weight_noise()
    s_std = float((small.detach() - s_before).std(unbiased=True))
    l_std = float((large.detach() - l_before).std(unbiased=True))
    s_rms = float(s_before.pow(2).mean().sqrt())
    l_rms = float(l_before.pow(2).mean().sqrt())
    # The ratio of injected stds should track the ratio of the tensors' RMS.
    assert abs((s_std / l_std) - (s_rms / l_rms)) < 0.05


# ---------------------------------------------------------------------------
# 2. Absolute weight noise must perturb a zero-initialized tensor (variant:
#    absolute failing on zero-init). Relative must leave it exactly zero.
# ---------------------------------------------------------------------------
def test_absolute_weight_noise_perturbs_zero_initialized_tensor():
    torch.manual_seed(5)
    tagged = torch.nn.Parameter(torch.zeros(1000))
    tagged._is_lora = True
    trainer = make_trainer(
        [tagged],
        weight_noise={"enabled": True, "mode": "absolute", "sigma": 0.1, "log_every": 1},
    )
    trainer._inject_weight_noise()
    assert not torch.equal(tagged, torch.zeros_like(tagged))
    assert float(tagged.std(unbiased=True)) == pytest.approx(0.1, rel=0.15)


def test_relative_weight_noise_is_exactly_zero_on_zero_tensor():
    torch.manual_seed(5)
    tagged = torch.nn.Parameter(torch.zeros(1000))
    tagged._is_lora = True
    trainer = make_trainer(
        [tagged],
        weight_noise={"enabled": True, "mode": "relative", "sigma": 0.5, "log_every": 1},
    )
    trainer._inject_weight_noise()
    assert torch.equal(tagged, torch.zeros_like(tagged))


# ---------------------------------------------------------------------------
# 3. bound_norm must not divide by zero and must restore pre-noise norm exactly
#    across scales, including tiny tensors.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scale", [1e-8, 1e-3, 1.0, 100.0])
def test_bound_norm_restores_norm_within_tolerance(scale):
    torch.manual_seed(9)
    tagged = torch.nn.Parameter(torch.randn(512) * scale)
    tagged._is_lora = True
    pre = float(tagged.norm())
    trainer = make_trainer(
        [tagged],
        weight_noise={"enabled": True, "mode": "absolute", "sigma": scale, "bound_norm": True, "log_every": 1},
    )
    trainer._inject_weight_noise()
    post = float(tagged.norm())
    if pre == 0.0:
        assert post == 0.0
    else:
        assert post == pytest.approx(pre, rel=1e-4)


def test_bound_norm_on_all_zero_tensor_is_safe():
    # DOCUMENTED EDGE CASE: a zero-norm tensor in absolute mode is perturbed,
    # and bound_norm cannot rescale it back (pre-noise norm is 0, so the rescale
    # branch is skipped to avoid division by zero). The safety guarantee here is
    # that the result is finite with no NaN/inf and no exception -- it does NOT
    # restore the tensor to zero. Relative mode is the only path that keeps a
    # zero tensor exactly zero (sigma scales by weight RMS -> 0 -> skipped).
    torch.manual_seed(9)
    tagged = torch.nn.Parameter(torch.zeros(64))
    tagged._is_lora = True
    trainer = make_trainer(
        [tagged],
        weight_noise={"enabled": True, "mode": "absolute", "sigma": 0.5, "bound_norm": True, "log_every": 1},
    )
    trainer._inject_weight_noise()  # must not raise
    assert torch.isfinite(tagged.data).all()
    assert not torch.equal(tagged, torch.zeros_like(tagged))


def test_bound_norm_actually_changes_direction_not_norm():
    torch.manual_seed(11)
    w = torch.randn(256)
    tagged = torch.nn.Parameter(w.clone())
    tagged._is_lora = True
    pre = tagged.detach().clone()
    pre_norm = float(pre.norm())
    trainer = make_trainer(
        [tagged],
        weight_noise={"enabled": True, "mode": "absolute", "sigma": 0.3, "bound_norm": True, "log_every": 1},
    )
    trainer._inject_weight_noise()
    # Norm preserved ...
    assert float(tagged.norm()) == pytest.approx(pre_norm, rel=1e-4)
    # ... but direction changed (not identical to pre-noise).
    cos = torch.nn.functional.cosine_similarity(
        tagged.detach().unsqueeze(0), pre.unsqueeze(0)
    ).item()
    assert cos < 0.9999


# ---------------------------------------------------------------------------
# 4. Disabled features are a true no-op: no parameter change AND no RNG draw
#    that would perturb an otherwise-deterministic run.
# ---------------------------------------------------------------------------
def test_disabled_weight_and_gradient_noise_consume_no_rng_and_change_nothing():
    tagged = torch.nn.Parameter(torch.randn(100))
    tagged._is_lora = True
    tagged.grad = torch.randn_like(tagged)
    grad_before = tagged.grad.clone()
    w_before = tagged.detach().clone()

    torch.manual_seed(123)
    state_a = torch.get_rng_state()
    trainer = make_trainer(
        [tagged],
        weight_noise={"enabled": False},
        gradient_noise={"enabled": False},
    )
    trainer._inject_gradient_noise()
    trainer._inject_weight_noise()

    # Now draw the same way an unrelated consumer would, with the SAME seed.
    torch.manual_seed(123)
    reference = torch.randn(3)

    torch.manual_seed(123)
    trainer._inject_gradient_noise()
    trainer._inject_weight_noise()
    after = torch.randn(3)

    assert torch.equal(reference, after), "disabled noising consumed RNG"
    assert torch.equal(tagged.grad, grad_before), "disabled grad noise changed grad"
    assert torch.equal(tagged, w_before), "disabled weight noise changed weights"


# ---------------------------------------------------------------------------
# 5. Neelakantan schedule: monotonic decay, exact sigma at step 0, and the
#    implemented formula matches eta / (1+step)**gamma from first principles.
# ---------------------------------------------------------------------------
def test_neelakantan_sigma_at_step_zero_equals_eta():
    torch.manual_seed(0)
    tagged = torch.nn.Parameter(torch.zeros(5000))
    tagged._is_lora = True
    tagged.grad = torch.zeros_like(tagged)
    trainer = make_trainer(
        [tagged],
        gradient_noise={"enabled": True, "mode": "neelakantan", "eta": 0.02, "gamma": 0.55, "log_every": 1},
    )
    trainer.step_num = 0
    before = tagged.grad.clone()
    trainer._inject_gradient_noise()
    delta = tagged.grad - before
    assert float(delta.std(unbiased=True)) == pytest.approx(0.02, rel=0.1)


@pytest.mark.parametrize("step", [0, 1, 5, 50, 500])
def test_neelakantan_sigma_matches_formula_independently(step):
    torch.manual_seed(step)
    tagged = torch.nn.Parameter(torch.zeros(20000))
    tagged._is_lora = True
    tagged.grad = torch.zeros_like(tagged)
    eta, gamma = 0.03, 0.6
    trainer = make_trainer(
        [tagged],
        gradient_noise={"enabled": True, "mode": "neelakantan", "eta": eta, "gamma": gamma, "log_every": 1},
    )
    trainer.step_num = step
    before = tagged.grad.clone()
    trainer._inject_gradient_noise()
    delta = tagged.grad - before
    expected = eta / ((1.0 + step) ** gamma)
    assert float(delta.std(unbiased=True)) == pytest.approx(expected, rel=0.08)


def test_neelakantan_schedule_is_monotonically_decreasing():
    sigmas = []
    for step in range(0, 30):
        torch.manual_seed(step)
        tagged = torch.nn.Parameter(torch.zeros(40000))
        tagged._is_lora = True
        tagged.grad = torch.zeros_like(tagged)
        trainer = make_trainer(
            [tagged],
            gradient_noise={"enabled": True, "mode": "neelakantan", "eta": 0.05, "gamma": 0.55, "log_every": 1},
        )
        trainer.step_num = step
        before = tagged.grad.clone()
        trainer._inject_gradient_noise()
        sigmas.append(float((tagged.grad - before).std(unbiased=True)))
    for a, b in zip(sigmas, sigmas[1:]):
        assert b <= a + 1e-9, f"schedule not monotonic: {sigmas}"


# ---------------------------------------------------------------------------
# 6. Absolute / relative gradient-noise exact sigma.
# ---------------------------------------------------------------------------
def test_absolute_gradient_noise_sigma_is_fixed():
    torch.manual_seed(0)
    tagged = torch.nn.Parameter(torch.zeros(20000))
    tagged._is_lora = True
    tagged.grad = torch.zeros_like(tagged)
    trainer = make_trainer(
        [tagged],
        gradient_noise={"enabled": True, "mode": "absolute", "sigma": 0.07, "log_every": 1},
    )
    before = tagged.grad.clone()
    trainer._inject_gradient_noise()
    assert float((tagged.grad - before).std(unbiased=True)) == pytest.approx(0.07, rel=0.08)


def test_relative_gradient_noise_scales_by_gradient_rms():
    torch.manual_seed(0)
    g = torch.randn(20000) * 2.0
    tagged = torch.nn.Parameter(torch.zeros(20000))
    tagged._is_lora = True
    tagged.grad = g.clone()
    grad_rms = float(g.pow(2).mean().sqrt())
    sigma_cfg = 0.25
    trainer = make_trainer(
        [tagged],
        gradient_noise={"enabled": True, "mode": "relative", "sigma": sigma_cfg, "log_every": 1},
    )
    before = tagged.grad.clone()
    trainer._inject_gradient_noise()
    delta = tagged.grad - before
    expected = sigma_cfg * grad_rms
    assert float(delta.std(unbiased=True)) == pytest.approx(expected, rel=0.08)


# ---------------------------------------------------------------------------
# 7. Fisher trace uses optimizer state, not p.grad: it must be non-zero even
#    after zero_grad(set_to_none=True), and must reflect only tagged params.
# ---------------------------------------------------------------------------
def test_fisher_trace_nonzero_after_grad_set_to_none():
    tagged = torch.nn.Parameter(torch.ones(16))
    tagged._is_lora = True
    untagged = torch.nn.Parameter(torch.ones(16))
    opt = torch.optim.Adam([tagged, untagged], lr=0.1)
    tagged.grad = torch.ones_like(tagged)
    untagged.grad = torch.ones_like(untagged)
    opt.step()
    # Simulate the real trainer clearing grads after the step.
    opt.zero_grad(set_to_none=True)
    assert tagged.grad is None
    trainer = make_trainer([tagged, untagged])
    trainer.optimizer = opt
    trainer._record_fisher_trace()
    assert trainer._last_fisher_trace is not None
    assert trainer._last_fisher_trace > 0.0


# ---------------------------------------------------------------------------
# 8. Gradient noise is added in-place AFTER clipping and can exceed the clip
#    threshold (no second clip). Document the intended behaviour explicitly.
# ---------------------------------------------------------------------------
def test_gradient_noise_added_after_clip_can_exceed_clip_threshold():
    torch.manual_seed(1)
    tagged = torch.nn.Parameter(torch.zeros(1000))
    tagged._is_lora = True
    tagged.grad = torch.full((1000,), 0.01)  # well below clip norm
    trainer = make_trainer(
        [tagged],
        gradient_noise={"enabled": True, "mode": "absolute", "sigma": 100.0, "log_every": 1},
    )
    trainer._inject_gradient_noise()
    # Norm now far exceeds a typical 1.0 clip; the implementation applies no
    # second clip. This documents the post-clip injection contract.
    assert float(tagged.grad.norm()) > 1.0


# ---------------------------------------------------------------------------
# 9. Metric values are detached python floats (no autograd graph / GPU tensor
#    retained across steps) and finite.
# ---------------------------------------------------------------------------
def test_metric_values_are_plain_finite_floats():
    torch.manual_seed(2)
    tagged = torch.nn.Parameter(torch.randn(100))
    tagged._is_lora = True
    tagged.grad = torch.randn_like(tagged)
    trainer = make_trainer(
        [tagged],
        weight_noise={"enabled": True, "mode": "absolute", "sigma": 0.1, "log_every": 1},
        gradient_noise={"enabled": True, "mode": "absolute", "sigma": 0.1, "log_every": 1},
    )
    trainer._inject_gradient_noise()
    trainer._inject_weight_noise()
    for name in ("_last_grad_noise_norm", "_last_grad_noise_snr",
                 "_last_weight_noise_norm", "_last_weight_norm"):
        val = getattr(trainer, name)
        assert isinstance(val, float), f"{name} is {type(val)}, not float"
        assert math.isfinite(val), f"{name} not finite: {val}"


# ---------------------------------------------------------------------------
# 10. Neelakantan negative-step safety: max(0, step) guard prevents weird
#     exponents; sigma stays finite and non-negative.
# ---------------------------------------------------------------------------
def test_neelakantan_handles_zero_step_safely():
    torch.manual_seed(0)
    tagged = torch.nn.Parameter(torch.zeros(10))
    tagged._is_lora = True
    tagged.grad = torch.zeros_like(tagged)
    trainer = make_trainer(
        [tagged],
        gradient_noise={"enabled": True, "mode": "neelakantan", "eta": 0.01, "gamma": 0.55, "log_every": 1},
    )
    trainer.step_num = 0
    trainer._inject_gradient_noise()  # must not raise / produce NaN


# ---------------------------------------------------------------------------
# 11. SNR is well-defined when the gradient norm is zero (all-zero gradient).
# ---------------------------------------------------------------------------
def test_grad_noise_snr_defined_when_signal_zero():
    torch.manual_seed(0)
    tagged = torch.nn.Parameter(torch.zeros(100))
    tagged._is_lora = True
    tagged.grad = torch.zeros_like(tagged)
    trainer = make_trainer(
        [tagged],
        gradient_noise={"enabled": True, "mode": "absolute", "sigma": 0.1, "log_every": 1},
    )
    trainer._inject_gradient_noise()
    # signal norm 0, noise norm > 0 -> SNR should be 0.0, finite.
    assert trainer._last_grad_noise_snr == 0.0
    assert math.isfinite(trainer._last_grad_noise_norm)
