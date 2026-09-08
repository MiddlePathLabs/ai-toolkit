from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from toolkit.config_modules import TrainConfig
from toolkit.optimizer import get_optimizer
from toolkit.optimizer_runtime import (
    OptimizerRuntimeAdapter,
    classify_optimizer,
    mean_window_scale,
    uses_adaptive_lr_step_scale,
)


def _train_cfg(**kwargs):
    defaults = dict(
        per_image_adaptive_lr=True,
        per_image_adaptive_lr_mode="lr",
        per_image_adaptive_lr_stats_only=False,
        gradient_accumulation=1,
        gradient_accumulation_steps=1,
        single_item_batching=False,
        loss_type="mse",
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _params():
    return torch.nn.Linear(4, 4, bias=False).parameters()


def _one_step(make_opt, scale, seed=0, phase="step"):
    torch.manual_seed(seed)
    model = torch.nn.Linear(8, 8, bias=False)
    opt = make_opt(model.parameters())
    adapter = OptimizerRuntimeAdapter.inspect(opt)
    x = torch.randn(4, 8)
    y = torch.randn(4, 8)
    if phase == "backward":
        adapter.begin_window(opt, scale)
        opt.zero_grad(set_to_none=True)
        F.mse_loss(model(x), y).backward()
        before = model.weight.detach().clone()
        opt.step()
        adapter.end_window(opt)
    else:
        opt.zero_grad(set_to_none=True)
        F.mse_loss(model(x), y).backward()
        before = model.weight.detach().clone()
        adapter.begin_window(opt, scale)
        opt.step()
        adapter.end_window(opt)
    return model.weight.detach() - before, opt, adapter


def test_mean_window_scale_counts_each_item_once():
    assert mean_window_scale({}) == 1.0
    assert mean_window_scale({"a": 0.5, "b": 1.5}) == pytest.approx(1.0)
    assert mean_window_scale({"a": 0.4}) == pytest.approx(0.4)
    # last occurrence wins — a dict cannot double-count a path
    members = {"a": 0.5}
    members["a"] = 1.0
    members["reg"] = 1.0
    assert mean_window_scale(members) == pytest.approx(1.0)


def test_begin_window_allows_zero_rejects_nonfinite_and_negative():
    model = torch.nn.Linear(2, 2, bias=False)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    adapter = OptimizerRuntimeAdapter.inspect(opt)
    adapter.begin_window(opt, 0.0)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.0)
    adapter.end_window(opt)
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3)
    with pytest.raises(ValueError, match="finite number"):
        adapter.begin_window(opt, float("nan"))
    with pytest.raises(ValueError, match="finite number"):
        adapter.begin_window(opt, float("inf"))
    with pytest.raises(ValueError, match="finite number"):
        adapter.begin_window(opt, -0.5)



def test_train_config_mode_defaults_to_loss_and_rejects_unknown():
    cfg = TrainConfig()
    assert cfg.per_image_adaptive_lr_mode == "loss"
    assert uses_adaptive_lr_step_scale(cfg) is False
    cfg = TrainConfig(per_image_adaptive_lr=True, per_image_adaptive_lr_mode="lr")
    assert uses_adaptive_lr_step_scale(cfg) is True
    cfg = TrainConfig(
        per_image_adaptive_lr=True,
        per_image_adaptive_lr_mode="lr",
        per_image_adaptive_lr_stats_only=True,
    )
    assert uses_adaptive_lr_step_scale(cfg) is False
    with pytest.raises(ValueError, match="per_image_adaptive_lr_mode"):
        TrainConfig(per_image_adaptive_lr_mode="nope")


def test_classify_factory_branches_cpu():
    cases = [
        ("rose", {}, "group_lr", "step", True),
        ("adam", {}, "group_lr", "step", True),
        ("adamw", {}, "group_lr", "step", True),
        ("adagrad", {}, "group_lr", "step", True),
        ("adafactor", {}, "group_lr", "step", True),
        ("adafactor", {"beta1": 0.9}, "native", "step", True),
        ("adafactor", {"scale_parameter": True}, "unsupported", "step", False),
        ("automagic", {}, "native", "step", True),
        ("automagic3", {"fused": False}, "native", "step", True),
        ("automagicexperiment", {"fused": False}, "native", "step", True),
        ("adamconvrot", {}, "native", "step", True),
        ("adam8", {}, "group_lr", "step", True),
        ("adamw8", {}, "group_lr", "step", True),
        ("prodigy8bit", {}, "native", "step", True),
        ("prodigy", {}, "unsupported", "step", False),
    ]
    for name, params, strategy, phase, supported in cases:
        opt = get_optimizer(_params(), name, 1.0 if name.startswith("prodigy") else 1e-3, params)
        caps = classify_optimizer(opt)
        assert caps.step_scale_strategy == strategy, name
        assert caps.update_phase == phase, name
        assert caps.supports_step_scale is supported, name
        assert caps.supports_active_param_mask is False, name


def test_classify_fused_backward_families():
    opt = get_optimizer(_params(), "automagic2", 1e-6, {})
    caps = classify_optimizer(opt)
    assert caps.step_scale_strategy == "native"
    assert caps.update_phase == "backward"
    opt = get_optimizer(_params(), "automagic3", 1e-6, {})
    assert classify_optimizer(opt).update_phase == "backward"
    opt = get_optimizer(_params(), "adamconvrot", 1e-3, {"fused": True})
    assert classify_optimizer(opt).update_phase == "backward"


def test_unknown_optimizer_is_unsupported():
    class External(torch.optim.SGD):
        pass

    opt = External(torch.nn.Linear(2, 2).parameters(), lr=1e-3)
    adapter = OptimizerRuntimeAdapter.inspect(opt)
    assert adapter.supports_step_scale is False
    with pytest.raises(ValueError, match="has not declared supports_step_scale"):
        adapter.validate_for_train_config(_train_cfg())


def test_prodigy_lr_mode_fails_closed():
    opt = get_optimizer(_params(), "prodigy", 1.0, {})
    adapter = OptimizerRuntimeAdapter.inspect(opt, optimizer_type="prodigy")
    adapter.validate_for_train_config(_train_cfg(per_image_adaptive_lr_mode="loss"))
    with pytest.raises(ValueError, match="d-hat"):
        adapter.validate_for_train_config(_train_cfg())


def test_adafactor_scale_parameter_fails_closed():
    opt = get_optimizer(_params(), "adafactor", 1e-3, {"scale_parameter": True})
    adapter = OptimizerRuntimeAdapter.inspect(opt, optimizer_type="adafactor")
    with pytest.raises(ValueError, match="scale_parameter"):
        adapter.validate_for_train_config(_train_cfg())


def test_fused_backward_rejects_multi_backward_windows():
    opt = get_optimizer(_params(), "automagic2", 1e-6, {})
    adapter = OptimizerRuntimeAdapter.inspect(opt, optimizer_type="automagic2")
    adapter.validate_for_train_config(_train_cfg())
    with pytest.raises(ValueError, match="fused-backward"):
        adapter.validate_for_train_config(_train_cfg(gradient_accumulation=2))
    with pytest.raises(ValueError, match="fused-backward"):
        adapter.validate_for_train_config(_train_cfg(gradient_accumulation_steps=4))
    with pytest.raises(ValueError, match="fused-backward"):
        adapter.validate_for_train_config(_train_cfg(single_item_batching=True))
    with pytest.raises(ValueError, match="mean_flow"):
        adapter.validate_for_train_config(_train_cfg(loss_type="mean_flow"))


def test_group_lr_scale_restored_after_step_and_exception():
    model = torch.nn.Linear(4, 4, bias=False)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    adapter = OptimizerRuntimeAdapter.inspect(opt)
    base = opt.param_groups[0]["lr"]
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    adapter.begin_window(opt, 0.5)
    assert opt.param_groups[0]["lr"] == pytest.approx(base * 0.5)
    opt.step()
    adapter.end_window(opt)
    assert opt.param_groups[0]["lr"] == pytest.approx(base)

    adapter.begin_window(opt, 0.25)
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        pass
    finally:
        adapter.end_window(opt)
    assert opt.param_groups[0]["lr"] == pytest.approx(base)



def test_rose_scale_one_matches_disabled_and_scales_delta():
    def make_opt(params):
        return get_optimizer(params, "rose", 1e-2, {"bf16_sr": False, "compute_dtype": "fp32"})

    d1, opt1, _ = _one_step(make_opt, 1.0)
    torch.manual_seed(0)
    model = torch.nn.Linear(8, 8, bias=False)
    opt = make_opt(model.parameters())
    x = torch.randn(4, 8)
    y = torch.randn(4, 8)
    opt.zero_grad(set_to_none=True)
    F.mse_loss(model(x), y).backward()
    before = model.weight.detach().clone()
    opt.step()
    d_off = model.weight.detach() - before
    assert torch.allclose(d1, d_off, atol=0, rtol=0)
    assert opt1.state == {}

    d_lo, _, _ = _one_step(make_opt, 0.5)
    d_hi, _, _ = _one_step(make_opt, 2.0)
    d_zero, _, _ = _one_step(make_opt, 0.0)
    # Rose is linear in lr; 2.0 vs 0.5 is exactly 4x. 0.0 is a no-op update.
    assert torch.allclose(d_hi, d_lo * 4, atol=1e-6, rtol=1e-5)
    assert d_lo.abs().sum() < d_hi.abs().sum()
    assert torch.allclose(d_zero, torch.zeros_like(d_zero), atol=0, rtol=0)


def test_adam_scale_one_matches_disabled_and_preserves_moments():
    def make_opt(params):
        return get_optimizer(params, "adam", 1e-3, {})

    d1, opt1, _ = _one_step(make_opt, 1.0)
    d_off, opt_off, _ = _one_step(lambda p: get_optimizer(p, "adam", 1e-3, {}), 1.0)
    # scale 1.0 vs begin_window(1.0) which is a no-op vs a second identical run.
    assert torch.allclose(d1, d_off, atol=0, rtol=0)

    d_lo, opt_lo, _ = _one_step(make_opt, 0.5)
    d_hi, opt_hi, _ = _one_step(make_opt, 2.0)
    assert torch.allclose(d_hi, d_lo * 4, atol=1e-6, rtol=1e-5)

    def _moment(opt):
        p = next(iter(opt.state))
        return opt.state[p]["exp_avg"].clone(), opt.state[p]["exp_avg_sq"].clone()

    m_lo, v_lo = _moment(opt_lo)
    m_hi, v_hi = _moment(opt_hi)
    assert torch.allclose(m_lo, m_hi)
    assert torch.allclose(v_lo, v_hi)


def test_native_automagic_scales_update_without_touching_lr_mask():
    def make_opt(params):
        return get_optimizer(params, "automagic", 1e-6, {"weight_decay": 0.0})

    d_lo, opt_lo, _ = _one_step(make_opt, 0.5)
    d_hi, opt_hi, _ = _one_step(make_opt, 2.0)
    assert torch.allclose(d_hi, d_lo * 4, atol=1e-6, rtol=1e-4)
    p_lo = next(iter(opt_lo.state))
    p_hi = next(iter(opt_hi.state))
    assert torch.allclose(
        opt_lo.state[p_lo]["lr_mask"].dequantize(),
        opt_hi.state[p_hi]["lr_mask"].dequantize(),
    )


def test_native_scale_not_serialized():
    model = torch.nn.Linear(4, 4, bias=False)
    opt = get_optimizer(model.parameters(), "automagic", 1e-6, {})
    adapter = OptimizerRuntimeAdapter.inspect(opt)
    adapter.begin_window(opt, 2.0)
    assert getattr(opt, "_runtime_step_scale") == 2.0
    blob = opt.state_dict()
    assert "_runtime_step_scale" not in blob
    assert "_runtime_step_scale" not in blob.get("param_groups", [{}])[0]
    adapter.end_window(opt)
    assert getattr(opt, "_runtime_step_scale") == 1.0


def test_end_window_is_idempotent():
    model = torch.nn.Linear(2, 2, bias=False)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    adapter = OptimizerRuntimeAdapter.inspect(opt)
    adapter.end_window(opt)
    adapter.end_window(opt)
    adapter.begin_window(opt, 0.5)
    adapter.end_window(opt)
    adapter.end_window(opt)
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bnb 8-bit kernels need CUDA")
def test_bnb_adamw8bit_group_lr_conformance():
    def make_opt(params):
        return get_optimizer(params, "adamw8bit", 1e-3, {})

    d_lo, opt_lo, adapter = _one_step(make_opt, 0.5)
    d_hi, opt_hi, _ = _one_step(make_opt, 2.0)
    assert adapter.supports_step_scale
    assert torch.allclose(d_hi, d_lo * 4, atol=1e-5, rtol=1e-4)
