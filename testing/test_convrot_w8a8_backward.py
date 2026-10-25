"""Dedicated W8A8 backward suite (model.convrot8_backward: w8a8).

Numerical gates are the tolerances declared by the validation study that
accepted the implementation (do not relax them to make a run pass):
input-gradient NRMS <= 2.5% vs fp64 (same noise class as the forward's own
activation quantization), cosine >= 0.999, W8A8-vs-original NRMS ratio <= 1.5x,
paired LoRA-gradient NRMS <= 2% with cosine >= 0.9995, and paired AdamW update
relative delta <= 2%.

Reuses the quantized-linear builder and metrics from the kitchen suite; the
kitchen suite itself must stay green with the W8A8 flag untouched (default
off), which test_disabled_by_default pins here.
"""

import copy
import os

import pytest
import torch

from test_convrot_kitchen import _AdapterChain, _linear, _metrics

import toolkit.util.convrot_quant as cq

CUDA = torch.cuda.is_available()
DEV = "cuda" if CUDA else "cpu"

T_NRMS_VS_FP64 = 0.025
T_COSINE = 0.999
T_VS_FWD_QUANT_RATIO = 1.5  # W8A8 NRMS / the forward's own per-row RTN act-quant NRMS
T_LORA_NRMS = 0.02
T_LORA_COSINE = 0.9995
T_ADAM_REL_DELTA = 0.02


@pytest.fixture(autouse=True)
def _restore_w8a8_state():
    prev = cq._INT8_W8A8_BACKWARD
    yield
    cq._INT8_W8A8_BACKWARD = prev
    cq._int8_w8a8_nonfinite_acc = None


def _fp64_truth(dy, module):
    q = module.cr8_qdata.double()
    s = module.cr8_scales.view(torch.float32).double()
    return dy.double() @ (q * s[:, None])


def _dequant_reference(dy, module):
    s = module.cr8_scales.view(torch.float32).to(dy.dtype)
    w = module.cr8_qdata.to(dy.dtype) * s.unsqueeze(1)
    return dy @ w


def _w8a8(dy, module):
    cq.set_w8a8_backward(True)
    try:
        return cq._int8_input_gradient(dy, module.cr8_qdata, module.cr8_scales).clone()
    finally:
        cq.set_w8a8_backward(False)


def _forward_quant_floor(dy):
    # NRMS of the forward's own per-row RTN activation quantization on the
    # same rows — the noise class the implementation is accepted to match
    q, s = cq.quantize_int8_rows(dy.float(), 127)
    recon = q.float() * s.unsqueeze(1)
    return _metrics(recon, dy.float())["nrms"]


def _case_gates(name, out, module, dy, floor_ratio=True):
    truth = _fp64_truth(dy, module)
    m_w = _metrics(out, truth)
    assert torch.isfinite(out.float()).all(), f"{name}: non-finite output"
    assert m_w["nrms"] <= T_NRMS_VS_FP64, f"{name}: NRMS {m_w['nrms']:.4%} > {T_NRMS_VS_FP64:.2%}"
    assert m_w["cosine"] >= T_COSINE, f"{name}: cosine {m_w['cosine']:.6f} < {T_COSINE}"
    # the floor ratio is only meaningful on well-scaled inputs: a single huge
    # row dominates the input-space norm and dilutes the floor metric (the
    # edge-input cases gate on the absolute bounds, like the validation study)
    if floor_ratio:
        ratio = m_w["nrms"] / max(_forward_quant_floor(dy), 1e-4)
        assert ratio <= T_VS_FWD_QUANT_RATIO, (
            f"{name}: W8A8/forward-quant NRMS ratio {ratio:.2f} > {T_VS_FWD_QUANT_RATIO}"
        )


def test_flag_setter_roundtrip():
    cq.set_w8a8_backward(True)
    assert cq.w8a8_backward_enabled() is True
    cq.set_w8a8_backward(False)
    assert cq.w8a8_backward_enabled() is False


def test_resolve_resets_between_sequential_jobs():
    # env unset in the test process: a native load after a w8a8 job must
    # reset the process-wide flag rather than inherit it
    assert cq.w8a8_backward_env_requested() is False
    cq.set_w8a8_backward(True)  # simulate the previous job's leftover state
    enabled, source = cq.resolve_w8a8_backward("native")
    assert enabled is False and source is None
    assert cq.w8a8_backward_enabled() is False

    enabled, source = cq.resolve_w8a8_backward("w8a8")
    assert enabled is True and source == "config"
    assert cq.w8a8_backward_enabled() is True

    enabled, source = cq.resolve_w8a8_backward("native")
    assert enabled is False and source is None


def test_disabled_by_default_matches_dequant_bitexact():
    # the committed default must be the original backward, bit for bit
    module = _linear(256, 192, device=DEV, dtype=torch.bfloat16, seed=11)
    dy = torch.randn(48, 192, device=DEV, dtype=torch.bfloat16, generator=torch.Generator(DEV).manual_seed(3))
    cq.set_w8a8_backward(False)
    out = cq._int8_input_gradient(dy, module.cr8_qdata, module.cr8_scales)
    assert torch.equal(out, _dequant_reference(dy, module))


def test_config_validation():
    from toolkit.config_modules import ModelConfig

    ok = ModelConfig(
        name_or_path="x", arch="krea2", quantize=True, qtype="convrot8",
        convrot8_backward="w8a8",
    )
    assert ok.convrot8_backward == "w8a8"

    default = ModelConfig(name_or_path="x", arch="krea2", quantize=True, qtype="convrot8")
    assert default.convrot8_backward == "native"

    with pytest.raises(ValueError, match="convrot8_backward"):
        ModelConfig(name_or_path="x", arch="krea2", qtype="convrot8",
                    convrot8_backward="int4")
    with pytest.raises(ValueError, match="krea2 and qtype: convrot8"):
        ModelConfig(name_or_path="x", arch="flux", quantize=True, qtype="convrot8",
                    convrot8_backward="w8a8")
    with pytest.raises(ValueError, match="krea2 and qtype: convrot8"):
        ModelConfig(name_or_path="x", arch="krea2", quantize=True, qtype="qfloat8",
                    convrot8_backward="w8a8")


@pytest.mark.skipif(not CUDA, reason="requires CUDA int8 tensor cores")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "in_f,out_f,m_rows",
    [(6144, 16384, 512), (16384, 6144, 512), (2560, 6912, 256), (6912, 2560, 97)],
)
def test_w8a8_accuracy_krea_shapes(in_f, out_f, m_rows, dtype):
    module = _linear(in_f, out_f, device=DEV, dtype=dtype, seed=17)
    dy = torch.randn(m_rows, out_f, device=DEV, dtype=dtype) * 0.05
    out = _w8a8(dy, module)
    _case_gates(f"{in_f}x{out_f} {dtype}", out, module, dy)


@pytest.mark.skipif(not CUDA, reason="requires CUDA")
def test_w8a8_edge_inputs():
    module = _linear(1536, 2048, device=DEV, dtype=torch.bfloat16, seed=19)
    base = torch.randn(128, 2048, device=DEV, dtype=torch.bfloat16) * 0.05

    out = _w8a8(torch.zeros_like(base), module)
    assert torch.count_nonzero(out).item() == 0

    _case_gates("fp32-dtype", _w8a8(base.float(), module), module, base.float())

    outliers = base.clone()
    outliers[3] *= 100.0
    _case_gates("outlier-row", _w8a8(outliers, module), module, outliers, floor_ratio=False)

    sat = base.clone()
    sat[7, 0] = 1e4
    _case_gates("saturation", _w8a8(sat, module), module, sat, floor_ratio=False)

    wide = torch.randn(128, 4096, device=DEV, dtype=torch.bfloat16) * 0.05
    noncontig = wide[:, ::2]
    _case_gates("noncontiguous", _w8a8(noncontig, module), module, noncontig.contiguous())


@pytest.mark.skipif(not CUDA, reason="requires CUDA")
def test_w8a8_fallback_paths_match_dequant():
    module = _linear(320, 256, device=DEV, dtype=torch.bfloat16, seed=23)
    dy = torch.randn(64, 256, device=DEV, dtype=torch.bfloat16)

    cpu_module = _linear(320, 256, device="cpu", dtype=torch.bfloat16, seed=23)
    cpu_dy = dy.to("cpu")
    cq.set_w8a8_backward(True)
    try:
        cpu_out = cq._int8_input_gradient(
            cpu_dy, cpu_module.cr8_qdata, cpu_module.cr8_scales
        )
        batched = cq._int8_input_gradient(
            dy.unsqueeze(0), module.cr8_qdata, module.cr8_scales
        )
        empty = cq._int8_input_gradient(
            dy[:0], module.cr8_qdata, module.cr8_scales
        )
    finally:
        cq.set_w8a8_backward(False)

    assert torch.equal(cpu_out, _dequant_reference(cpu_dy, cpu_module))
    assert torch.equal(batched, _dequant_reference(dy.unsqueeze(0), module))
    assert empty.shape == (0, 320)


@pytest.mark.skipif(not CUDA, reason="requires CUDA")
def test_nonfinite_row_accounting():
    module = _linear(512, 384, device=DEV, dtype=torch.bfloat16, seed=29)
    dy = torch.randn(32, 384, device=DEV, dtype=torch.bfloat16) * 0.05
    dy[0] = float("nan")
    dy[1] = float("inf")
    out = _w8a8(dy, module)
    # NaN-only rows sanitize inside the int8 clamp to finite near-zero output;
    # inf rows leave a non-finite scale, output row, and a counted diagnostic.
    assert torch.isfinite(out[0]).all()
    assert not torch.isfinite(out[1]).all()
    assert cq.drain_w8a8_nonfinite() == 1
    assert cq.drain_w8a8_nonfinite() == 0


@pytest.mark.skipif(not CUDA, reason="requires CUDA")
def test_lora_chain_gradients_and_adamw_update():
    torch.manual_seed(5)
    bases = [_linear(6144, 6144, device=DEV, dtype=torch.bfloat16, seed=s) for s in (21, 22, 23)]
    chain = _AdapterChain(bases)
    x = torch.randn(2, 17, 6144, device=DEV, dtype=torch.bfloat16, requires_grad=True)

    def chain_grads(w8a8: bool):
        cq.set_w8a8_backward(w8a8)
        try:
            out = chain(x)
            loss = out.float().mean()
            params = list(chain.adapters.parameters())
            return out, torch.autograd.grad(loss, [x, *params])
        finally:
            cq.set_w8a8_backward(False)

    out_off, grads_off = chain_grads(False)
    out_on, grads_on = chain_grads(True)
    assert torch.equal(out_off, out_on), "forward must be unaffected by the backward mode"

    worst_nrms, worst_cos = 0.0, 1.0
    for a, b in zip(grads_on[1:], grads_off[1:]):
        assert a.count_nonzero().item() > 0, "disconnected LoRA gradient"
        mtr = _metrics(a.float(), b.float())
        worst_nrms = max(worst_nrms, mtr["nrms"])
        worst_cos = min(worst_cos, mtr["cosine"])
    assert worst_nrms <= T_LORA_NRMS, f"LoRA grad NRMS {worst_nrms:.4%} > {T_LORA_NRMS:.0%}"
    assert worst_cos >= T_LORA_COSINE, f"LoRA grad cosine {worst_cos:.6f} < {T_LORA_COSINE}"

    params = list(chain.adapters.parameters())

    # warm the optimizer state with a fixed dummy step so the compared steps
    # operate in the m/sqrt(v) regime; from empty state the first AdamW step
    # is ~sign(g)*lr and sign flips of near-zero elements dominate the delta
    warm_opt = torch.optim.AdamW(params, lr=1e-3, foreach=False)
    for p in params:
        p.grad = torch.full_like(p, 0.01)
    warm_opt.step()
    state_snap = copy.deepcopy(warm_opt.state_dict())
    param_snap = [p.detach().clone() for p in params]

    def adam_step_with(grads):
        for p, snap in zip(params, param_snap):
            p.data.copy_(snap)
        opt = torch.optim.AdamW(params, lr=1e-3, foreach=False)
        opt.load_state_dict(copy.deepcopy(state_snap))
        opt.zero_grad(set_to_none=True)
        for p, g in zip(params, grads[1:]):
            p.grad = g.detach().clone()
        opt.step()
        return [p.detach().clone() for p in params]

    upd_off = adam_step_with(grads_off)
    upd_on = adam_step_with(grads_on)
    worst = max(
        ((a.float() - b.float()).norm()
         / (b.float() - s.float()).norm().clamp_min(1e-30)).item()
        for a, b, s in zip(upd_on, upd_off, param_snap)
    )
    assert worst <= T_ADAM_REL_DELTA, (
        f"AdamW update relative delta {worst:.4%} > {T_ADAM_REL_DELTA:.0%}"
    )
