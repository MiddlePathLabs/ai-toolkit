"""Behavioral gates for the optional Krea ConvRot8 training backend.

Run with CONVROT_KITCHEN_REQUIRE_CUDA=1 for an acceptance run: unavailable CUDA,
BF16, kitchen, or compilation is then a failure, not a passing skipped gate.
Forward bounds are deterministic INT8-grid/roundoff bounds, not a convergence
quality claim. No model checkpoint, tokenizer, or dataset is downloaded.
"""

import copy
import importlib
import json
import math
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from toolkit.util import convrot_kitchen as kitchen
from toolkit.util.convrot_quant import ConvRotInt8Quantizer, rotate
from toolkit.util.ostris_quant import (
    convert_linear_to_ostris,
    load_quantized_layers,
    save_quantized_layers,
)

# KREA2_MMDIT_CONFIG: width=6144, heads=48, kvheads=12, txtdim=2560,
# multiplier=4. SwiGLU rounds int(2*width/3)*4 up to a multiple of 128.
KREA_SHAPES = [
    (6144, 6144),
    (6144, 1536),
    (6144, 16384),
    (16384, 6144),
    (2560, 2560),
    (2560, 6912),
    (6912, 2560),
]
DTYPES = [torch.bfloat16, torch.float16]
REQUIRE_CUDA = os.environ.get("CONVROT_KITCHEN_REQUIRE_CUDA") == "1"


def _unavailable(reason):
    if REQUIRE_CUDA:
        pytest.fail(f"Required accelerator validation unavailable: {reason}")
    pytest.skip(reason)


@pytest.fixture
def cuda_device():
    if not torch.cuda.is_available():
        _unavailable("CUDA unavailable")
    device = torch.device("cuda", torch.cuda.current_device())
    try:
        importlib.import_module("comfy_kitchen.backends.cuda")
    except (ImportError, OSError) as exc:
        _unavailable(f"kitchen CUDA backend unavailable: {exc}")
    if torch.cuda.get_device_capability(device) < (8, 0):
        _unavailable("kitchen fused training requires sm80 or newer")
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield device
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def _linear(
    in_features=256,
    out_features=256,
    *,
    device="cpu",
    dtype=torch.float32,
    bias=True,
    seed=123,
):
    torch.manual_seed(seed)
    module = torch.nn.Linear(
        in_features, out_features, bias=bias, device=device, dtype=dtype
    )
    module.requires_grad_(False)
    quantizer = ConvRotInt8Quantizer()
    quantizer.qtype = "convrot8"
    assert convert_linear_to_ostris(module, quantizer)
    return module


def _pair(*args, **kwargs):
    native = _linear(*args, **kwargs)
    candidate = copy.deepcopy(native)
    return native, candidate


def _configure(module, *, strict=True, compiled=False):
    return kitchen.configure_krea_convrot8(
        module, backend="comfy_kitchen", strict=strict, compiled=compiled
    )


def _metrics(actual, expected):
    a, b = actual.detach().double().flatten(), expected.detach().double().flatten()
    delta = a - b
    tiny = torch.finfo(torch.float64).tiny
    rms = b.square().mean().sqrt().clamp_min(tiny)
    # Relative maxima use one storage ULP at reference RMS near zero; an
    # elementwise relative error at exactly zero has no meaningful finite value.
    floor = rms * torch.finfo(expected.dtype).eps
    an, bn = a.norm(), b.norm()
    cosine = torch.where(
        (an > 0) & (bn > 0),
        (a @ b) / (an * bn).clamp_min(tiny),
        ((an == 0) & (bn == 0)).double(),
    )
    return {
        "nrms": (delta.square().mean().sqrt() / rms).item(),
        "maxabs": delta.abs().max().item(),
        "maxrel": (delta.abs() / b.abs().clamp_min(floor.clamp_min(tiny))).max().item(),
        "cosine": cosine.item(),
        "normdiff": ((an - bn).abs() / bn.clamp_min(tiny)).item(),
    }


def _record(record_property, name, actual, expected):
    metrics = _metrics(actual, expected)
    record_property(name, json.dumps(metrics, sort_keys=True))
    assert torch.isfinite(actual).all(), (name, metrics)
    assert torch.isfinite(expected).all(), (name, metrics)
    return metrics


def _assert_roundoff(actual, expected, *, operations=2):
    # Standard gamma_n bound for n floating operations. The absolute floor is
    # one storage ULP at RMS, avoiding an arbitrary near-zero allclose constant.
    eps = torch.finfo(expected.dtype).eps
    gamma = operations * eps / (1 - operations * eps)
    floor = expected.detach().double().square().mean().sqrt().item() * eps
    torch.testing.assert_close(actual, expected, rtol=gamma, atol=floor)


def _assert_quality(actual, native, precise, dtype_control, record_property, name):
    """Do not add more RMS error than the native grid plus dtype-only drift."""
    a, n, p, d = [
        value.detach().double() for value in (actual, native, precise, dtype_control)
    ]

    def rms(value):
        return value.square().mean().sqrt().item()

    baseline_error = rms(n - p)
    dtype_error = rms(d - p)
    floor = torch.finfo(actual.dtype).eps * rms(p)
    limit = baseline_error + dtype_error + floor
    actual_error = rms(a - p)
    record_property(
        f"{name}_quality_budget",
        json.dumps(
            {
                "native_rms_error": baseline_error,
                "dtype_control_rms_error": dtype_error,
                "storage_ulp_floor": floor,
                "allowed_rms_error": limit,
                "candidate_rms_error": actual_error,
            },
            sort_keys=True,
        ),
    )
    assert actual_error <= limit, (name, actual_error, limit)
    metrics = _record(record_property, f"{name}_quality_parity", actual, native)
    # Triangle inequality gives the pairwise NRMS/norm-difference limit.
    # Geometry of the radius-t ball around a nonzero native tensor gives
    # cosine >= sqrt(1-t^2); this is not an unrelated fixed cosine tolerance.
    pair_limit = (limit + baseline_error) / max(rms(n), torch.finfo(torch.float64).tiny)
    assert metrics["nrms"] <= pair_limit
    assert metrics["normdiff"] <= pair_limit
    if pair_limit < 1:
        assert (
            metrics["cosine"]
            >= math.sqrt(1 - pair_limit**2) - 8 * torch.finfo(torch.float64).eps
        )


def _dtype_normalization_control(rotated, dtype):
    # v0.2.40 int8_linear.cu::quant_div_float_to_float: absmax/scale
    # selection precedes numerator, denominator, and quotient dtype rounding.
    # Remove only nearbyintf, not the floating normalization arithmetic.
    scale = rotated.abs().amax(dim=-1, keepdim=True).clamp_max(torch.finfo(dtype).max)
    scale = (scale * (1.0 / 127.0)).clamp_min(1e-30)
    numerator = rotated.to(dtype).float()
    denominator = scale.to(dtype).float()
    normalized = (numerator / denominator).to(dtype).float()
    # A scale that underflows in FP16 has no finite normalization. Preserve
    # signed saturation for nonzero inputs and the continuous zero-row value;
    # never invent a larger scale or allow NaNs into the calibration control.
    normalized = torch.nan_to_num(normalized, nan=0.0, posinf=127.0, neginf=-128.0)
    return normalized * scale


def _forward_gate(native, candidate, x, record_property):
    yn, yk = native(x), candidate(x)
    with torch.no_grad():
        xf = x.detach().float()
        rot = native.cr8_rot_size
        exact_rot = rotate(xf, rot)
        rounded_rot = rotate(x.detach(), rot).float()
        w = native.cr8_qdata.float() * native.cr8_scales.view(torch.float32)[:, None]
        reference = F.linear(
            exact_rot.reshape(-1, native.in_features),
            w,
            None if native.bias is None else native.bias.float(),
        )
        reference = reference.reshape_as(yn)
        dtype_reference = (
            F.linear(
                _dtype_normalization_control(exact_rot, x.dtype).reshape(
                    -1, native.in_features
                ),
                w,
                None if native.bias is None else native.bias.float(),
            )
            .reshape_as(yn)
            .to(x.dtype)
            .float()
        )
        _assert_quality(
            yk.float(),
            yn.float(),
            reference,
            dtype_reference,
            record_property,
            "forward",
        )
        cuda, _ = kitchen._load_kitchen()
        if x.dtype == torch.float16:
            qa, sa = cuda.quantize_int8_rowwise(
                rotate(x.detach(), rot)
                .float()
                .reshape(-1, native.in_features)
                .contiguous()
            )
        else:
            qa, sa = cuda.quantize_int8_rowwise_convrot64(
                x.detach().float().reshape(-1, native.in_features).contiguous(), rot
            )
        rows = qa.shape[0]
        qa = F.pad(qa, (0, 0, 0, (-rows) % 32))
        integer_sum = torch._int_mm(qa, native.cr8_qdata.t())[:rows].float()
        grid_output = integer_sum * sa.reshape(-1, 1)
        grid_output *= native.cr8_scales.view(torch.float32).reshape(1, -1)
        if native.bias is not None:
            grid_output += native.bias.float()
        grid_output = grid_output.reshape_as(yk).to(yk.dtype)
        _record(
            record_property, "kitchen_vs_quantized_integer_reference", yk, grid_output
        )
        _assert_roundoff(yk, grid_output, operations=2)
        # Each independently rounded signed-INT8 activation is at most half a
        # step from its rotated value. FHT has log2(rot) butterfly additions;
        # gamma_n bounds its arithmetic, plus output rounding. This conservative
        # L1 propagation remains valid for correlated rounding/outlier rows.
        eps = torch.finfo(x.dtype).eps
        stages = int(math.log2(rot)) + 1
        gamma = stages * eps / (1 - stages * eps)
        block_abs = xf.abs().reshape(-1, native.in_features // rot, rot)
        rotation_bound = block_abs.sum(-1, keepdim=True) / math.sqrt(rot)
        rotation_bound = (
            rotation_bound.expand_as(block_abs).reshape_as(exact_rot) * gamma
        )
        step = (exact_rot.abs() + rotation_bound).amax(-1, keepdim=True) / 127
        activation_bound = step + rotation_bound + (rounded_rot - exact_rot).abs()
        envelope = activation_bound.reshape(-1, native.in_features) @ w.abs().t()
        envelope = envelope.reshape_as(yn) + eps * (yn.float().abs() + yk.float().abs())
        _record(record_property, "native_vs_fp32", yn.float(), reference)
        _record(record_property, "kitchen_vs_fp32", yk.float(), reference)
        _record(record_property, "kitchen_vs_native", yk, yn)
        record_property("forward_grid_bound_maxabs", envelope.max().item())
        assert ((yk.float() - yn.float()).abs() <= envelope).all()
    return yn, yk


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "shape", KREA_SHAPES, ids=lambda shape: f"{shape[0]}x{shape[1]}"
)
def test_representative_krea_forward_and_input_gradient(
    cuda_device, dtype, shape, record_property
):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        _unavailable("BF16 unsupported")
    native, candidate = _pair(*shape, device=cuda_device, dtype=dtype, bias=False)
    before = {name: value.clone() for name, value in native.named_buffers()}
    diagnostics = _configure(candidate)
    x = torch.randn(
        1, 33, shape[0], device=cuda_device, dtype=dtype, requires_grad=True
    )
    yn, yk = _forward_gate(native, candidate, x, record_property)
    dy = torch.randn_like(yn)
    gn = torch.autograd.grad(yn, x, dy, retain_graph=True)[0]
    gk = torch.autograd.grad(yk, x, dy, retain_graph=True)[0]
    # The unchanged derivative casts scales BEFORE multiplication, unlike an
    # FP32 dequantization followed by a low-precision cast.
    w = (
        native.cr8_qdata.to(dtype)
        * native.cr8_scales.view(torch.float32).to(dtype)[:, None]
    )
    expected = rotate((dy.reshape(-1, shape[1]) @ w).reshape_as(x), native.cr8_rot_size)
    _record(record_property, "input_gradient", gk, gn)
    _assert_roundoff(gk, expected)
    _assert_roundoff(gk, gn)
    assert torch.equal(torch.autograd.grad(yk, x, dy)[0], gk)
    for name, value in candidate.named_buffers():
        assert torch.equal(value, before[name]), name
        assert not value.requires_grad
    assert all(parameter.grad is None for parameter in candidate.parameters())
    record_property(
        "backend", json.dumps(diagnostics.require_complete(), sort_keys=True)
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "case", ["noncontiguous", "irregular", "zero", "rounding", "tiny", "outlier"]
)
def test_activation_edges(cuda_device, dtype, case, record_property):
    native, candidate = _pair(device=cuda_device, dtype=dtype)
    diagnostics = _configure(candidate)
    x = torch.randn(2, 17, 256, device=cuda_device, dtype=dtype)
    if case == "noncontiguous":
        x = x.transpose(0, 1)
        assert not x.is_contiguous()
    elif case == "irregular":
        x = x[:, :1, :]
    elif case == "zero":
        x.zero_()
    elif case == "rounding":
        grid = torch.arange(256, device=cuda_device).float() % 127 - 63 + 0.5
        grid[-1] = 127
        x = rotate(grid.expand_as(x).to(dtype), 256)
    elif case == "tiny":
        x *= torch.finfo(dtype).tiny * 127
    elif case == "outlier":
        x[..., 0] = 128
        x[..., 1:] *= 0.001
    x.requires_grad_()
    yn, yk = _forward_gate(native, candidate, x, record_property)
    dy = torch.randn_like(yn)
    gn = torch.autograd.grad(yn, x, dy)[0]
    gk = torch.autograd.grad(yk, x, dy)[0]
    _record(record_property, "edge_input_gradient", gk, gn)
    _assert_roundoff(gk, gn)
    if case == "zero":
        torch.testing.assert_close(yk, candidate.bias.expand_as(yk), rtol=0, atol=0)
    record_property(
        "backend", json.dumps(diagnostics.require_complete(), sort_keys=True)
    )


class _Network:
    network_type = "lora"
    is_lorm = False
    is_active = True
    is_merged_in = False
    _multiplier = 1.0

    def multiplier_for_batch(self, batch_size):
        return torch.ones(batch_size, device=self.device)


class _AdapterChain(torch.nn.Module):
    def __init__(self, bases):
        super().__init__()
        from toolkit.lora_special import LoRAModule

        self.network = _Network()
        self.network.device = bases[0].cr8_qdata.device
        self.bases = torch.nn.ModuleList(bases)
        self.adapters = torch.nn.ModuleList()
        for name, base in zip(("upstream", "middle", "downstream"), bases):
            adapter = LoRAModule(
                name, base, lora_dim=16, alpha=16, network=self.network
            )
            adapter.to(device=self.network.device, dtype=torch.float32)
            with torch.no_grad():
                # Standard zero-up initialization cannot exercise down gradients.
                adapter.lora_down.weight.normal_(std=0.02)
                adapter.lora_up.weight.normal_(std=0.02)
            adapter.apply_to()
            self.adapters.append(adapter)

    def forward(self, x):
        for layer in self.bases:
            x = layer(x)
        return x


def _chain_pair(device, dtype):
    native_bases = [_linear(device=device, dtype=dtype, seed=seed) for seed in range(3)]
    kitchen_bases = [copy.deepcopy(base) for base in native_bases]
    native, candidate = _AdapterChain(native_bases), _AdapterChain(kitchen_bases)
    for actual, expected in zip(
        candidate.adapters.parameters(), native.adapters.parameters()
    ):
        actual.data.copy_(expected.data)
    return native, candidate


class _ReferenceSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, base):
        ctx.base = base
        # Preserve the training forward value even though Function.forward runs
        # under no_grad; only the independent reference backward is substituted.
        with torch.enable_grad():
            value = base.ostris_quantizer.forward(base, x.detach().requires_grad_(True))
        return value.detach()

    @staticmethod
    def backward(ctx, grad):
        base = ctx.base
        w = base.cr8_qdata.to(grad.dtype)
        w = w * base.cr8_scales.view(torch.float32).to(grad.dtype)[:, None]
        gx = grad.reshape(-1, base.out_features) @ w
        gx = rotate(gx, base.cr8_rot_size).reshape(*grad.shape[:-1], base.in_features)
        return gx, None


class _FloatingSTE(_ReferenceSTE):
    @staticmethod
    def forward(ctx, x, base):
        ctx.base = base
        rotated = rotate(x.float(), base.cr8_rot_size)
        w = base.cr8_qdata.float() * base.cr8_scales.view(torch.float32)[:, None]
        return F.linear(
            rotated, w, None if base.bias is None else base.bias.float()
        ).to(x.dtype)


class _DtypeControlSTE(_ReferenceSTE):
    @staticmethod
    def forward(ctx, x, base):
        ctx.base = base
        rotated = _dtype_normalization_control(
            rotate(x.float(), base.cr8_rot_size), x.dtype
        )
        w = base.cr8_qdata.float() * base.cr8_scales.view(torch.float32)[:, None]
        return F.linear(
            rotated, w, None if base.bias is None else base.bias.float()
        ).to(x.dtype)


def _reference_chain(model, x, floating=None):
    for base, adapter in zip(model.bases, model.adapters):
        operator = {"precise": _FloatingSTE, "dtype": _DtypeControlSTE}.get(
            floating, _ReferenceSTE
        )
        value = operator.apply(x, base)
        delta = adapter.lora_up(adapter.lora_down(x.float()))
        x = value + delta.to(value.dtype)
    return x


def _grads(
    model, x, *, reference=False, checkpointed=False, accumulation=1, floating=None
):
    call = (
        (lambda value: _reference_chain(model, value, floating))
        if reference or floating
        else model
    )
    output = checkpoint(call, x, use_reentrant=False) if checkpointed else call(x)
    # Identical external dY isolates the derivative from a changed forward loss.
    loss = output.float().mean() / accumulation
    parameters = list(model.adapters.parameters())
    gradients = torch.autograd.grad(loss, [x, *parameters])
    return output, gradients


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("checkpointed", [False, True], ids=["eager", "nonreentrant"])
def test_real_lora_upstream_downstream_and_checkpoint(
    cuda_device, dtype, checkpointed, record_property
):
    native, candidate = _chain_pair(cuda_device, dtype)
    diagnostics = _configure(candidate)
    x = torch.randn(2, 17, 256, device=cuda_device, dtype=dtype, requires_grad=True)
    yn, gn = _grads(native, x, checkpointed=checkpointed)
    yk, gk = _grads(candidate, x, checkpointed=checkpointed)
    yr, gr = _grads(candidate, x, reference=True)
    yp, gp = _grads(native, x, floating="precise")
    yd, gd = _grads(native, x, floating="dtype")
    _assert_quality(yk, yn, yp, yd, record_property, "chain_output")
    torch.testing.assert_close(yk, yr, rtol=0, atol=0)
    _record(record_property, "chain_output", yk, yn)
    names = ["input", *[name for name, _ in candidate.adapters.named_parameters()]]
    for name, actual, baseline, expected, precise, control in zip(
        names, gk, gn, gr, gp, gd
    ):
        _assert_quality(
            actual, baseline, precise, control, record_property, f"gradient_{name}"
        )
        assert actual.count_nonzero() > 0, name
        _assert_roundoff(actual, expected, operations=4)
    assert all(
        parameter.grad is None
        for base in candidate.bases
        for parameter in base.parameters()
    )
    record_property(
        "backend", json.dumps(diagnostics.require_complete(), sort_keys=True)
    )


def _adam_reference(parameter, gradient, state, group):
    beta1, beta2 = group["betas"]
    step = int(state["step"].item()) + 1
    first = state["exp_avg"] * beta1 + gradient * (1 - beta1)
    second = state["exp_avg_sq"] * beta2 + gradient.square() * (1 - beta2)
    denominator = second.sqrt() / math.sqrt(1 - beta2**step) + group["eps"]
    updated = parameter * (1 - group["lr"] * group["weight_decay"])
    updated = updated - (group["lr"] / (1 - beta1**step)) * first / denominator
    return updated, first, second


@pytest.mark.parametrize("dtype", DTYPES)
def test_accumulation_and_cloned_adamw_update(cuda_device, dtype, record_property):
    native, candidate = _chain_pair(cuda_device, dtype)
    diagnostics = _configure(candidate)
    parameters = [list(model.adapters.parameters()) for model in (native, candidate)]
    optimizers = [
        torch.optim.AdamW(values, lr=1e-3, foreach=False) for values in parameters
    ]
    # Populate nonzero moment state before cloning, not merely two empty states.
    for parameter in parameters[0]:
        parameter.grad = torch.full_like(parameter, 0.01)
    optimizers[0].step()
    for actual, expected in zip(parameters[1], parameters[0]):
        actual.data.copy_(expected.data)
    optimizers[1].load_state_dict(copy.deepcopy(optimizers[0].state_dict()))
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    states = [copy.deepcopy(optimizer.state_dict()) for optimizer in optimizers]
    starts = [[p.detach().clone() for p in values] for values in parameters]
    expected_accumulation = [None, None]
    control_accumulation = {mode: None for mode in ("precise", "dtype")}
    for microbatch in range(3):
        torch.manual_seed(500 + microbatch)
        value = torch.randn(1, 17, 256, device=cuda_device, dtype=dtype)
        for mode, accumulated in control_accumulation.items():
            x = value.detach().requires_grad_(True)
            _, control_grads = _grads(native, x, floating=mode, accumulation=3)
            if accumulated is None:
                control_accumulation[mode] = [
                    g.detach().clone() for g in control_grads[1:]
                ]
            else:
                for total, grad in zip(accumulated, control_grads[1:]):
                    total.add_(grad.detach())
        for index, model in enumerate((native, candidate)):
            x = value.detach().requires_grad_(True)
            _, reference_grads = _grads(model, x, reference=True, accumulation=3)
            if expected_accumulation[index] is None:
                expected_accumulation[index] = [
                    g.detach().clone() for g in reference_grads[1:]
                ]
            else:
                for total, grad in zip(
                    expected_accumulation[index], reference_grads[1:]
                ):
                    total.add_(grad.detach())
            output = checkpoint(model, x, use_reentrant=False)
            (output.float().mean() / 3).backward()
    for ordinal, (actual, baseline) in enumerate(zip(parameters[1], parameters[0])):
        _assert_quality(
            actual.grad,
            baseline.grad,
            control_accumulation["precise"][ordinal],
            control_accumulation["dtype"][ordinal],
            record_property,
            f"accumulated_gradient_{ordinal}",
        )
    control_updates = {}
    for mode, gradients in control_accumulation.items():
        control_updates[mode] = [
            _adam_reference(
                start,
                gradient,
                states[0]["state"][ordinal],
                optimizers[0].param_groups[0],
            )
            for ordinal, (start, gradient) in enumerate(zip(starts[0], gradients))
        ]
    for index, optimizer in enumerate(optimizers):
        expected_updates = []
        for ordinal, (parameter, start, expected_grad) in enumerate(
            zip(parameters[index], starts[index], expected_accumulation[index])
        ):
            _assert_roundoff(parameter.grad, expected_grad, operations=8)
            state = states[index]["state"][ordinal]
            expected_updates.append(
                _adam_reference(start, parameter.grad, state, optimizer.param_groups[0])
            )
        optimizer.step()
        for parameter, start, expected in zip(
            parameters[index], starts[index], expected_updates
        ):
            update, first, second = expected
            _assert_roundoff(parameter, update, operations=16)
            _assert_roundoff(optimizer.state[parameter]["exp_avg"], first, operations=4)
            _assert_roundoff(
                optimizer.state[parameter]["exp_avg_sq"], second, operations=4
            )
            assert not torch.equal(parameter, start)
    for ordinal, (actual, baseline) in enumerate(zip(parameters[1], parameters[0])):
        start = starts[0][ordinal]
        precise, control = (
            control_updates["precise"][ordinal],
            control_updates["dtype"][ordinal],
        )
        _assert_quality(
            actual - start,
            baseline - start,
            precise[0] - start,
            control[0] - start,
            record_property,
            f"adapter_update_{ordinal}",
        )
        for offset, name in enumerate(("exp_avg", "exp_avg_sq"), start=1):
            _assert_quality(
                optimizers[1].state[actual][name],
                optimizers[0].state[baseline][name],
                precise[offset],
                control[offset],
                record_property,
                f"{name}_{ordinal}",
            )
    record_property(
        "backend", json.dumps(diagnostics.require_complete(), sort_keys=True)
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_compiled_forward_and_backward(cuda_device, dtype, record_property):
    if not hasattr(torch, "compile"):
        _unavailable("torch.compile unavailable")
    _native, candidate = _pair(device=cuda_device, dtype=dtype)
    diagnostics = _configure(candidate, compiled=True)
    x = torch.randn(1, 33, 256, device=cuda_device, dtype=dtype, requires_grad=True)
    eager = candidate(x)
    dy = torch.randn_like(eager)
    expected = torch.autograd.grad(eager, x, dy)[0]
    # A counting backend using Inductor proves graph capture actually happened;
    # suppress_errors=False prevents eager fallback being counted as a pass.
    captures = []

    def backend(graph, inputs):
        from torch._dynamo.backends.registry import lookup_backend

        captures.append(graph)
        return lookup_backend("inductor")(graph, inputs)

    with torch._dynamo.config.patch(suppress_errors=False):
        compiled = torch.compile(candidate, backend=backend, fullgraph=True)
        actual = compiled(x)
        gradient = torch.autograd.grad(actual, x, dy)[0]
    assert captures, "compile executed no captured graph"
    _record(record_property, "compiled_output", actual, eager)
    _record(record_property, "compiled_gradient", gradient, expected)
    _assert_roundoff(actual, eager)
    _assert_roundoff(gradient, expected)
    record_property(
        "backend", json.dumps(diagnostics.require_complete(), sort_keys=True)
    )


@pytest.mark.parametrize(
    "training,grad_enabled", [(False, True), (False, False), (True, False)]
)
def test_inference_is_native(cuda_device, training, grad_enabled):
    native, candidate = _pair(device=cuda_device, dtype=torch.bfloat16)
    diagnostics = _configure(candidate)
    native.train(training)
    candidate.train(training)
    # Small M exercises the native inference GEMV, not just GEMM.
    x = torch.randn(
        1, 3, 256, device=cuda_device, dtype=torch.bfloat16, requires_grad=True
    )
    with torch.set_grad_enabled(grad_enabled):
        expected, actual = native(x), candidate(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert diagnostics.report()["observed_training_layers"] == 0


def test_native_configuration_is_noop():
    module = _linear()
    quantizer = module.ostris_quantizer
    buffers = dict(module.named_buffers())
    diagnostics = kitchen.configure_krea_convrot8(module, backend="native")
    assert module.ostris_quantizer is quantizer
    assert all(value is buffers[name] for name, value in module.named_buffers())
    json.dumps(diagnostics.report())
    assert diagnostics.report()["observed_training_layers"] == 0


def test_quantized_save_load_keeps_native_format(tmp_path, monkeypatch):
    native, candidate = _pair(dtype=torch.float32)
    monkeypatch.setattr(kitchen, "_load_kitchen", lambda: (SimpleNamespace(), "0.2.40"))
    _configure(candidate, strict=False)
    files = [tmp_path / "native.safetensors", tmp_path / "kitchen.safetensors"]
    for module, path in zip((native, candidate), files):
        save_quantized_layers({"projection": module}, str(path))
    from safetensors import safe_open
    from safetensors.torch import load_file

    a, b = [load_file(str(path)) for path in files]
    assert a.keys() == b.keys()
    for name in a:
        assert torch.equal(a[name], b[name]), name
    metadata = []
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata.append(handle.metadata())
    assert metadata[0] == metadata[1]
    restored = torch.nn.ModuleDict({"projection": torch.nn.Linear(256, 256)})
    assert load_quantized_layers(restored, str(files[1])) == 1
    assert type(restored["projection"].ostris_quantizer) is ConvRotInt8Quantizer
    assert restored["projection"].ostris_quantizer.qtype == "convrot8"
    x = torch.randn(2, 256)
    torch.testing.assert_close(restored["projection"](x), native(x), rtol=0, atol=0)
    # The normal dequantized state_dict contract also remains usable.
    ordinary = torch.nn.Linear(256, 256)
    ordinary.load_state_dict(
        {
            key: value.dequantize() if hasattr(value, "dequantize") else value
            for key, value in candidate.state_dict().items()
        }
    )
    torch.testing.assert_close(
        ordinary.weight, native.dequantize_weight(), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "failure",
    [
        ImportError("No module named comfy_kitchen"),
        ImportError("incompatible comfy-kitchen API"),
        OSError("CUDA DLL unavailable"),
    ],
    ids=["missing", "incompatible", "missing-dll"],
)
def test_dependency_fallback_and_strict_failure(monkeypatch, failure):
    def unavailable():
        raise failure

    monkeypatch.setattr(kitchen, "_load_kitchen", unavailable)
    native, candidate = _pair()
    original = candidate.ostris_quantizer
    with pytest.warns(UserWarning, match="using native"):
        diagnostics = _configure(candidate, strict=False)
    assert candidate.ostris_quantizer is original
    x = torch.randn(2, 256, requires_grad=True)
    expected, actual = native(x), candidate(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gn = torch.autograd.grad(expected.sum(), x)[0]
    gk = torch.autograd.grad(actual.sum(), x)[0]
    torch.testing.assert_close(gk, gn, rtol=0, atol=0)
    report = diagnostics.report()
    assert report["selected_backend"] == "native"
    assert report["fallback_reasons"]
    with pytest.raises(RuntimeError, match="Incomplete"):
        diagnostics.require_complete()
    with pytest.raises(RuntimeError, match="strict"):
        _configure(candidate, strict=True)


def test_incompatible_package_version_is_not_selected(monkeypatch):
    monkeypatch.setattr(kitchen.importlib.metadata, "version", lambda name: "0.0.0")
    module = _linear()
    with pytest.warns(UserWarning, match="0.0.0"):
        diagnostics = _configure(module, strict=False)
    assert diagnostics.report()["selected_backend"] == "native"
    with pytest.raises(RuntimeError, match="0.0.0"):
        _configure(module)


@pytest.mark.parametrize("requires_grad", [False, True])
def test_cpu_runtime_fallback_preserves_native(monkeypatch, requires_grad):
    monkeypatch.setattr(kitchen, "_load_kitchen", lambda: (SimpleNamespace(), "0.2.40"))
    native, candidate = _pair()
    diagnostics = _configure(candidate, strict=False)
    x = torch.randn(2, 256, requires_grad=requires_grad)
    expected = native(x)
    with pytest.warns(UserWarning, match="native"):
        actual = candidate(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if requires_grad:
        gn = torch.autograd.grad(expected.sum(), x)[0]
        gk = torch.autograd.grad(actual.sum(), x)[0]
        torch.testing.assert_close(gk, gn, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="Incomplete"):
        diagnostics.require_complete()
    strict_module = _linear()
    _configure(strict_module)
    with pytest.raises(RuntimeError, match="NVIDIA CUDA"):
        strict_module(x)


def test_unsupported_layer_and_empty_selection(monkeypatch):
    monkeypatch.setattr(kitchen, "_load_kitchen", lambda: (SimpleNamespace(), "0.2.40"))
    # This is valid native ConvRot8 with a group smaller than kitchen's 256.
    module = _linear(64, 64)
    with pytest.warns(UserWarning, match="rotation"):
        diagnostics = _configure(module, strict=False)
    assert diagnostics.report()["ineligible_layers"] == 1
    with pytest.raises(RuntimeError, match="rotation"):
        _configure(module)
    with pytest.raises(RuntimeError, match="no ConvRot8"):
        _configure(torch.nn.Linear(8, 8))


def test_strict_validation_cannot_report_unexecuted_layers(monkeypatch):
    monkeypatch.setattr(kitchen, "_load_kitchen", lambda: (SimpleNamespace(), "0.2.40"))
    diagnostics = _configure(_linear())
    assert diagnostics.report()["observed_training_layers"] == 0
    with pytest.raises(RuntimeError, match="Incomplete"):
        diagnostics.require_complete()


def test_frozen_first_layer_input_still_uses_kitchen(cuda_device):
    module = _linear(device=cuda_device, dtype=torch.bfloat16)
    diagnostics = _configure(module)
    x = torch.randn(1, 33, 256, device=cuda_device, dtype=torch.bfloat16)
    assert not x.requires_grad
    assert torch.isfinite(module(x)).all()
    report = diagnostics.require_complete()
    assert report["observed_training_layer_counts"] == {"comfy_kitchen": 1}


def test_unrelated_quantizers_and_plain_layers_are_unchanged(monkeypatch):
    monkeypatch.setattr(kitchen, "_load_kitchen", lambda: (SimpleNamespace(), "0.2.40"))
    related = _linear()
    unrelated = _linear()
    unrelated.ostris_quantizer.qtype = "convrotint8"
    plain = torch.nn.Linear(8, 8)
    transformer = torch.nn.ModuleList([related, unrelated, plain])
    old_quantizer, old_plain_forward = unrelated.ostris_quantizer, plain.forward
    diagnostics = _configure(transformer)
    assert diagnostics.report()["total_layers"] == 1
    assert unrelated.ostris_quantizer is old_quantizer
    assert plain.forward == old_plain_forward


def test_unknown_backend_and_strict_native_are_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        kitchen.configure_krea_convrot8(_linear(), backend="not-a-backend")
    with pytest.raises(ValueError, match="strict"):
        kitchen.configure_krea_convrot8(_linear(), backend="native", strict=True)


@pytest.mark.parametrize(
    "arch,qtype",
    [
        ("flux", "convrot8"),
        ("krea2", "convrot4"),
        ("qwen_image_2", "convrot8"),
    ],
)
def test_model_configuration_restricts_backend_to_krea_convrot8(arch, qtype):
    from toolkit.config_modules import ModelConfig

    with pytest.raises(ValueError, match="requires arch: krea2 and qtype: convrot8"):
        ModelConfig(
            name_or_path="unused-local-model",
            arch=arch,
            qtype=qtype,
            convrot8_backend="comfy_kitchen",
        )


def test_model_configuration_default_and_valid_opt_in():
    from toolkit.config_modules import ModelConfig

    defaults = ModelConfig(
        name_or_path="unused-local-model", arch="krea2", qtype="convrot8"
    )
    assert defaults.convrot8_backend == "native"
    assert defaults.convrot8_backend_strict is False
    opted_in = ModelConfig(
        name_or_path="unused-local-model",
        arch="krea2",
        qtype="convrot8",
        convrot8_backend="comfy_kitchen",
        convrot8_backend_strict=True,
    )
    assert opted_in.convrot8_backend == "comfy_kitchen"
    assert opted_in.convrot8_backend_strict is True


@pytest.mark.parametrize("seed", [7, 101])
def test_additional_seed_quantization_envelope(cuda_device, seed, record_property):
    native, candidate = _pair(
        2560, 2560, device=cuda_device, dtype=torch.bfloat16, seed=seed
    )
    diagnostics = _configure(candidate)
    x = torch.randn(
        1, 35, 2560, device=cuda_device, dtype=torch.bfloat16, requires_grad=True
    )
    _forward_gate(native, candidate, x, record_property)
    diagnostics.require_complete()


def test_dtype_control_includes_fractional_normalization_without_integer_rounding():
    rotated = torch.tensor([[1.0, 0.3, 0.2, 0.0]], dtype=torch.float32)
    restored = _dtype_normalization_control(rotated, torch.bfloat16)
    codes = restored * 127.0
    assert torch.isfinite(restored).all()
    assert not torch.equal(restored, rotated)
    assert (codes - codes.round()).abs().max() > 0


@pytest.mark.parametrize("dtype", DTYPES)
def test_dtype_control_zero_and_underflow_are_finite(dtype):
    rotated = torch.zeros(2, 4, dtype=torch.float32)
    rotated[1, 0] = torch.finfo(dtype).smallest_normal * torch.finfo(dtype).eps
    restored = _dtype_normalization_control(rotated, dtype)
    assert torch.isfinite(restored).all()
    assert restored[0].count_nonzero() == 0
    assert restored[1, 0] >= 0
    assert restored[1, 1:].count_nonzero() == 0
