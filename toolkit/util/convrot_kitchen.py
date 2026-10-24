"""Opt-in Krea 2 ConvRot8 training kernels; storage and inference stay native."""

import importlib.metadata
import warnings
import weakref
from collections import Counter

import torch

from toolkit.util.convrot_quant import (
    ConvRotInt8Quantizer,
    _int8_input_gradient,
    _int8_linear_ste_op,
    regular_hadamard,
    rotate,
)

_REPORTS = weakref.WeakValueDictionary()
_INSTALL = "Install in AI Toolkit's Python: python -m pip install --no-deps --only-binary=:all: comfy-kitchen==0.2.40"


def _load_kitchen():
    version = importlib.metadata.version("comfy-kitchen")
    if version != "0.2.40":
        raise ImportError(
            f"Validated comfy-kitchen version is 0.2.40, found {version}. {_INSTALL}"
        )
    import comfy_kitchen
    from comfy_kitchen.backends import cuda

    if not comfy_kitchen.registry.is_available("cuda"):
        raise ImportError(f"comfy-kitchen CUDA backend is unavailable. {_INSTALL}")
    if not all(
        callable(getattr(cuda, name, None))
        for name in ("int8_linear", "_convrot_fused_shared_memory_fits")
    ):
        raise ImportError(
            f"comfy-kitchen CUDA ConvRot8 API is incompatible. {_INSTALL}"
        )
    return cuda, version


def _layer_reason(module):
    if module.cr8_rot_size != 256:
        return "rotation group must be 256"
    if not 256 <= module.in_features <= 16384 or module.in_features % 256:
        return "input width must be a multiple of 256 in [256, 16384]"
    if module.out_features % 8:
        return "output width must be divisible by 8"
    if not module.cr8_qdata.is_contiguous():
        return "weight codes must be contiguous"
    return None


def _call_reason(x, qdata, scales, bias, cuda):
    if not x.is_cuda or torch.version.hip:
        return "requires NVIDIA CUDA"
    if x.dtype not in (torch.bfloat16, torch.float16):
        return "activation dtype must be BF16 or FP16"
    if torch.cuda.get_device_capability(x.device) < (8, 0):
        return "requires sm80 or newer"
    if x.numel() == 0:
        return "empty activation"
    if qdata.device != x.device or scales.device != x.device:
        return "quantized buffers and input must share a CUDA device"
    if qdata.dtype != torch.int8 or scales.dtype != torch.uint8:
        return "incompatible quantized buffer dtypes"
    if qdata.data_ptr() % 16 or scales.data_ptr() % 16:
        return "quantized buffers must be 16-byte aligned"
    if bias is not None and bias.device != x.device:
        return "bias and input must share a CUDA device"
    # Version-pinned predicate mirrors the installed CUDA entry point; no launch
    # is attempted if it would select the unfused rotation path.
    if not cuda._convrot_fused_shared_memory_fits(x, x.shape[-1], 256):
        return "fused rotation exceeds device shared memory"
    return None


class KitchenDiagnostics:
    def __init__(self, requested, strict, compiled):
        self.requested = requested
        self.strict = strict
        self.compiled = compiled
        self.version = None
        self.cuda = None
        self.layers = {}
        self.observed = {}
        self.fallbacks = {}
        self.rotation_modes = {}
        self._warned = set()
        _REPORTS[id(self)] = self

    def fallback(self, name, reason):
        if self.strict:
            raise RuntimeError(
                f"ConvRot8 strict comfy_kitchen coverage failed for {name}: {reason}. {_INSTALL}"
            )
        self.fallbacks[name] = reason
        key = (name, reason)
        if key not in self._warned:
            self._warned.add(key)
            warnings.warn(f"ConvRot8 {name}: using native: {reason}", stacklevel=3)

    def report(self):
        eligible = sum(
            reason is None and name not in self.fallbacks
            for name, reason in self.layers.items()
        )
        observed = Counter(backend for backend, _ in self.observed.values())
        reasons = Counter(
            {
                name: reason
                for name, reason in (self.layers | self.fallbacks).items()
                if reason
            }.values()
        )
        kitchen_layers = sum(
            reason is None and name not in self.fallbacks and self.cuda is not None
            for name, reason in self.layers.items()
        )
        selected = (
            "mixed"
            if 0 < kitchen_layers < len(self.layers)
            else "comfy_kitchen"
            if kitchen_layers
            else "native"
        )
        return {
            "requested_backend": self.requested,
            "selected_backend": selected,
            "comfy_kitchen_version": self.version,
            "total_layers": len(self.layers),
            "eligible_layers": eligible,
            "ineligible_layers": len(self.layers) - eligible,
            "selected_layer_counts": {
                "comfy_kitchen": kitchen_layers,
                "native": len(self.layers) - kitchen_layers,
            },
            "observed_training_layer_counts": dict(observed),
            "observed_training_layers": len(self.observed),
            "fallback_reasons": dict(reasons),
            "fused_rotation": any(
                mode == "fused_fp32" for mode in self.rotation_modes.values()
            ),
            "observed_rotation_layer_counts": dict(
                Counter(self.rotation_modes.values())
            ),
            "rotation_policy": "BF16: fused FP32; FP16: native rotation then FP32 quantization",
            "activation_preprocessing_dtype": "float32",
            "gemm_scaling_dispatch": "internal CUTLASS/cuBLAS selection; confirm with profiler",
            "execution": "compiled requested; confirm graph coverage separately"
            if self.compiled
            else "eager",
            "strict": self.strict,
        }

    def require_complete(self):
        report = self.report()
        if (
            not self.layers
            or report["ineligible_layers"]
            or self.cuda is None
            or len(self.observed) != len(self.layers)
            or any(backend != "comfy_kitchen" for backend, _ in self.observed.values())
        ):
            raise RuntimeError(f"Incomplete experimental training coverage: {report}")
        return report


@torch.library.custom_op("ostris::convrot_kitchen_linear_ste", mutates_args=())
def _kitchen_linear_ste_op(
    x: torch.Tensor,
    qdata: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None,
    rot: int,
    report_id: int,
    name: str,
    hadamard: torch.Tensor,
) -> torch.Tensor:
    diagnostics = _REPORTS[report_id]
    reason = diagnostics.layers[name]
    if reason is None:
        reason = _call_reason(x, qdata, scales, bias, diagnostics.cuda)
    if reason is not None:
        diagnostics.fallback(name, reason)
        diagnostics.observed[name] = ("native", reason)
        return _int8_linear_ste_op(
            rotate(x, rot), qdata, scales, bias, 127, str(x.dtype).split(".")[-1]
        )
    # Kitchen rounds normalization through the input dtype. FP32 preprocessing
    # avoids the extra BF16/FP16 scale/division error while preserving output dtype.
    # FP16 keeps native rotation rounding; fused FP32 rotation failed its
    # accumulated-adapter numerical gate. BF16 passes with fused FP32 rotation.
    fused_rotation = x.dtype == torch.bfloat16
    kernel_input = x.float() if fused_rotation else rotate(x, rot).float()
    # No exception recovery after CUDA execution: device faults must propagate.
    out = diagnostics.cuda.int8_linear(
        kernel_input,
        qdata,
        scales.view(torch.float32),
        bias=bias,
        out_dtype=x.dtype,
        convrot=fused_rotation,
        convrot_groupsize=rot,
    )
    if name not in diagnostics.observed:
        diagnostics.observed[name] = ("comfy_kitchen", None)
    if name not in diagnostics.rotation_modes:
        diagnostics.rotation_modes[name] = (
            "fused_fp32" if fused_rotation else "native_rotation_fp32_quant"
        )
    return out


@_kitchen_linear_ste_op.register_fake
def _kitchen_linear_ste_fake(x, qdata, scales, bias, rot, report_id, name, hadamard):
    return x.new_empty((x.shape[0], qdata.shape[0]))


def _kitchen_setup(ctx, inputs, output):
    _, qdata, scales, _, _, _, _, hadamard = inputs
    ctx.save_for_backward(qdata, scales, hadamard)


@torch.library.custom_op("ostris::convrot_kitchen_input_gradient", mutates_args=())
def _kitchen_input_gradient(
    grad: torch.Tensor,
    qdata: torch.Tensor,
    scales: torch.Tensor,
    hadamard: torch.Tensor,
) -> torch.Tensor:
    # Keep low-precision scale/multiply/materialization boundaries identical in
    # eager and compiled STE backward; Inductor must not fold them into GEMM.
    grad_rot = _int8_input_gradient(grad, qdata, scales)
    shape = grad_rot.shape
    rot = hadamard.shape[0]
    return (grad_rot.reshape(-1, shape[-1] // rot, rot) @ hadamard).reshape(shape)


@_kitchen_input_gradient.register_fake
def _kitchen_input_gradient_fake(grad, qdata, scales, hadamard):
    return grad.new_empty((grad.shape[0], qdata.shape[1]))


def _kitchen_backward(ctx, grad):
    qdata, scales, hadamard = ctx.saved_tensors
    return (
        _kitchen_input_gradient(grad, qdata, scales, hadamard),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


_kitchen_linear_ste_op.register_autograd(
    _kitchen_backward, setup_context=_kitchen_setup
)


class ConvRotKitchenQuantizer(ConvRotInt8Quantizer):
    def __init__(self, native, diagnostics, name):
        super().__init__(native.rot_size)
        self.qtype = native.qtype
        self.diagnostics = diagnostics
        self.name = name

    def forward(self, module, x):
        if not module.training or not torch.is_grad_enabled():
            return super().forward(module, x)
        reason = self.diagnostics.layers[self.name]
        if not x.is_cuda or torch.version.hip:
            reason = "requires NVIDIA CUDA"
        elif x.dtype not in (torch.bfloat16, torch.float16):
            reason = "activation dtype must be BF16 or FP16"
        elif torch.cuda.get_device_capability(x.device) < (8, 0):
            reason = "requires sm80 or newer"
        if reason:
            self.diagnostics.fallback(self.name, reason)
            self.diagnostics.observed[self.name] = ("native", reason)
            return super().forward(module, x)
        out = _kitchen_linear_ste_op(
            x.reshape(-1, module.in_features).contiguous(),
            module.cr8_qdata,
            module.cr8_scales,
            module.bias,
            module.cr8_rot_size,
            id(self.diagnostics),
            self.name,
            regular_hadamard(module.cr8_rot_size, x.device, x.dtype),
        )
        return out.reshape(*x.shape[:-1], module.out_features)


def configure_krea_convrot8(
    transformer, backend="native", strict=False, compiled=False
):
    if backend not in ("native", "comfy_kitchen"):
        raise ValueError(f"Unknown ConvRot8 backend: {backend}")
    if strict and backend != "comfy_kitchen":
        raise ValueError("convrot8_backend_strict requires comfy_kitchen")
    diagnostics = KitchenDiagnostics(backend, strict, compiled)
    layers = [
        (name, module)
        for name, module in transformer.named_modules()
        if type(getattr(module, "ostris_quantizer", None))
        in (ConvRotInt8Quantizer, ConvRotKitchenQuantizer)
        and module.ostris_quantizer.qtype == "convrot8"
    ]
    if backend == "native":
        for name, module in layers:
            if isinstance(module.ostris_quantizer, ConvRotKitchenQuantizer):
                native = ConvRotInt8Quantizer(module.ostris_quantizer.rot_size)
                native.qtype = "convrot8"
                module.ostris_quantizer = native
            diagnostics.layers[name] = _layer_reason(module)
        return diagnostics
    try:
        diagnostics.cuda, diagnostics.version = _load_kitchen()
    except (ImportError, OSError) as exc:
        reason = f"{exc}. {_INSTALL}"
        for name, module in layers:
            diagnostics.layers[name] = reason
            if isinstance(module.ostris_quantizer, ConvRotKitchenQuantizer):
                native = ConvRotInt8Quantizer(module.ostris_quantizer.rot_size)
                native.qtype = "convrot8"
                module.ostris_quantizer = native
        diagnostics.fallback("dependency", reason)
        return diagnostics
    for name, module in layers:
        reason = _layer_reason(module)
        diagnostics.layers[name] = reason
        if reason:
            diagnostics.fallback(name, reason)
        module.ostris_quantizer = ConvRotKitchenQuantizer(
            module.ostris_quantizer, diagnostics, name
        )
    if strict and not layers:
        raise RuntimeError(
            "Strict comfy_kitchen requested but no ConvRot8 layers were found"
        )
    return diagnostics
