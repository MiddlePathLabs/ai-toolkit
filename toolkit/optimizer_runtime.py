"""Transient optimizer runtime windows for step-scale (and later, active-param masks).

Owns no optimizer state. Window context is never written into ``state_dict``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Optional

import torch

from toolkit.accelerator import unwrap_model
UpdatePhase = Literal["step", "backward"]
StepScaleStrategy = Literal["group_lr", "native", "unsupported"]

_NATIVE_ATTR = "_runtime_step_scale"


def unwrap_optimizer(optimizer: Any) -> Any:
    """Reach the real torch optimizer through Accelerate / compiled wrappers."""
    inner = optimizer
    seen: set[int] = set()
    while inner is not None:
        ident = id(inner)
        if ident in seen:
            break
        seen.add(ident)
        nxt = getattr(inner, "optimizer", None)
        if nxt is None or nxt is inner:
            break
        inner = nxt
    try:
        inner = unwrap_model(inner)
    except Exception:
        pass
    return inner


def uses_adaptive_lr_step_scale(train_config: Any) -> bool:
    """True when watcher multipliers must be applied as an optimizer step scale."""
    if train_config is None:
        return False
    if getattr(train_config, "per_image_adaptive_lr_stats_only", False):
        return False
    if not getattr(train_config, "per_image_adaptive_lr", False):
        return False
    mode = getattr(train_config, "per_image_adaptive_lr_mode", "loss") or "loss"
    return str(mode).lower() == "lr"


def mean_window_scale(members: dict[str, float] | Iterable[float]) -> float:
    """Arithmetic mean of contributing file-item multipliers, each item once.

    A dict is keyed by item identity (path): a repeated image in the window
    contributes several gradient samples but one mean entry (last multiplier
    wins). Regularisation items that appear in the window are contributing
    members — they dilute the mean if their multiplier is 1.0. That is the
    Workstream 0 contract, not an accident. Empty → 1.0.
    """
    if isinstance(members, dict):
        values = list(members.values())
    else:
        values = list(members)
    if not values:
        return 1.0
    return float(sum(values) / len(values))


def _snapshot_lr(lr: Any) -> Any:
    if torch.is_tensor(lr):
        return lr.detach().clone()
    return lr


def _scaled_lr(lr: Any, scale: float) -> Any:
    if torch.is_tensor(lr):
        return lr * scale
    return lr * scale


def _group_flag(optimizer: Any, key: str, default: Any = None) -> Any:
    groups = getattr(optimizer, "param_groups", None)
    if not groups:
        return default
    return groups[0].get(key, default)


def _class_name(optimizer: Any) -> str:
    return type(optimizer).__name__


def _module_name(optimizer: Any) -> str:
    return type(optimizer).__module__ or ""


def _is_fused_backward(optimizer: Any) -> bool:
    name = _class_name(optimizer)
    if name == "Automagic2":
        return True
    fused = getattr(optimizer, "fused", None)
    if fused is None:
        return False
    return bool(fused)


@dataclass
class OptimizerCapabilities:
    supports_step_scale: bool
    supports_active_param_mask: bool
    update_phase: UpdatePhase
    step_scale_strategy: StepScaleStrategy
    unsupported_reason: Optional[str]
    optimizer_label: str


def classify_optimizer(optimizer: Any) -> OptimizerCapabilities:
    """Capability table for one constructed optimizer instance.

    Classification is by concrete class and live options, not by the factory
    name string. Unknown / external classes are unsupported until declared.
    """
    opt = unwrap_optimizer(optimizer)
    name = _class_name(opt)
    module = _module_name(opt)
    label = f"{module}.{name}" if module else name
    no_mask = False

    def group_lr(phase: UpdatePhase = "step") -> OptimizerCapabilities:
        return OptimizerCapabilities(
            supports_step_scale=True,
            supports_active_param_mask=no_mask,
            update_phase=phase,
            step_scale_strategy="group_lr",
            unsupported_reason=None,
            optimizer_label=label,
        )

    def native(phase: UpdatePhase) -> OptimizerCapabilities:
        return OptimizerCapabilities(
            supports_step_scale=True,
            supports_active_param_mask=no_mask,
            update_phase=phase,
            step_scale_strategy="native",
            unsupported_reason=None,
            optimizer_label=label,
        )

    def unsupported(reason: str, phase: UpdatePhase = "step") -> OptimizerCapabilities:
        return OptimizerCapabilities(
            supports_step_scale=False,
            supports_active_param_mask=no_mask,
            update_phase=phase,
            step_scale_strategy="unsupported",
            unsupported_reason=reason,
            optimizer_label=label,
        )

    if name == "Rose":
        return group_lr("step")

    if name in ("Adam", "AdamW", "Adagrad") and "torch.optim" in module:
        # torch fused=True is a step-time CUDA kernel, not fused-backward.
        return group_lr("step")

    if name == "Adam8bit" and "toolkit.optimizers.adam8bit" in module:
        return group_lr("step")

    if name in ("Adam8bit", "AdamW8bit", "AdEMAMix8bit", "Lion8bit"):
        return group_lr("step")

    if name == "Adafactor":
        if _group_flag(opt, "relative_step", False):
            return unsupported(
                "Adafactor relative_step=True ignores group lr (_get_lr); "
                "a group-LR scale would be a silent no-op. Set relative_step: false "
                "or disable per_image_adaptive_lr_mode: lr."
            )
        if _group_flag(opt, "scale_parameter", False):
            return unsupported(
                "Adafactor scale_parameter=True makes its per-parameter LR a function "
                "of post-update weights (state['RMS']); a one-window step scale would "
                "permanently perturb it. Set scale_parameter: false or use "
                "per_image_adaptive_lr_mode: loss."
            )
        if _group_flag(opt, "beta1", None) is not None:
            return native("step")
        return group_lr("step")

    if name == "Automagic":
        return native("step")
    if name == "Automagic2":
        return native("backward")
    if name == "Automagic3":
        return native("backward" if _is_fused_backward(opt) else "step")
    if name == "AutomagicEXPERIMENT":
        return native("backward" if _is_fused_backward(opt) else "step")
    if name == "AdamConvRot":
        return native("backward" if _is_fused_backward(opt) else "step")

    if name == "Prodigy8bit":
        return native("step")

    if name == "Prodigy":
        return unsupported(
            "prodigyopt.Prodigy feeds group lr into the d-hat recursion "
            "(numerator and s memory); a temporary group-LR mutation corrupts d. "
            "A copied-step subclass is not a small reviewable patch. Use "
            "per_image_adaptive_lr_mode: loss, disable the feature, or switch to "
            "prodigy8bit (native final-update scale)."
        )

    if name.startswith("DAdapt"):
        return unsupported(
            "D-Adaptation uses group lr inside its adaptation equations; "
            "a temporary group-LR mutation is not a faithful final-update scale. "
            "Use per_image_adaptive_lr_mode: loss or disable the feature."
        )

    if name == "Lion" and "lion_pytorch" in module:
        return unsupported(
            f"{label} is not classified for step scaling in this fork "
            "(lion_pytorch is not a maintained factory target). "
            "Use per_image_adaptive_lr_mode: loss or disable the feature."
        )

    return unsupported(
        f"{label} has not declared supports_step_scale. "
        "Declare a faithful group-LR or native strategy before using "
        "per_image_adaptive_lr_mode: lr."
    )


class OptimizerRuntimeAdapter:
    """Transient step-scale / active-param window around one optimizer update.

    ``begin_window`` / ``end_window`` must be paired; ``end_window`` is safe
    to call extra times and must run from ``finally``.
    """

    def __init__(self, caps: OptimizerCapabilities):
        self.supports_step_scale = caps.supports_step_scale
        self.supports_active_param_mask = caps.supports_active_param_mask
        self.update_phase: UpdatePhase = caps.update_phase
        self.step_scale_strategy: StepScaleStrategy = caps.step_scale_strategy
        self.unsupported_reason = caps.unsupported_reason
        self.optimizer_label = caps.optimizer_label
        self.last_step_scale = 1.0
        self._snapshot_lrs: Optional[list[tuple[dict, Any]]] = None
        self._window_open = False
        self._native_opt: Any = None

    @classmethod
    def inspect(cls, optimizer: Any, optimizer_type: Optional[str] = None) -> "OptimizerRuntimeAdapter":
        caps = classify_optimizer(optimizer)
        if optimizer_type:
            caps.optimizer_label = f"{optimizer_type} ({caps.optimizer_label})"
        return cls(caps)

    def validate_for_train_config(self, train_config: Any) -> None:
        if not uses_adaptive_lr_step_scale(train_config):
            return
        if not self.supports_step_scale:
            reason = self.unsupported_reason or "supports_step_scale is false"
            raise ValueError(
                f"per_image_adaptive_lr_mode='lr' requires optimizer step scaling, but "
                f"{self.optimizer_label} does not support it: {reason}"
            )
        if self.update_phase == "backward":
            self._reject_fused_incompatibilities(train_config)

    def _reject_fused_incompatibilities(self, train_config: Any) -> None:
        grad_accum = int(getattr(train_config, "gradient_accumulation", 1) or 1)
        grad_accum_steps = getattr(train_config, "gradient_accumulation_steps", 1)
        try:
            grad_accum_steps = int(grad_accum_steps)
        except (TypeError, ValueError):
            grad_accum_steps = 1
        single_item = bool(getattr(train_config, "single_item_batching", False))
        loss_type = getattr(train_config, "loss_type", "mse")

        problems = []
        if grad_accum > 1:
            problems.append(f"gradient_accumulation={grad_accum}")
        if grad_accum_steps != 1:
            problems.append(f"gradient_accumulation_steps={grad_accum_steps}")
        if single_item:
            problems.append("single_item_batching=True")
        if loss_type == "mean_flow":
            problems.append("loss_type='mean_flow' (second backward bypasses the fused window)")
        if not problems:
            return
        raise ValueError(
            f"per_image_adaptive_lr_mode='lr' with fused-backward optimizer "
            f"{self.optimizer_label} cannot preserve one-update-per-backward semantics "
            f"with {', '.join(problems)}. Set those options to a single backward per "
            f"update, switch to per_image_adaptive_lr_mode: loss, or disable the feature."
        )

    def begin_window(
        self,
        optimizer: Any,
        step_scale: float = 1.0,
        active_params: Optional[Iterable[Any]] = None,
    ) -> None:
        if self._window_open:
            self.end_window(optimizer)
        scale = float(step_scale)
        if not math.isfinite(scale) or scale < 0.0:
            raise ValueError(
                f"step_scale must be a finite number >= 0, got {step_scale!r}"
            )
        if active_params is not None and not self.supports_active_param_mask:
            raise RuntimeError(
                f"{self.optimizer_label} does not support an active-parameter mask "
                f"(supports_active_param_mask is false)."
            )
        self._window_open = True
        self.last_step_scale = scale
        opt = unwrap_optimizer(optimizer)
        if scale == 1.0:
            return
        if not self.supports_step_scale:
            reason = self.unsupported_reason or "supports_step_scale is false"
            raise RuntimeError(
                f"Cannot apply step_scale={scale} on {self.optimizer_label}: {reason}"
            )
        if self.step_scale_strategy == "group_lr":
            snapshot: list[tuple[dict, Any]] = []
            for group in opt.param_groups:
                # Stamp before the first scaled window so Rose wd_schedule does
                # not capture a scaled lr via setdefault("initial_lr", lr).
                if "initial_lr" not in group:
                    group["initial_lr"] = _snapshot_lr(group["lr"])
                lr = group["lr"]
                snapshot.append((group, _snapshot_lr(lr)))
                group["lr"] = _scaled_lr(lr, scale)
            self._snapshot_lrs = snapshot
            return
        if self.step_scale_strategy == "native":
            setattr(opt, _NATIVE_ATTR, scale)
            self._native_opt = opt
            return
        raise RuntimeError(
            f"{self.optimizer_label} has no implementable step-scale strategy"
        )

    def end_window(self, optimizer: Any = None) -> None:
        snapshot = self._snapshot_lrs
        self._snapshot_lrs = None
        if snapshot is not None:
            for group, lr in snapshot:
                group["lr"] = lr
        native_opt = self._native_opt
        self._native_opt = None
        if native_opt is None and optimizer is not None:
            native_opt = unwrap_optimizer(optimizer)
        if native_opt is not None and hasattr(native_opt, _NATIVE_ATTR):
            setattr(native_opt, _NATIVE_ATTR, 1.0)
        self._window_open = False
