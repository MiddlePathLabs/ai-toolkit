"""Sampling-only Turbo LoRA for Qwen-Image 2.1 previews.

Ports the MiniMax-H3 preview-turbo pattern (``minimax_h3/src/turbo.py``) to
this arch: the adapter file is matched against the loaded transformer's
Linear modules by shape, wrapped as a frozen ``LoRASpecialNetwork``, parked
on CPU, and toggled around ``generate_images`` via the
``_enter/_exit_generate_adapters`` hooks. Unlike the pruned H3 base there
are no AdaLN pairs to inject through an e-grid — every entry in a
well-formed file matches a Linear shape directly, and anything that does
not is reported as skipped instead of silently dropped.

The training LoRA keeps running underneath: both networks wrap the same
Linears, the training network stays active, this one contributes only while
a sample event holds it open.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from safetensors.torch import load_file

_DOWN_SUFFIXES = (".lora_down.weight", ".lora_A.weight")
_UP_SUFFIXES = (".lora_up.weight", ".lora_B.weight")

# Viggle turbo sampling recipe (model card, Qwen-Image-2.1-viggle-turbo):
# raw sigmas handed to the pipeline, which applies its resolution-dependent
# shift on top. The low-noise tail is fixed; extra steps subdivide [1.0,
# 0.875] at the high-noise end only. 5/6/7-step grids are verbatim from the
# card, larger counts follow the same rule.
_TURBO_SIGMA_TAIL = (0.875, 0.75, 0.5, 0.25)
_TURBO_SIGMA_GRID_6 = (1.0, 0.9375) + _TURBO_SIGMA_TAIL
_TURBO_SIGMA_GRID_7 = (1.0, 0.95833333, 0.91666667) + _TURBO_SIGMA_TAIL


def turbo_sigma_grid(steps: int) -> List[float]:
    if steps < 5:
        raise ValueError(
            f"turbo previews need at least 5 sample steps, got {steps}"
        )
    if steps == 5:
        return [1.0] + list(_TURBO_SIGMA_TAIL)
    if steps == 6:
        return list(_TURBO_SIGMA_GRID_6)
    if steps == 7:
        return list(_TURBO_SIGMA_GRID_7)
    extra = steps - 5
    head = [1.0 - (1.0 - _TURBO_SIGMA_TAIL[0]) * i / (extra + 1) for i in range(1, extra + 1)]
    return [1.0] + head + list(_TURBO_SIGMA_TAIL)


def resolve_preview_lora_path(path: str, models_path: str) -> str:
    """Local file, filename under models/loras, or user/repo/file.safetensors."""
    if os.path.isfile(path):
        return path
    from toolkit.models.v2.resolver import find_file_recursive

    filename = os.path.basename(path)
    found = find_file_recursive(os.path.join(models_path, "loras"), filename)
    if found is not None:
        return found
    parts = path.split("/")
    if len(parts) != 3:
        raise ValueError(
            f"Preview LoRA path {path} is not a local path, a file under "
            f"{os.path.join(models_path, 'loras')}, or a "
            "'user/repo/file.safetensors' hub path."
        )
    import huggingface_hub

    target_dir = os.path.join(models_path, "loras", "preview_adapters")
    os.makedirs(target_dir, exist_ok=True)
    try:
        return huggingface_hub.hf_hub_download(
            repo_id="/".join(parts[:2]),
            filename=parts[2],
            local_dir=target_dir,
        )
    except Exception as exc:
        raise ValueError(f"Failed to download preview LoRA from {path}: {exc}") from exc


def _sd_bases(state_dict: Dict[str, torch.Tensor]) -> set:
    bases = set()
    suffixes = _DOWN_SUFFIXES + _UP_SUFFIXES + (".alpha",)
    for key in state_dict:
        for suffix in suffixes:
            if key.endswith(suffix):
                bases.add(key[: -len(suffix)])
                break
    return bases


def _lookup_pair(state_dict: Dict[str, torch.Tensor], base: str):
    down = next((state_dict[base + s] for s in _DOWN_SUFFIXES if base + s in state_dict), None)
    up = next((state_dict[base + s] for s in _UP_SUFFIXES if base + s in state_dict), None)
    alpha = state_dict.get(f"{base}.alpha")
    return down, up, alpha


def _candidate_bases(module_path: str) -> List[str]:
    underscored = module_path.replace(".", "_")
    return [
        f"lora_unet_{underscored}",
        f"lora_transformer_{underscored}",
        f"transformer.{module_path}",
        f"diffusion_model.{module_path}",
        f"base_model.model.{module_path}",
        module_path,
        underscored,
    ]


def _keep_key(module_path: str, suffix: str) -> str:
    """Dotted PEFT key; LoRASpecialNetwork(is_transformer=True) maps . -> $$."""
    return f"transformer.{module_path}.{suffix}"


def _resolve_alpha(alpha, rank: int) -> float:
    return float(alpha.item()) if torch.is_tensor(alpha) else (
        float(alpha) if alpha is not None else float(rank)
    )


def _fused_split_gate(state_dict, module_path: str, linear):
    """Diffusers split layout (``gate_layer`` + ``proj``) -> fused ``gate_up``.

    The official diffusers checkpoints (and adapter files trained against
    them) split the SwiGLU gate projection in two Linears; this class keeps
    the ComfyUI fused one (see ``convert_state_dict_on_load``). For full
    weights the conversion is a row concat; for a LoRA pair with independent
    down-matrices the exact equivalent is rank-2r: concatenated down,
    block-diagonal up (zero off-blocks), halved-precision-free. Gate half
    first, matching the loader's ``[gate; up]`` concat order.

    Returns ``(fused_down, fused_up, fused_alpha, used_bases)`` or None.
    """
    parent = module_path[: -len(".gate_up")]
    for root in _candidate_bases(parent):
        gate_base, proj_base = f"{root}.gate_layer", f"{root}.proj"
        g_down, g_up, g_alpha = _lookup_pair(state_dict, gate_base)
        p_down, p_up, p_alpha = _lookup_pair(state_dict, proj_base)
        if g_down is None or g_up is None or p_down is None or p_up is None:
            continue
        if g_down.shape[1] != linear.in_features or p_down.shape[1] != linear.in_features:
            continue
        if g_up.shape[0] + p_up.shape[0] != linear.out_features:
            continue
        g_rank, p_rank = int(g_down.shape[0]), int(p_down.shape[0])
        g_scale = _resolve_alpha(g_alpha, g_rank) / g_rank
        p_scale = _resolve_alpha(p_alpha, p_rank) / p_rank
        if abs(g_scale - p_scale) > 1e-9:
            # halves need different lora scales; a single module cannot carry that
            continue
        fused_down = torch.cat([g_down, p_down], dim=0)
        fused_up = torch.cat([
            torch.cat([g_up, torch.zeros(p_up.shape[0], g_rank, dtype=g_up.dtype)], dim=1),
            torch.cat([torch.zeros(g_up.shape[0], p_rank, dtype=g_up.dtype), p_up], dim=1),
        ], dim=0)
        fused_alpha = (g_rank + p_rank) * g_scale
        return fused_down, fused_up, fused_alpha, [gate_base, proj_base]
    return None


def partition_lora_weights(
    dit: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], List[str], int]:
    """Match the adapter file onto the transformer's Linear modules by shape.

    Returns the converted weight dict, the file's unmatched base keys, and
    the number of split-gate pairs fused into ``gate_up`` modules.
    """
    linears = {
        name: module
        for name, module in dit.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    keep: Dict[str, torch.Tensor] = {}
    used_bases = set()
    fused = 0

    for path, linear in linears.items():
        matched = False
        for base in _candidate_bases(path):
            down, up, alpha = _lookup_pair(state_dict, base)
            if down is None or up is None:
                continue
            if down.shape[1] == linear.in_features and up.shape[0] == linear.out_features:
                keep[_keep_key(path, "lora_down.weight")] = down
                keep[_keep_key(path, "lora_up.weight")] = up
                keep[_keep_key(path, "alpha")] = torch.tensor(_resolve_alpha(alpha, down.shape[0]))
                used_bases.add(base)
                matched = True
                break
        if not matched and path.endswith(".gate_up"):
            result = _fused_split_gate(state_dict, path, linear)
            if result is not None:
                down, up, alpha, bases = result
                keep[_keep_key(path, "lora_down.weight")] = down
                keep[_keep_key(path, "lora_up.weight")] = up
                keep[_keep_key(path, "alpha")] = torch.tensor(alpha)
                used_bases.update(bases)
                fused += 1

    skipped = sorted(_sd_bases(state_dict) - used_bases)
    if not keep:
        raise ValueError(
            "Qwen preview LoRA matched no modules on this base — wrong-family "
            "or all-mismatched file."
        )
    return keep, skipped, fused


def _build_network(dit, keep, strength, target_lin_modules):
    from toolkit.config_modules import NetworkConfig
    from toolkit.lora_special import LoRASpecialNetwork

    modules_dim = {}
    modules_alpha = {}
    for key, value in keep.items():
        if key.endswith(".lora_down.weight"):
            name = key[: -len(".lora_down.weight")].replace(".", "$$")
            modules_dim[name] = int(value.shape[0])
        elif key.endswith(".alpha"):
            name = key[: -len(".alpha")].replace(".", "$$")
            modules_alpha[name] = float(value.item()) if torch.is_tensor(value) else float(value)
    for name in modules_dim:
        modules_alpha.setdefault(name, float(modules_dim[name]))

    # a mixed-rank file is fine: each module's dim comes from its own tensors,
    # these are just the fallbacks for the network-level defaults
    rank = max(modules_dim.values())
    alpha = next(iter(modules_alpha.values()), float(rank))

    network_config = NetworkConfig(
        type="lora",
        linear=rank,
        linear_alpha=alpha,
        transformer_only=False,
    )
    network = LoRASpecialNetwork(
        text_encoder=None,
        unet=dit,
        lora_dim=rank,
        multiplier=float(strength),
        alpha=float(alpha),
        train_unet=True,
        train_text_encoder=False,
        network_config=network_config,
        network_type="lora",
        transformer_only=False,
        is_transformer=True,
        target_lin_modules=target_lin_modules,
        is_assistant_adapter=True,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
    )
    network.apply_to(None, dit, apply_text_encoder=False, apply_unet=True)
    network.load_weights(keep)
    network.requires_grad_(False)
    network.eval()
    network.is_active = False
    for param in network.parameters():
        param.requires_grad_(False)
    if not network.unet_loras:
        raise ValueError(
            "Qwen preview LoRA matched no modules on this base — wrong-family "
            "or all-mismatched file."
        )
    return network


@dataclass
class PreviewLoadReport:
    path: str
    matched: int
    skipped: List[str]
    strength: float
    fused_split_gates: int = 0

    def summary(self) -> str:
        skip = f" ({len(self.skipped)} skipped)" if self.skipped else ""
        fused = f", {self.fused_split_gates} split-gates fused" if self.fused_split_gates else ""
        return (
            f"[qwen-preview-lora] {self.matched} modules at strength "
            f"{self.strength:g}{fused}{skip}"
        )


@dataclass
class QwenPreviewTurbo:
    network: torch.nn.Module
    report: PreviewLoadReport
    strength: float
    dtype: torch.dtype
    _device_was: Optional[torch.device] = field(default=None)

    def activate(self, dit: torch.nn.Module, device, dtype):
        self.network.force_to(device, dtype)
        self.network.multiplier = float(self.strength)
        self.network._update_torch_multiplier()
        self.network.is_active = True

    def deactivate(self):
        self.network.is_active = False
        self.network.force_to("cpu", self.dtype)

    def park(self):
        self.deactivate()


def load_preview_lora(
    dit: torch.nn.Module,
    path: str,
    strength: float = 1.0,
    *,
    target_lin_modules: Optional[List[str]] = None,
) -> QwenPreviewTurbo:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Qwen preview LoRA not found: {path}")
    strength = float(strength)
    if not (strength == strength) or abs(strength) == float("inf"):
        raise ValueError(f"preview_lora_strength must be a finite number, got {strength}")

    state_dict = load_file(path)
    keep, skipped, fused = partition_lora_weights(dit, state_dict)
    targets = target_lin_modules or ["QwenImage21Transformer2DModel"]
    network = _build_network(dit, keep, strength, targets)
    dtype = next(dit.parameters()).dtype
    network.force_to("cpu", dtype)
    report = PreviewLoadReport(
        path=path,
        matched=len(network.unet_loras),
        skipped=skipped,
        strength=strength,
        fused_split_gates=fused,
    )
    return QwenPreviewTurbo(
        network=network,
        report=report,
        strength=strength,
        dtype=dtype,
    )
