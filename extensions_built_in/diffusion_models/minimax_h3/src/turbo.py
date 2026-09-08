"""Sampling-only MiniMax-H3 Turbo LoRA.

A generic second ``LoRASpecialNetwork`` cannot host the official Turbo file on a
pruned base. Matching-shape pairs wrap as a frozen, inactive network. The
2688-wide AdaLN pairs are injected at sample time from the bundled full-model
``silu(t_emb)`` e-grid.

E-grid provenance
    source: larryvrh ComfyUI-MiniMax-H3-Turbo (Apache-2.0)
    path: assets/h3_silu_temb_grid.safetensors
    key: silu_t_emb_grid
    shape: (1025, 2688) bfloat16
    sha256: 30eb3c2cc7fb6b470d9717ff840d359313ac27cd64b705e32da1baa10f72d6a8
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from .transformer import MiniMaxH3AdalnProj

EGRID_SOURCE = "larryvrh/ComfyUI-MiniMax-H3-Turbo (Apache-2.0)"
EGRID_FILENAME = "h3_silu_temb_grid.safetensors"
EGRID_KEY = "silu_t_emb_grid"
EGRID_SHAPE = (1025, 2688)
EGRID_DTYPE = torch.bfloat16
EGRID_SHA256 = "30eb3c2cc7fb6b470d9717ff840d359313ac27cd64b705e32da1baa10f72d6a8"
FULL_MODEL_TEMB_WIDTH = 2688

_DOWN_SUFFIXES = (".lora_down.weight", ".lora_A.weight")
_UP_SUFFIXES = (".lora_up.weight", ".lora_B.weight")

_EGRID_CACHE: Optional[torch.Tensor] = None

AdalnPair = Tuple[MiniMaxH3AdalnProj, torch.Tensor, torch.Tensor]


def egrid_path() -> str:
    return os.path.join(os.path.dirname(__file__), "..", "assets", EGRID_FILENAME)


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_egrid(path: Optional[str] = None, *, verify: bool = True) -> torch.Tensor:
    """Load the full-model silu(t_emb) grid. Cached. Fail-closed on provenance."""
    global _EGRID_CACHE
    if _EGRID_CACHE is not None and path is None:
        return _EGRID_CACHE
    resolved = os.path.normpath(path or egrid_path())
    if not os.path.isfile(resolved):
        raise FileNotFoundError(
            f"H3 Turbo e-grid missing at {resolved}. Source: {EGRID_SOURCE}."
        )
    if verify:
        digest = file_sha256(resolved)
        if digest != EGRID_SHA256:
            raise ValueError(
                f"H3 Turbo e-grid checksum mismatch at {resolved}: "
                f"got {digest}, expected {EGRID_SHA256} ({EGRID_SOURCE})."
            )
    grid = load_file(resolved)[EGRID_KEY]
    if tuple(grid.shape) != EGRID_SHAPE:
        raise ValueError(
            f"H3 Turbo e-grid shape {tuple(grid.shape)} != {EGRID_SHAPE}."
        )
    if grid.dtype != EGRID_DTYPE:
        raise ValueError(
            f"H3 Turbo e-grid dtype {grid.dtype} != {EGRID_DTYPE}."
        )
    if path is None:
        _EGRID_CACHE = grid
    return grid


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
        module_path,
        underscored,
    ]


def _keep_key(module_path: str, suffix: str) -> str:
    """Dotted PEFT key; LoRASpecialNetwork(is_transformer=True) maps . -> $$."""
    return f"transformer.{module_path}.{suffix}"


def partition_turbo_weights(
    dit: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
    strength: float,
) -> Tuple[Dict[str, torch.Tensor], List[AdalnPair], List[str], int, float]:
    """Split a Turbo file into matching LoRA weights and 2688-wide AdaLN pairs."""
    linears = {
        name: module
        for name, module in dit.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    named = dict(dit.named_modules())
    keep: Dict[str, torch.Tensor] = {}
    adaln_pairs: List[AdalnPair] = []
    used_bases = set()
    rank = None
    alpha_value = None

    for path, linear in linears.items():
        for base in _candidate_bases(path):
            down, up, alpha = _lookup_pair(state_dict, base)
            if down is None or up is None:
                continue
            if down.shape[1] == linear.in_features and up.shape[0] == linear.out_features:
                keep[_keep_key(path, "lora_down.weight")] = down
                keep[_keep_key(path, "lora_up.weight")] = up
                resolved_alpha = float(alpha.item()) if torch.is_tensor(alpha) else (
                    float(alpha) if alpha is not None else float(down.shape[0])
                )
                keep[_keep_key(path, "alpha")] = torch.tensor(resolved_alpha)
                rank = int(down.shape[0]) if rank is None else rank
                alpha_value = resolved_alpha if alpha_value is None else alpha_value
                used_bases.add(base)
                break
            parent_path, _, child = path.rpartition(".")
            parent = named.get(parent_path)
            if (
                child == "linear"
                and isinstance(parent, MiniMaxH3AdalnProj)
                and up.shape[0] == parent.linear.out_features
                and down.shape[1] == FULL_MODEL_TEMB_WIDTH
            ):
                adaln_pairs.append(
                    (parent, down.detach().clone(), up.detach().clone() * float(strength))
                )
                rank = int(down.shape[0]) if rank is None else rank
                used_bases.add(base)
                break

    skipped = sorted(_sd_bases(state_dict) - used_bases)
    if not keep:
        raise ValueError(
            "H3 preview LoRA matched no modules on this base — wrong-family "
            "or all-mismatched file."
        )
    return keep, adaln_pairs, skipped, int(rank), float(alpha_value if alpha_value is not None else rank)


def _adaln_forward(base: MiniMaxH3AdalnProj, updates: Sequence[Tuple[torch.Tensor, torch.Tensor]],
                   table: torch.Tensor, egrid: torch.Tensor):
    """Instance forward: original AdaLN linear plus every matching Turbo/context update."""

    def forward(temb: torch.Tensor):
        temb_in = F.silu(temb) if base.apply_silu else temb
        weight = base.linear.weight.to(device=temb.device, dtype=torch.float32)
        bias = base.linear.bias
        if bias is not None:
            bias = bias.to(device=temb.device, dtype=torch.float32)
        x = F.linear(temb_in.float(), weight, bias)
        idx = torch.cdist(
            temb.detach().float(), table.to(device=temb.device, dtype=torch.float32)
        ).argmin(dim=1)
        standin = egrid.to(device=temb.device, dtype=x.dtype)[idx]
        for down, up in updates:
            a = down.to(device=x.device, dtype=x.dtype)
            b = up.to(device=x.device, dtype=x.dtype)
            x = x + (b @ (a @ standin.T)).T
        x = x.view(x.shape[0] * base.modalities, base.expand * base.hidden)
        return x.chunk(base.expand, dim=-1)

    return forward


def patch_adaln(
    dit: torch.nn.Module,
    pairs: Iterable[AdalnPair],
    device,
    dtype,
    egrid: torch.Tensor,
) -> List[MiniMaxH3AdalnProj]:
    """Install combined AdaLN injections. Same module gets every matching update added."""
    pair_list = list(pairs)
    if not pair_list:
        return []
    table = getattr(dit, "adaln_t_table", None)
    if table is None:
        raise ValueError(
            "H3 preview Turbo has AdaLN pairs but this base has no adaln_t_table. "
            "The e-grid injection is only for pruned H3 checkpoints."
        )
    if table.shape[0] != egrid.shape[0]:
        raise ValueError(
            f"H3 Turbo e-grid rows {egrid.shape[0]} != adaln_t_table rows "
            f"{table.shape[0]} — refusing AdaLN injection."
        )
    grid = egrid.to(device=device)
    by_mod = {}
    for mod, down, up in pair_list:
        if down.shape[1] != grid.shape[1]:
            continue
        by_mod.setdefault(id(mod), (mod, []))[1].append(
            (down.to(device=device, dtype=dtype), up.to(device=device, dtype=dtype))
        )
    patched = []
    for mod, updates in by_mod.values():
        mod.forward = _adaln_forward(mod, updates, table, grid)
        patched.append(mod)
    return patched


def unpatch_adaln(modules: Iterable[MiniMaxH3AdalnProj]) -> None:
    for mod in modules:
        if "forward" in mod.__dict__:
            del mod.forward


@dataclass
class TurboLoadReport:
    path: str
    matched: int
    injected_adaln: int
    skipped: List[str]
    rank: int
    alpha: float
    strength: float
    egrid_source: str = EGRID_SOURCE
    egrid_shape: Tuple[int, int] = EGRID_SHAPE
    egrid_sha256: str = EGRID_SHA256

    def summary(self) -> str:
        skip = f" ({len(self.skipped)} skipped)" if self.skipped else ""
        adaln = f" + {self.injected_adaln} adaln via e-grid" if self.injected_adaln else ""
        return (
            f"[h3-turbo] {self.matched} modules at strength {self.strength:g}"
            f"{adaln}{skip}"
        )


@dataclass
class H3PreviewTurbo:
    network: torch.nn.Module
    adaln_pairs: List[AdalnPair]
    report: TurboLoadReport
    strength: float
    egrid: torch.Tensor
    dtype: torch.dtype
    _patched: List[MiniMaxH3AdalnProj] = field(default_factory=list)

    def activate(self, dit: torch.nn.Module, device, dtype, extra_adaln: Optional[List[AdalnPair]] = None):
        self.deactivate()
        self.network.force_to(device, dtype)
        self.network.multiplier = float(self.strength)
        self.network._update_torch_multiplier()
        self.network.is_active = True
        pairs = list(self.adaln_pairs)
        if extra_adaln:
            pairs.extend(extra_adaln)
        self._patched = patch_adaln(dit, pairs, device, dtype, self.egrid)

    def deactivate(self):
        unpatch_adaln(self._patched)
        self._patched = []
        self.network.is_active = False
        self.network.force_to("cpu", self.dtype)
        self.adaln_pairs = [
            (mod, down.cpu(), up.cpu()) for mod, down, up in self.adaln_pairs
        ]

    def park(self):
        self.deactivate()


def _build_network(dit, keep, rank, alpha, strength, target_lin_modules):
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
            "H3 preview LoRA matched no modules on this base — wrong-family "
            "or all-mismatched file."
        )
    return network


def load_preview_turbo(
    dit: torch.nn.Module,
    path: str,
    strength: float = 1.0,
    *,
    target_lin_modules: Optional[List[str]] = None,
    egrid: Optional[torch.Tensor] = None,
) -> H3PreviewTurbo:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"H3 preview LoRA not found: {path}")
    strength = float(strength)
    if not (strength == strength) or abs(strength) == float("inf"):
        raise ValueError(f"preview_lora_strength must be a finite number, got {strength}")

    state_dict = load_file(path)
    keep, adaln_pairs, skipped, rank, alpha = partition_turbo_weights(dit, state_dict, strength)
    targets = target_lin_modules or ["MiniMaxH3Transformer"]
    network = _build_network(dit, keep, rank, alpha, strength, targets)
    dtype = next(dit.parameters()).dtype
    network.force_to("cpu", dtype)
    grid = egrid if egrid is not None else (load_egrid() if adaln_pairs else None)
    if adaln_pairs and grid is None:
        raise ValueError("H3 Turbo AdaLN pairs require the silu(t_emb) e-grid.")
    if grid is None:
        grid = torch.empty(0)
    report = TurboLoadReport(
        path=path,
        matched=len(network.unet_loras),
        injected_adaln=len(adaln_pairs),
        skipped=skipped,
        rank=rank,
        alpha=alpha,
        strength=strength,
    )
    return H3PreviewTurbo(
        network=network,
        adaln_pairs=adaln_pairs,
        report=report,
        strength=strength,
        egrid=grid,
        dtype=dtype,
    )
