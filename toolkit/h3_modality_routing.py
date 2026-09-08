"""H3 trunk-block ownership and per-window active-parameter sets.

Do not parse serialized or runtime ``lora_name`` (PEFT converts dots to ``$$``;
token refiners also contain a nested ``blocks`` segment). Ownership is
``orig_module_ref()`` against ``transformer.named_modules()``, and only paths
matching ``^blocks.<idx>.`` are trunk-owned. Refiners and every non-trunk
module stay active.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Optional

import torch

from toolkit.accelerator import unwrap_model
from toolkit.print import print_acc

H3_ARCH_PREFIX = "minimax_h3"
_TRUNK_BLOCK_RE = re.compile(r"^blocks\.(\d+)\.")
_RANGE_RE = re.compile(r"^(\d+)\s*-\s*(\d+)$")
_INDEX_RE = re.compile(r"^\d+$")


def uses_modality_block_routing(train_config: Any) -> bool:
    routing = getattr(train_config, "modality_block_routing", None)
    if routing is None:
        return False
    return bool(getattr(routing, "enabled", False))


def parse_block_spec(spec: Any, num_blocks: Optional[int] = None) -> list[int]:
    """'3-12, 14-15, 22,27,31-33' -> sorted unique indices.

    Fail closed on empty input, unreadable chunks, reversed ranges, duplicate
    indices (a repeated index is treated as a typo), and out-of-range values.
    """
    text = str(spec if spec is not None else "").strip()
    if not text:
        raise ValueError("no blocks given")
    out: list[int] = []
    seen: set[int] = set()

    def add(index: int, chunk: str) -> None:
        if index in seen:
            raise ValueError(
                f"duplicate block index {index} in {text!r} (from {chunk!r})"
            )
        seen.add(index)
        out.append(index)

    for part in text.split(","):
        chunk = part.strip()
        if not chunk:
            continue
        matched = _RANGE_RE.fullmatch(chunk)
        if matched:
            lo, hi = int(matched.group(1)), int(matched.group(2))
            if lo > hi:
                raise ValueError(f"range runs backwards: {chunk!r}")
            for index in range(lo, hi + 1):
                add(index, chunk)
            continue
        if _INDEX_RE.fullmatch(chunk):
            add(int(chunk), chunk)
            continue
        raise ValueError(
            f"cannot read {chunk!r} — use numbers and ranges, "
            f"e.g. '3-12, 14-15, 22, 31-33'"
        )
    if not out:
        raise ValueError("no blocks given")
    if num_blocks is not None:
        bad = sorted(i for i in out if i < 0 or i >= num_blocks)
        if bad:
            raise ValueError(
                f"block(s) {bad} do not exist — this model has {num_blocks} "
                f"(0-{num_blocks - 1})"
            )
    return sorted(out)


def format_block_spec(indices: Iterable[int]) -> str:
    ordered = sorted(indices)
    if not ordered:
        return ""
    runs: list[tuple[int, int]] = []
    start = prev = ordered[0]
    for index in ordered[1:]:
        if index == prev + 1:
            prev = index
            continue
        runs.append((start, prev))
        start = prev = index
    runs.append((start, prev))
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def classify_file_item(item: Any) -> str:
    """photo | clip | voice. Voice is ``is_audio_only`` (WS5 standalone recordings)."""
    if bool(getattr(item, "is_audio_only", False)):
        return "voice"
    if bool(getattr(item, "is_video", False)):
        return "clip"
    return "photo"


def classify_batch_kinds(batch: Any) -> set[str]:
    items = getattr(batch, "file_items", None) or ()
    return {classify_file_item(item) for item in items}


def window_kind(kinds: set[str]) -> str:
    if len(kinds) == 1:
        return next(iter(kinds))
    return "mixed"


def is_h3_arch(arch: Any) -> bool:
    return str(arch or "").startswith(H3_ARCH_PREFIX)


def resolve_h3_transformer(model: Any) -> torch.nn.Module:
    candidates = [
        getattr(model, "model", None),
        model,
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            inner = unwrap_model(candidate)
        except Exception:
            inner = candidate
        if hasattr(inner, "blocks"):
            return inner
    raise ValueError(
        "modality_block_routing requires an H3 transformer with a .blocks "
        "ModuleList (model.arch minimax_h3*)."
    )


def _adapter_modules(network: Any) -> list[Any]:
    modules: list[Any] = []
    for attr in ("unet_loras", "text_encoder_loras"):
        modules.extend(getattr(network, attr, None) or [])
    if not modules:
        raise ValueError(
            "modality_block_routing requires a LoRA/LoKr network with "
            "unet_loras; full fine-tune routing is not implemented."
        )
    missing = [m for m in modules if not callable(getattr(m, "orig_module_ref", None))]
    if missing:
        raise ValueError(
            "modality_block_routing maps adapters by orig_module_ref(); "
            f"{len(missing)} adapter module(s) lack it. Do not parse lora_name."
        )
    return modules


def build_trunk_param_index(
    transformer: torch.nn.Module,
    network: Any,
) -> tuple[dict[int, list[torch.nn.Parameter]], list[torch.nn.Parameter]]:
    """Map trunk block index -> LoRA params; non-trunk params stay always-active."""
    id_to_path = {id(module): name for name, module in transformer.named_modules()}
    by_block: dict[int, list[torch.nn.Parameter]] = {}
    always_active: list[torch.nn.Parameter] = []
    trunk_hits = 0
    for adapter in _adapter_modules(network):
        orig = adapter.orig_module_ref()
        if orig is None:
            raise ValueError(
                "modality_block_routing: orig_module_ref() returned None; "
                "the wrapped module was collected."
            )
        path = id_to_path.get(id(orig), "")
        matched = _TRUNK_BLOCK_RE.match(path)
        params = list(adapter.parameters())
        if matched:
            index = int(matched.group(1))
            by_block.setdefault(index, []).extend(params)
            trunk_hits += 1
        else:
            always_active.extend(params)
    if trunk_hits == 0:
        raise ValueError(
            "modality_block_routing matched no trunk LoRA modules via "
            "orig_module_ref() against transformer.named_modules() paths "
            "starting with 'blocks.<idx>.'. Refusing to guess from lora_name."
        )
    return by_block, always_active


class ModalityBlockRouter:
    """Resolve the active parameter set for one optimizer window."""

    def __init__(
        self,
        *,
        photo_blocks: Optional[set[int]],
        clip_blocks: Optional[set[int]],
        voice_blocks: Optional[set[int]],
        by_block: dict[int, list[torch.nn.Parameter]],
        always_active: list[torch.nn.Parameter],
        all_params: list[torch.nn.Parameter],
    ):
        self.allowed = {
            "photo": photo_blocks,
            "clip": clip_blocks,
            "voice": voice_blocks,
        }
        self.by_block = by_block
        self.always_active = always_active
        self.all_params = all_params
        self._kinds: set[str] = set()

    def observe_batch(self, batch: Any) -> None:
        self._kinds |= classify_batch_kinds(batch)

    def reset_window(self) -> None:
        self._kinds = set()

    def consume_window(self) -> Optional[list[torch.nn.Parameter]]:
        kinds = self._kinds
        self._kinds = set()
        return self.active_params_for_kinds(kinds)

    def active_params_for_batch(self, batch: Any) -> Optional[list[torch.nn.Parameter]]:
        return self.active_params_for_kinds(classify_batch_kinds(batch))

    def active_params_for_kinds(
        self, kinds: set[str]
    ) -> Optional[list[torch.nn.Parameter]]:
        if len(kinds) != 1:
            return None
        kind = next(iter(kinds))
        allowed = self.allowed.get(kind)
        if allowed is None:
            return None
        return self._params_for_blocks(allowed)

    def _params_for_blocks(self, allowed: set[int]) -> list[torch.nn.Parameter]:
        out = list(self.always_active)
        for index in allowed:
            out.extend(self.by_block.get(index, []))
        return out


def bind_modality_router(
    train_config: Any,
    model: Any,
    network: Any,
) -> Optional[ModalityBlockRouter]:
    if not uses_modality_block_routing(train_config):
        return None
    arch = getattr(model, "arch", None)
    if arch is None:
        model_config = getattr(model, "model_config", None)
        arch = getattr(model_config, "arch", None)
    if not is_h3_arch(arch):
        raise ValueError(
            f"modality_block_routing is H3-only (gate by model.arch), got arch={arch!r}."
        )
    if network is None:
        raise ValueError(
            "modality_block_routing requires a trainable LoRA/LoKr network."
        )
    transformer = resolve_h3_transformer(model)
    n_blocks = len(transformer.blocks)
    routing = train_config.modality_block_routing

    def _parse(spec: Any) -> Optional[set[int]]:
        if spec is None or str(spec).strip() == "":
            return None
        indices = parse_block_spec(spec, n_blocks)
        if not indices:
            raise ValueError(f"modality_block_routing spec {spec!r} resolved empty")
        return set(indices)

    photo = _parse(routing.photo_blocks)
    clip = _parse(routing.clip_blocks)
    voice = _parse(routing.voice_blocks)
    by_block, always_active = build_trunk_param_index(transformer, network)
    all_params: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for group in (always_active, *(by_block.values())):
        for param in group:
            ident = id(param)
            if ident in seen:
                continue
            seen.add(ident)
            all_params.append(param)
    router = ModalityBlockRouter(
        photo_blocks=photo,
        clip_blocks=clip,
        voice_blocks=voice,
        by_block=by_block,
        always_active=always_active,
        all_params=all_params,
    )
    parts = []
    if photo is not None:
        parts.append(
            f"photo -> blocks {format_block_spec(photo)} "
            f"(+refiners/non-trunk; {len(router._params_for_blocks(photo))} tensors)"
        )
    else:
        parts.append("photo -> unrestricted")
    if clip is not None:
        parts.append(f"clip -> blocks {format_block_spec(clip)}")
    else:
        parts.append("clip -> unrestricted")
    if voice is not None:
        parts.append(f"voice -> blocks {format_block_spec(voice)}")
    else:
        parts.append("voice -> unrestricted (no standalone voice items until WS5)")
    print_acc("[modality-routing] " + "; ".join(parts))
    return router
