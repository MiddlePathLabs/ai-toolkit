"""``network.train_blocks``: LoRA on a range of H3 trunk blocks and nothing else.

With no trainable parameter before the first selected block, autograd never
needs a gradient there, so the backward pass stops at that block instead of
running the whole 50-block trunk. That is the difference from
``modality_block_routing``, which masks the optimizer after a full backward and
keeps the token refiner training.

Selection goes through the existing ``only_if_contains`` / ``ignore_if_contains``
name filters (``blocks.<i>.``, refiner excluded), then is verified by module
ownership (``orig_module_ref`` against ``transformer.named_modules()``, the same
mapping modality routing uses), so a naming change can only fail the run, never
widen it. A one-shot forward hook on the block just before the window checks
the guarantee on the first training forward.
"""
from __future__ import annotations

from typing import Any, Optional

import torch

from toolkit.h3_modality_routing import (
    _TRUNK_BLOCK_RE,
    _adapter_modules,
    format_block_spec,
    is_h3_arch,
    parse_block_spec,
    resolve_h3_transformer,
)
from toolkit.print import print_acc

_UNSUPPORTED_NETWORK_TYPES = ("locon", "lycoris", "lorm")


def _arch_of(sd: Any) -> Any:
    arch = getattr(sd, "arch", None)
    if arch is None:
        arch = getattr(getattr(sd, "model_config", None), "arch", None)
    return arch


def prepare_train_blocks(
    network_config: Any,
    sd: Any,
    network_kwargs: dict,
    *,
    train_text_encoder: bool = False,
    train_embedding: bool = False,
) -> tuple[dict, Optional[list[int]]]:
    """Return (network_kwargs to build with, selected block indices or None)."""
    spec = getattr(network_config, "train_blocks", None)
    if spec is None:
        return network_kwargs, None
    arch = _arch_of(sd)
    if not is_h3_arch(arch):
        raise ValueError(f"network.train_blocks is H3-only (model.arch minimax_h3*), got {arch!r}")
    if train_text_encoder or train_embedding:
        # text rows run through every block, so a trainable text path would
        # carry the backward through the frozen blocks anyway
        raise ValueError(
            "network.train_blocks cannot stop the backward while the text encoder "
            "or an embedding is trained (text rows pass through every block)"
        )
    network_type = str(getattr(network_config, "type", "lora")).lower()
    if network_type in _UNSUPPORTED_NETWORK_TYPES:
        raise ValueError(
            f"network.train_blocks supports LoRA/LoKr networks, got type {network_type!r}"
        )
    if network_kwargs.get("only_if_contains"):
        raise ValueError(
            "network.train_blocks and network_kwargs.only_if_contains both pick "
            "modules; use one of them"
        )
    transformer = resolve_h3_transformer(sd)
    indices = parse_block_spec(spec, len(transformer.blocks))
    kwargs = dict(network_kwargs)
    kwargs["only_if_contains"] = [f"blocks.{i}." for i in indices]
    ignore = list(kwargs.get("ignore_if_contains") or [])
    if "token_refiner" not in ignore:
        # the refiner nests its own blocks.<i>. — keep it out by name, then
        # verify by ownership below
        ignore.append("token_refiner")
    kwargs["ignore_if_contains"] = ignore
    return kwargs, indices


def verify_train_blocks(
    sd: Any,
    network: Any,
    indices: Optional[list[int]],
    *,
    arm_runtime_check: bool = True,
) -> None:
    """Fail closed unless every adapter is owned by a selected trunk block and
    every selected block got at least one; then arm the one-shot backward
    check. The check is a Python forward hook, so it is skipped when the model
    is torch.compiled (removing a hook inside a compiled graph recompiles it)."""
    if indices is None:
        return
    transformer = resolve_h3_transformer(sd)
    allowed = set(indices)
    id_to_path = {id(module): name for name, module in transformer.named_modules()}
    per_block: dict[int, int] = {}
    outside: list[str] = []
    n_params = 0
    for adapter in _adapter_modules(network):
        orig = adapter.orig_module_ref()
        path = id_to_path.get(id(orig), "") if orig is not None else ""
        matched = _TRUNK_BLOCK_RE.match(path)
        if matched is None or int(matched.group(1)) not in allowed:
            outside.append(path or getattr(adapter, "lora_name", "?"))
            continue
        index = int(matched.group(1))
        per_block[index] = per_block.get(index, 0) + 1
        n_params += sum(p.numel() for p in adapter.parameters())
    if outside:
        shown = ", ".join(outside[:5]) + (" ..." if len(outside) > 5 else "")
        raise ValueError(
            f"network.train_blocks {format_block_spec(indices)}: {len(outside)} adapter "
            f"module(s) outside the window ({shown}). They would pull the backward "
            f"past block {min(indices)}."
        )
    missing = sorted(allowed - set(per_block))
    if missing:
        raise ValueError(
            f"network.train_blocks: block(s) {format_block_spec(missing)} got no LoRA "
            f"module (did ignore_if_contains remove them all?)"
        )
    counts = sorted(set(per_block.values()))
    per = f"{counts[0]} per block" if len(counts) == 1 else f"{counts[0]}-{counts[-1]} per block"
    first = min(indices)
    print_acc(
        f"[train-blocks] LoRA on blocks {format_block_spec(indices)} only: "
        f"{sum(per_block.values())} modules ({per}), {n_params:,} params. "
        + (
            f"Nothing before block {first} is trainable; the backward stops there."
            if first > 0
            else "Block 0 is selected, so the backward runs the whole trunk."
        )
    )
    if first > 0:
        if arm_runtime_check:
            _arm_backward_stop_check(transformer.blocks[first - 1], first)
        else:
            print_acc(
                "[train-blocks] model is compiled: skipping the first-step backward "
                "check (the module ownership check above still applies)"
            )


def _arm_backward_stop_check(block: torch.nn.Module, first: int) -> None:
    """On the first grad-enabled forward, the output of the block before the
    window must not require grad; if it does, something upstream is trainable
    and the backward would run through the frozen blocks anyway."""
    handle = None

    def hook(_module, _inputs, output):
        if not torch.is_grad_enabled():
            return
        handle.remove()
        out = output[0] if isinstance(output, (tuple, list)) else output
        if isinstance(out, torch.Tensor) and out.requires_grad:
            raise RuntimeError(
                f"network.train_blocks: the output of block {first - 1} requires grad, "
                f"so something before block {first} is trainable and the backward "
                f"would run through the frozen blocks. Check for other trainable "
                f"modules (embeddings, full-weight layers, adapters)."
            )
        print_acc(f"[train-blocks] checked: block {first - 1} output needs no grad")

    handle = block.register_forward_hook(hook)
