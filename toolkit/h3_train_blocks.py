"""``network.train_blocks``: LoRA on a range of trunk blocks and nothing else.

Supported on MiniMax-H3 (``blocks.<i>``, token refiner excluded) and
Qwen-Image 2.1 (``transformer_blocks.<i>``).

With no trainable parameter before the first selected block, autograd never
needs a gradient there, so the backward pass stops at that block instead of
running the whole trunk. On H3 that is the difference from
``modality_block_routing``, which masks the optimizer after a full backward and
keeps the token refiner training.

A spec is a block list (``"20-49"``, ``"2, 10-14"``) or a named set the arch
defines: on Qwen-Image 2.1 ``identity`` is blocks 10-14, where Fizgig's
render-based profiling of character LoRAs found the likeness lives (switching
them off removes almost all of it, on their own they carry most of it).
Training only them is Fizgig's "Fast Identity Mode": ~1.5x faster per step
with close to full-model likeness, and the rest of the base model's look is
left alone.

Selection goes through the existing ``only_if_contains`` / ``ignore_if_contains``
name filters, then is verified by module ownership (``orig_module_ref`` against
``transformer.named_modules()``, the same mapping modality routing uses), so a
naming change can only fail the run, never widen it. A one-shot forward hook on
the block just before the window checks the guarantee on the first training
forward.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from toolkit.accelerator import unwrap_model
from toolkit.h3_modality_routing import (
    _adapter_modules,
    format_block_spec,
    is_h3_arch,
    parse_block_spec,
    resolve_h3_transformer,
)
from toolkit.print import print_acc

_UNSUPPORTED_NETWORK_TYPES = ("locon", "lycoris", "lorm")


@dataclass(frozen=True)
class BlockLayout:
    attr: str  # transformer attribute holding the trunk ModuleList
    ignore: tuple = ()  # extra ignore_if_contains entries
    named: dict = field(default_factory=dict)  # named spec -> block spec

    @property
    def block_re(self) -> re.Pattern:
        return re.compile(rf"^{re.escape(self.attr)}\.(\d+)\.")

    def selector(self, index: int) -> str:
        return f"{self.attr}.{index}."


# the refiner nests its own blocks.<i>. — kept out by name, then verified by
# ownership
_H3_LAYOUT = BlockLayout(attr="blocks", ignore=("token_refiner",))
_QWEN_IMAGE_2_LAYOUT = BlockLayout(
    attr="transformer_blocks", named={"identity": "10-14"}
)


def _arch_of(sd: Any) -> Any:
    arch = getattr(sd, "arch", None)
    if arch is None:
        arch = getattr(getattr(sd, "model_config", None), "arch", None)
    return arch


def layout_for_arch(arch: Any) -> Optional[BlockLayout]:
    if is_h3_arch(arch):
        return _H3_LAYOUT
    if str(arch or "") == "qwen_image_2":
        return _QWEN_IMAGE_2_LAYOUT
    return None


def _resolve_transformer(sd: Any, layout: BlockLayout) -> torch.nn.Module:
    if layout is _H3_LAYOUT:
        return resolve_h3_transformer(sd)
    for candidate in (getattr(sd, "model", None), sd):
        if candidate is None:
            continue
        try:
            inner = unwrap_model(candidate)
        except Exception:
            inner = candidate
        if hasattr(inner, layout.attr):
            return inner
    raise ValueError(
        f"network.train_blocks: could not find the transformer's {layout.attr!r}"
    )


def _expand_named(spec: Any, layout: BlockLayout) -> Any:
    key = str(spec).strip().lower()
    return layout.named.get(key, spec)


def prepare_h3_default_guards(
    network_config: Any,
    sd: Any,
    network_kwargs: dict,
) -> dict:
    """Default-safe LoRA scope on MiniMax-H3 when no ``train_blocks`` window is
    set: the token refiner and trunk blocks 0-5 never carry a LoRA unless the
    config asks for them (``network.train_token_refiner`` /
    ``network.train_low_blocks``).

    The refiner's text rows enter at block 0, so a trainable refiner drags the
    backward through every block; training it also softened output and added
    preview judder with no likeness gain (Fizgig v5.8.0, which froze it by
    default for exactly these reasons). Blocks 0-5 deform anatomy and pull the
    dataset's colour into the render (Fizgig v6.2.0's measured defaults pass:
    likeness targets reached by epoch 8 instead of 15-21 once they were out).
    The ``train_blocks`` path needs no guard -- it is scoped by block list and
    always excludes the refiner.
    """
    if not is_h3_arch(_arch_of(sd)):
        return network_kwargs
    if getattr(network_config, "train_blocks", None) is not None:
        return network_kwargs
    ignore = list(network_kwargs.get("ignore_if_contains") or [])
    added = []
    if not getattr(network_config, "train_token_refiner", False):
        if "token_refiner" not in ignore:
            ignore.append("token_refiner")
            added.append("the token refiner")
    if not getattr(network_config, "train_low_blocks", False):
        # Trunk-qualified: the refiner nests its own blocks.<i>. (two blocks,
        # its whole attention/mlp surface), and the builder's filter is
        # substring-based -- a bare "blocks.0." would also exclude
        # token_refiner.blocks.0.* and silently kill the independent
        # train_token_refiner opt-in. LoRA names carry the "lora_transformer"
        # prefix ("transformer" in peft format); "transformer.blocks.<i>." is
        # a substring of both prefix forms' trunk paths but never of the
        # refiner's paths.
        low = [f"transformer.blocks.{i}." for i in range(6)]
        new = [w for w in low if w not in ignore]
        if new:
            ignore.extend(new)
            added.append("blocks 0-5")
    if added:
        network_kwargs = dict(network_kwargs)
        network_kwargs["ignore_if_contains"] = ignore
        print_acc(
            f"[h3-guards] default LoRA scope on MiniMax-H3 excludes {' and '.join(added)} "
            f"(network.train_token_refiner / network.train_low_blocks to opt in)"
        )
    return network_kwargs


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
    layout = layout_for_arch(arch)
    if layout is None:
        raise ValueError(
            "network.train_blocks is supported on model.arch minimax_h3* and "
            f"qwen_image_2, got {arch!r}"
        )
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
    if getattr(network_config, "train_token_refiner", False):
        raise ValueError(
            "network.train_blocks always excludes the token refiner (its text rows "
            "enter at block 0 and would drag the backward through every block); "
            "train it by removing train_blocks, not with train_token_refiner"
        )
    if network_kwargs.get("only_if_contains"):
        raise ValueError(
            "network.train_blocks and network_kwargs.only_if_contains both pick "
            "modules; use one of them"
        )
    transformer = _resolve_transformer(sd, layout)
    named = _expand_named(spec, layout)
    indices = parse_block_spec(named, len(getattr(transformer, layout.attr)))
    if named is not spec:
        print_acc(
            f"[train-blocks] {str(spec).strip()!r} = blocks {format_block_spec(indices)}"
        )
    kwargs = dict(network_kwargs)
    kwargs["only_if_contains"] = [layout.selector(i) for i in indices]
    ignore = list(kwargs.get("ignore_if_contains") or [])
    for word in layout.ignore:
        if word not in ignore:
            ignore.append(word)
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
    layout = layout_for_arch(_arch_of(sd))
    transformer = _resolve_transformer(sd, layout)
    block_re = layout.block_re
    allowed = set(indices)
    id_to_path = {id(module): name for name, module in transformer.named_modules()}
    per_block: dict[int, int] = {}
    outside: list[str] = []
    n_params = 0
    for adapter in _adapter_modules(network):
        orig = adapter.orig_module_ref()
        path = id_to_path.get(id(orig), "") if orig is not None else ""
        matched = block_re.match(path)
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
            _arm_backward_stop_check(getattr(transformer, layout.attr)[first - 1], first)
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
