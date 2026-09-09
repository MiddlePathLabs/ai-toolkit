"""H3 TREAD token routing (Krause et al., arXiv 2501.04765).

Training-only, default-off. On a clip step a random ``ratio`` of *target*
video rows skip blocks ``[start, end)`` and rejoin in their start-block state
(kept fraction ``1 - ratio``). Live text, condition, audio, and pad rows
always stay. Stills (one latent
frame), batch size != 1, and inference (no grad) never route. VSA is rejected
until a routed VSA context exists.

Algorithmically equivalent to Fizgig's clip-step route, not bit-compatible:
this pack is ``[text | condition | audio | target video]`` with a batch axis
and an attention mask, so gather also subsets rotary, AdaLN indices, and mask
key columns.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

from toolkit.h3_modality_routing import is_h3_arch, resolve_h3_transformer
from toolkit.print import print_acc

# TREAD supplement: bf16 instability when only 1-3 blocks remain after rejoin.
# Fizgig's prior is end=47 on a 50-block trunk (3 remaining). Remaining < 3
# is rejected at bind; remaining == 3 is the known prior, not a new experiment.
MIN_POST_REJOIN_BLOCKS = 3
H3_VSA_ARCH = "minimax_h3_vsa"


def uses_tread(train_config: Any) -> bool:
    ratio = getattr(train_config, "tread_ratio", 0.0)
    try:
        return float(ratio) > 0.0
    except (TypeError, ValueError):
        return False


def validate_tread_span(start: int, end: int, n_blocks: int) -> None:
    if n_blocks < MIN_POST_REJOIN_BLOCKS + 1:
        raise ValueError(
            f"TREAD needs a trunk with at least {MIN_POST_REJOIN_BLOCKS + 1} "
            f"blocks, got {n_blocks}."
        )
    if not (0 <= start < end <= n_blocks):
        raise ValueError(
            f"TREAD span [{start}, {end}) is not inside a {n_blocks}-block "
            f"trunk (need 0 <= start < end <= {n_blocks})."
        )
    remaining = n_blocks - end
    if remaining < MIN_POST_REJOIN_BLOCKS:
        max_end = n_blocks - MIN_POST_REJOIN_BLOCKS
        raise ValueError(
            f"TREAD rejoins at block {end}, leaving {remaining} block(s) "
            f"before the final layer. The TREAD supplement reports bf16 "
            f"instability with fewer than {MIN_POST_REJOIN_BLOCKS} remaining "
            f"blocks. Raise tread_end no higher than {max_end} for this "
            f"{n_blocks}-block trunk."
        )


def _arch_of(model: Any) -> Any:
    arch = getattr(model, "arch", None)
    if arch is None:
        model_config = getattr(model, "model_config", None)
        arch = getattr(model_config, "arch", None)
    return arch


def bind_tread(train_config: Any, model: Any) -> Optional[Tuple[float, int, int]]:
    """Install ``transformer._tread`` or clear it. Fail closed before the first batch."""
    arch = _arch_of(model)
    if not uses_tread(train_config):
        if is_h3_arch(arch):
            try:
                transformer = resolve_h3_transformer(model)
                transformer._tread = None
                transformer._tread_generator = None
            except ValueError:
                pass
        return None
    if not is_h3_arch(arch):
        raise ValueError(
            f"tread_ratio is H3-only (gate by model.arch), got arch={arch!r}."
        )
    if str(arch) == H3_VSA_ARCH:
        raise ValueError(
            "TREAD token routing is not available with VSA "
            "(model.arch=minimax_h3_vsa) until a routed VSA context exists. "
            "Set tread_ratio: 0 or use a dense H3 checkpoint."
        )
    try:
        transformer = resolve_h3_transformer(model)
    except ValueError:
        raise ValueError(
            "tread_ratio requires an H3 transformer with a .blocks "
            "ModuleList (model.arch minimax_h3*)."
        ) from None
    if getattr(transformer, "vsa_sparsity", None) is not None:
        raise ValueError(
            "TREAD token routing is not available while transformer.vsa_sparsity "
            "is set. Disable VSA or set tread_ratio: 0."
        )
    params = getattr(transformer, "params", None)
    if bool(getattr(params, "gate_compress", False)):
        raise ValueError(
            "TREAD token routing is not available on a VSA/gate_compress "
            "checkpoint until a routed VSA context exists. Set tread_ratio: 0."
        )
    ratio = float(train_config.tread_ratio)
    start = int(train_config.tread_start)
    end = int(train_config.tread_end)
    n_blocks = len(transformer.blocks)
    validate_tread_span(start, end, n_blocks)
    transformer._tread = (ratio, start, end)
    seed = getattr(train_config, "seed", None)
    if seed is None:
        seed = torch.initial_seed()
    transformer._tread_generator = torch.Generator(device="cpu").manual_seed(
        int(seed) & 0xFFFFFFFF
    )
    print_acc(
        f"[tread] token routing ON — {ratio:.0%} of target video tokens skip "
        f"blocks {start}-{end - 1} on CLIP steps (rejoin in block-{start} "
        f"state; text, condition and audio rows stay; stills, batch>1, and "
        f"previews never route). arXiv 2501.04765."
    )
    return transformer._tread


def plan_tread_keep_idx(
    *,
    seq_len: int,
    batch_size: int,
    video_indices: torch.Tensor,
    target_video_grid: Optional[Tuple[int, int, int]],
    ratio: float,
    start: int,
    end: int,
    n_blocks: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> Optional[torch.Tensor]:
    """Sorted packed-sequence indices that stay through ``[start, end)``.

    Target video is the last ``prod(grid)`` entries of ``video_indices`` —
    packing puts condition video first, then target video. Returns None when
    routing must not run (still, batch != 1, empty/invalid span).
    """
    if batch_size != 1:
        return None
    if target_video_grid is None:
        return None
    t, h, w = (int(x) for x in target_video_grid)
    if t <= 1:
        return None
    n_target = t * h * w
    if n_target <= 1:
        return None
    if not (0.0 < float(ratio) < 1.0):
        return None
    if not (0 <= int(start) < int(end) <= int(n_blocks)):
        return None
    if int(video_indices.numel()) < n_target:
        raise ValueError(
            f"TREAD target grid {target_video_grid} needs {n_target} video "
            f"rows, got {int(video_indices.numel())}"
        )
    target = video_indices[-n_target:]
    n_keep = max(1, int(round(n_target * (1.0 - float(ratio)))))
    if generator is None:
        perm = torch.randperm(n_target, device=device)
    else:
        perm = torch.randperm(
            n_target, generator=generator, device=generator.device
        ).to(device)
    keep_vid = target[perm[:n_keep].sort().values]
    live = torch.ones(seq_len, dtype=torch.bool, device=device)
    live[target] = False
    live[keep_vid] = True
    return live.nonzero(as_tuple=False).reshape(-1)


def gather_tread_state(
    x: torch.Tensor,
    rotary_emb: Tuple[torch.Tensor, torch.Tensor],
    adaln_indices: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    keep_idx: torch.Tensor,
):
    """Subset packed rows, rotary, AdaLN indices, and attention-mask keys."""
    x_r = x.index_select(1, keep_idx)
    cos, sin = rotary_emb
    rotary_r = (cos.index_select(1, keep_idx), sin.index_select(1, keep_idx))
    adaln_r = adaln_indices.index_select(1, keep_idx)
    mask_r = None if attn_mask is None else attn_mask.index_select(-1, keep_idx)
    return x_r, rotary_r, adaln_r, mask_r


def scatter_tread_hidden(
    x_reduced: torch.Tensor,
    x_full: torch.Tensor,
    keep_idx: torch.Tensor,
) -> torch.Tensor:
    """Write kept rows into a clone of the start-block full sequence."""
    x_new = x_full.clone()
    x_new.index_copy_(1, keep_idx, x_reduced)
    return x_new
