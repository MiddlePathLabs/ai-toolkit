"""Sampling-only Turbo LoRA for Qwen-Image 2.1 previews.

Ports the MiniMax-H3 preview-turbo pattern (``minimax_h3/src/turbo.py``) to
this arch: the adapter file is matched against the loaded transformer's
Linear modules by shape, wrapped as a frozen ``LoRASpecialNetwork``, parked
on CPU, and toggled around ``generate_images`` via the
``_enter/_exit_generate_adapters`` hooks. Unlike the pruned H3 base there
are no AdaLN pairs to inject through an e-grid — every entry in a
well-formed file matches a Linear shape directly, and anything that does
not is reported as skipped instead of silently dropped.

The generic load/match machinery lives in ``toolkit/preview_lora.py``
(shared with moss_image); this module keeps the Qwen-specific sampling
recipes and re-exports the shared pieces it was the original home of.

Two preview adapters are known, each with its own sampling recipe:

  - ``turbo``: Comfy-Org's Qwen-Image-2.1 Turbo LoRA (extracted from Qwen's
    Turbo checkpoint). 8 steps, CFG 1, the Turbo checkpoint's fixed sigmas
    with no shift. The file also carries ``.diff`` deltas for the q/k and
    text-input RMSNorm weights, applied while the adapter is active.
  - ``viggle``: Viggle's community turbo. 5+ steps on its own sigma grid,
    with the pipeline's resolution shift on top.

The same frozen-network machinery also loads the training adapter
(``assistant_lora_path``): active for training steps, off for previews.
"""

from __future__ import annotations

import os
import re
from typing import List, Optional

# shared machinery (this module was its first home; moved to toolkit when
# moss_image grew its own preview adapters)
from toolkit.preview_lora import (  # noqa: F401  (re-exported for callers)
    PreviewLoadReport,
    PreviewTurbo,
    load_frozen_lora,
    partition_lora_weights,
    partition_weight_diffs,
    resolve_preview_lora_path,
)

# back-compat alias for the pre-split name
QwenPreviewTurbo = PreviewTurbo

PREVIEW_SCHEDULES = ("auto", "turbo", "viggle")

# Qwen-Image-2.1-Turbo's own schedule (model_index.json sample_sigmas). Its
# scheduler_config turns the dynamic shift off (use_dynamic_shifting false,
# shift 1.0, shift_terminal null), so these are used exactly as given.
TURBO_SIGMAS = (1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568)
TURBO_STEPS = len(TURBO_SIGMAS)


def resolve_preview_schedule(
    schedule: str, path: str, source: Optional[str] = None
) -> str:
    """'auto' picks by name: Viggle's releases carry 'viggle' in theirs,
    anything else is taken to be the Comfy-Org / Qwen Turbo LoRA. The
    configured ``source`` path's last two segments (hub repo + file) count
    too, since a downloaded file's own name may not carry the hint."""
    schedule = (schedule or "auto").lower()
    if schedule not in PREVIEW_SCHEDULES:
        raise ValueError(
            f"preview_lora_schedule must be one of {PREVIEW_SCHEDULES}, got {schedule!r}"
        )
    if schedule != "auto":
        return schedule
    names = [os.path.basename(path)]
    if source:
        names.extend([seg for seg in re.split(r"[\\/]", str(source)) if seg][-2:])
    return "viggle" if "viggle" in " ".join(names).lower() else "turbo"

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


def load_preview_lora(
    dit,
    path: str,
    strength: float = 1.0,
    *,
    target_lin_modules=None,
    schedule: str = "auto",
    source: Optional[str] = None,
) -> PreviewTurbo:
    from toolkit.preview_lora import load_frozen_lora as _load_frozen

    turbo = _load_frozen(
        dit,
        path,
        strength,
        target_lin_modules=target_lin_modules,
        label="Qwen preview LoRA",
        tag="qwen-preview-lora",
    )
    turbo.schedule = resolve_preview_schedule(schedule, path, source=source)
    return turbo
