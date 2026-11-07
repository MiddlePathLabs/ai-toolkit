"""Sampling-recipe knowledge for MOSS-Image preview Turbo LoRAs.

The frozen-network machinery is shared (``toolkit/preview_lora.py``); what
varies per adapter is the step count. All three known community turbos run
CFG 1 on the stock shifted sigma grid (``moss_sigmas``, shift 5) — none
ships its own sigma schedule — so the recipe here reduces to "force CFG 1,
pick N steps":

  - ``dmd`` — lodestones/MOSS-Image-Base-8-steps-dmd (2026-11). DMD2
    distillation of OpenMOSS-Team/MOSS-Image-**Base**, our training
    substrate. Card: 4-8 steps, CFG 1, shift 5, standard Euler; works as a
    plain LoRA on the stock transformer.
  - ``sft4`` — akhaliq/MOSS-Image-SFT-4-steps (2026-11). 4 steps up to
    ~1024x1024, 6 around 1536-wide, 8 at 2048x2048, CFG 1. Distilled on
    **SFT** (pairs with ``preview_checkpoint``); the card says it also runs
    on Base with slightly softer output. NEVER at the default 25-50 step /
    CFG 3 template — that smudges and overcooks.
  - ``c16`` — Coffmeteor/MOSS-Image-Turbo (2026-11). 16 steps, CFG 1,
    shift 5, Euler/simple; distilled on **SFT** bf16.

Turbo adapters are distilled against one specific base; running a
Base-distilled file on the SFT preview DiT (or the reverse) renders, but is
off-recipe — quality is the adapter's property, not the plumbing's.
"""

from __future__ import annotations

import os
import re
from typing import Optional

MOSS_PREVIEW_SCHEDULES = ("auto", "dmd", "sft4", "c16")

# dmd card's working range; configured steps are clamped into it
DMD_MIN_STEPS = 4
DMD_MAX_STEPS = 8
# sft4 card: 4 steps <=1024^2, 6 for ~1536-wide (~2.3 MP), 8 at 2048^2
SFT4_MP_BANDS = (1.05, 2.5)  # megapixel thresholds between 4 / 6 / 8 steps
# c16 card: a fixed 16-step single pass
C16_STEPS = 16


# "4-steps", "8step", "16_steps", ... (not "v4steps"-style tails of other words)
_STEP_COUNT_RE = re.compile(r"(?<![a-z0-9])(\d{1,2})[-_ ]?steps?(?![a-z])")


def _schedule_hint_text(path: str, source: Optional[str]) -> str:
    """The text 'auto' matches against: the resolved file name plus the last
    two segments of the configured path. A hub path
    (``user/repo/file.safetensors``) often names the recipe only in the repo
    (``.../MOSS-Image-SFT-4-steps/model.safetensors``), and that is gone once
    the path resolves to a local file. Earlier segments (the user name, local
    directories) are left out so they cannot steer the pick."""
    parts = [os.path.basename(path)]
    if source:
        segments = [s for s in re.split(r"[\\/]", str(source)) if s]
        parts.extend(segments[-2:])
    return " ".join(parts).lower()


def resolve_moss_preview_schedule(
    schedule: str, path: str, source: Optional[str] = None
) -> str:
    """'auto' picks by name (``path``'s file name and, when given, the
    configured ``source`` path's repo/file segments): ``dmd`` by name, then
    an explicit step count (4 -> sft4, <=8 -> dmd, more -> c16), then
    ``turbo`` -> c16. Unknown files take the dmd recipe (any low-step
    distilled adapter is safest at <=8 steps, CFG 1)."""
    schedule = (schedule or "auto").lower()
    if schedule not in MOSS_PREVIEW_SCHEDULES:
        raise ValueError(
            f"preview_lora_schedule must be one of {MOSS_PREVIEW_SCHEDULES} "
            f"for moss_image, got {schedule!r}"
        )
    if schedule != "auto":
        return schedule
    name = _schedule_hint_text(path, source)
    if "dmd" in name:
        return "dmd"
    match = _STEP_COUNT_RE.search(name)
    if match is not None:
        steps = int(match.group(1))
        if steps == 4:
            return "sft4"
        return "dmd" if steps <= DMD_MAX_STEPS else "c16"
    if "sft-4" in name or "sft4" in name or "sft_4" in name:
        return "sft4"
    if "turbo" in name:
        # Coffmeteor/MOSS-Image-Turbo (15/16-step SFT acceleration)
        return "c16"
    return "dmd"


def moss_turbo_steps(schedule: str, num_steps: int, width: int, height: int) -> int:
    """The step count a schedule renders at, from the configured
    ``num_steps`` and the sample resolution."""
    if schedule == "sft4":
        mp = (width * height) / (1024.0 * 1024.0)
        if mp <= SFT4_MP_BANDS[0]:
            return 4
        if mp <= SFT4_MP_BANDS[1]:
            return 6
        return 8
    if schedule == "c16":
        return C16_STEPS
    # dmd: honor a configured count inside the card's 4-8 window
    return max(DMD_MIN_STEPS, min(int(num_steps), DMD_MAX_STEPS))
