"""``train.timestep_focus_prob``: land a share of the draws in a base-sigma band.

musubi-tuner's H3 timestep focus. The uniform base draw spends most steps at
very high effective noise (shift 12 maps base 0.2 to video sigma 0.75). With
probability ``prob`` a draw is replaced by one uniform over the grid points
whose base sigma lies in ``[lo, hi)`` (default 0.4-0.8, where content is
decided); otherwise it stays as drawn. Band density becomes
``prob + (1 - prob) * band_share`` and the rest of the range keeps
``1 - prob`` of the draws. musubi measured ~2x faster convergence of that band
at prob 0.5.

Base sigma is pre-shift (1 = pure noise): models with a fixed training shift
expose ``timestep_to_base_sigma`` (MiniMax-H3 undoes its shift 12), others use
timesteps / 1000. The band is intersected with min/max_denoising_steps, so a
narrowed range keeps the focused draws inside it.
"""
from __future__ import annotations

from typing import Any, Optional

import torch

from toolkit.print import print_acc

_STEP_TYPES = ("two_step", "four_step", "eight_step", "next_sample", "one_step")


def validate_timestep_focus(train_config: Any) -> None:
    """Fail closed on draws the remap would not describe."""
    prob = float(getattr(train_config, "timestep_focus_prob", 0.0) or 0.0)
    if prob <= 0.0:
        return
    problems = []
    if getattr(train_config, "timestep_type", None) in _STEP_TYPES:
        problems.append(f"timestep_type={train_config.timestep_type!r} draws fixed steps")
    for key in ("content_or_style", "content_or_style_reg"):
        value = getattr(train_config, key, "balanced")
        if value != "balanced":
            problems.append(f"{key}={value!r} (needs 'balanced'; the cubic bias would stack)")
    if problems:
        raise ValueError("train.timestep_focus_prob: " + "; ".join(problems))


def base_sigma_of_grid(timesteps: torch.Tensor, sd: Any) -> torch.Tensor:
    sigma = timesteps.float() / 1000.0
    to_base = getattr(sd, "timestep_to_base_sigma", None)
    return to_base(sigma) if callable(to_base) else sigma


def focus_band_indices(
    base_sigmas: torch.Tensor, lo: float, hi: float, min_idx: int, max_idx: int
) -> torch.Tensor:
    """Grid indices in [min_idx, max_idx] whose base sigma is in [lo, hi)."""
    idx = torch.arange(base_sigmas.shape[0], device=base_sigmas.device)
    keep = (base_sigmas >= lo) & (base_sigmas < hi) & (idx >= min_idx) & (idx <= max_idx)
    band = idx[keep]
    if band.numel() == 0:
        raise ValueError(
            f"train.timestep_focus band [{lo}, {hi}) has no grid points inside "
            f"the denoising range [{min_idx}, {max_idx}]"
        )
    return band


def apply_timestep_focus(
    timestep_indices: torch.Tensor,
    band: torch.Tensor,
    prob: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Replace each index with a uniform band index with probability ``prob``."""
    if prob <= 0.0:
        return timestep_indices
    n = timestep_indices.shape[0]
    device = timestep_indices.device
    hit = torch.rand(n, generator=generator, device="cpu").to(device) < prob
    pick = torch.randint(band.numel(), (n,), generator=generator, device="cpu").to(device)
    focused = band.to(device)[pick].to(timestep_indices.dtype)
    return torch.where(hit, focused, timestep_indices)


class TimestepFocus:
    """Caches the band for the current grid; the trainer calls it per step."""

    def __init__(self, train_config: Any, sd: Any):
        self.prob = float(getattr(train_config, "timestep_focus_prob", 0.0) or 0.0)
        self.lo = float(getattr(train_config, "timestep_focus_min", 0.4))
        self.hi = float(getattr(train_config, "timestep_focus_max", 0.8))
        self.sd = sd
        self._key = None
        self._band = None
        self._logged = False

    @property
    def enabled(self) -> bool:
        return self.prob > 0.0

    def __call__(self, timestep_indices, grid_timesteps, min_idx: int, max_idx: int):
        if not self.enabled:
            return timestep_indices
        key = (grid_timesteps.shape[0], float(grid_timesteps[0]), float(grid_timesteps[-1]),
               int(min_idx), int(max_idx))
        if key != self._key:
            base = base_sigma_of_grid(grid_timesteps, self.sd)
            self._band = focus_band_indices(base, self.lo, self.hi, min_idx, max_idx)
            self._key = key
            if not self._logged:
                share = self._band.numel() / float(max_idx - min_idx + 1)
                print_acc(
                    f"[timestep-focus] {self.prob:.0%} of draws land in base sigma "
                    f"[{self.lo:g}, {self.hi:g}) ({self._band.numel()} grid points); "
                    f"band density {self.prob + (1 - self.prob) * share:.0%} "
                    f"(uniform draw: {share:.0%})."
                )
                self._logged = True
        return apply_timestep_focus(timestep_indices, self._band, self.prob)
