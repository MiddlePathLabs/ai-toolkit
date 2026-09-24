"""``train.low_noise_share``: set how much of training lands below sigma 0.5.

The 'shift' timestep type draws indices uniformly over a grid u in
[sigma_min, sigma_max] of the scheduler and maps each through the static shift
``sigma = s*u / (1 + (s-1)*u)``. Below sigma 0.5 means ``u < 1/(1+s)``, so the
share of the grid there is ``(1/(1+s) - u_min) / (u_max - u_min)``. Solving for
``s`` gives the training shift that puts exactly ``p`` of the grid below 0.5.

On MiniMax-H3 (model shift 12) the default draw puts ~6.6% of steps below 0.5.
Fizgig's "Likeness and Style" recipe is 0.6.

Only the training draw changes. The scheduler's configured shift, sampling
and previews keep the model's own shift.

H3 audio: the share is a VIDEO share. H3 pairs each video sigma with an audio
sigma through ``remap_sigma(sigma_v)`` (shift 12 -> 3), the same fixed pairing
the inference pipeline uses at every step, so training stays on the
(sigma_v, sigma_a) pairs the model actually sees. Remapping from the training
shift instead would pair a clean video sigma with a noisy audio sigma that
inference never produces. Fizgig trains its 0.6 recipe with the same shift-12
remap. The consequence is that audio lands cleaner than the video share says:
0.6 video puts ~86% of audio draws below 0.5 (model default ~24%). Both are
logged at bind.
Fail-closed on anything that would bend the grid so the share is no longer
what the number says (dynamic shifting, karras/exponential/beta sigmas,
shift_terminal, cubic content/style bias, a narrowed denoising range,
first_timestep_chance, multistage models).
"""
from __future__ import annotations

import inspect
from typing import Any, Optional

import torch

from toolkit.h3_modality_routing import is_h3_arch
from toolkit.print import print_acc

LOW_NOISE_SIGMA = 0.5


def shift_for_low_noise_share(share: float, u_min: float, u_max: float) -> float:
    """Static shift that puts ``share`` of a uniform u-grid on [u_min, u_max]
    below sigma 0.5."""
    if not (0.0 < share < 1.0):
        raise ValueError(f"low_noise_share must be in (0, 1), got {share}")
    if not (0.0 <= u_min < LOW_NOISE_SIGMA < u_max <= 1.0):
        raise ValueError(
            f"training grid [{u_min}, {u_max}] must straddle sigma {LOW_NOISE_SIGMA}"
        )
    u_star = u_min + share * (u_max - u_min)
    return 1.0 / u_star - 1.0


def low_noise_share_of_shift(shift: float, u_min: float, u_max: float) -> float:
    """Inverse of shift_for_low_noise_share (for logging a model's default)."""
    u_star = 1.0 / (1.0 + shift)
    return min(1.0, max(0.0, (u_star - u_min) / (u_max - u_min)))


def _shifted(shift: float, u: float) -> float:
    return shift * u / (1.0 + (shift - 1.0) * u)


def h3_audio_share(shift: float, u_min: float, u_max: float, n: int = 1000) -> float:
    """Share of H3 audio sigmas below 0.5 when video is drawn at ``shift`` on a
    uniform u-grid and paired through the model's fixed remap_sigma."""
    from extensions_built_in.diffusion_models.minimax_h3.src.packing import remap_sigma

    u = torch.linspace(u_max, u_min, n, dtype=torch.float64)
    sigma_v = shift * u / (1.0 + (shift - 1.0) * u)
    return float((remap_sigma(sigma_v) < LOW_NOISE_SIGMA).double().mean())


def bind_low_noise_share(train_config: Any, sd: Any) -> Optional[float]:
    """Validate ``train.low_noise_share`` against the loaded model and return
    the training shift to pass as ``set_train_timesteps(shift_override=...)``.
    None when the option is off."""
    share = getattr(train_config, "low_noise_share", None)
    if share is None:
        return None

    problems = []
    if getattr(train_config, "noise_scheduler", None) != "flowmatch":
        problems.append(f"noise_scheduler={train_config.noise_scheduler!r} (needs 'flowmatch')")
    if getattr(train_config, "timestep_type", None) != "shift":
        problems.append(f"timestep_type={train_config.timestep_type!r} (needs 'shift')")
    if getattr(train_config, "linear_timesteps", False) or getattr(train_config, "linear_timesteps2", False):
        problems.append("linear_timesteps/linear_timesteps2 is on")
    for key in ("content_or_style", "content_or_style_reg"):
        value = getattr(train_config, key, "balanced")
        if value != "balanced":
            problems.append(f"{key}={value!r} (needs 'balanced'; the cubic bias would stack)")
    num_train_timesteps = int(getattr(train_config, "num_train_timesteps", 1000))
    if int(getattr(train_config, "min_denoising_steps", 0)) != 0 or int(
        getattr(train_config, "max_denoising_steps", num_train_timesteps - 1)
    ) < num_train_timesteps - 1:
        problems.append("min/max_denoising_steps narrow the range")
    if float(getattr(train_config, "first_timestep_chance", 0.0) or 0.0) > 0.0:
        problems.append("first_timestep_chance > 0")
    if getattr(sd, "is_multistage", False):
        problems.append("multistage models pick their own timestep ranges")

    scheduler = getattr(sd, "noise_scheduler", None)
    set_train = getattr(scheduler, "set_train_timesteps", None)
    if set_train is None or "shift_override" not in inspect.signature(set_train).parameters:
        problems.append(
            f"scheduler {type(scheduler).__name__} does not support a training shift override"
        )
    config = getattr(scheduler, "config", None)
    if config is not None:
        if config.get("use_dynamic_shifting", False):
            problems.append("scheduler uses dynamic (resolution-dependent) shifting")
        for key in ("shift_terminal", "use_karras_sigmas", "use_exponential_sigmas",
                    "use_beta_sigmas", "invert_sigmas"):
            if config.get(key, None):
                problems.append(f"scheduler {key} is set")
    if problems:
        raise ValueError(
            "train.low_noise_share needs a static-shift flowmatch draw it can "
            "describe exactly: " + "; ".join(problems)
        )

    u_min = float(scheduler.sigma_min)
    u_max = float(scheduler.sigma_max)
    shift = shift_for_low_noise_share(float(share), u_min, u_max)
    model_shift = float(scheduler.shift)
    default_share = low_noise_share_of_shift(model_shift, u_min, u_max)
    print_acc(
        f"[low-noise-share] {share:.0%} of training draws below sigma 0.5 "
        f"(model default {default_share:.1%}): training shift {shift:.4g} on the "
        f"u-grid [{u_min:.4g}, {u_max:.4g}], sigma range "
        f"[{_shifted(shift, u_min):.4g}, {_shifted(shift, u_max):.4g}]. "
        f"Model shift {model_shift:g} is unchanged for sampling."
    )
    arch = getattr(sd, "arch", None) or getattr(getattr(sd, "model_config", None), "arch", None)
    if is_h3_arch(arch):
        print_acc(
            f"[low-noise-share] H3 audio rows keep the model's fixed video->audio "
            f"pairing (remap from shift 12, as at inference): audio below sigma 0.5 on "
            f"{h3_audio_share(shift, u_min, u_max):.1%} of draws (model default "
            f"{h3_audio_share(model_shift, u_min, u_max):.1%}). The share above is "
            f"the video share."
        )
    return shift
