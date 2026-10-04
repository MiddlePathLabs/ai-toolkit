"""Body-proportion anchor loss: visibility-weighted L1 of bone-length ratios
plus a differentiable confidence shortfall for missing reference parts.

The frozen ViTPose perceptor runs on the live x0-decoded pixels and its 8 (or
10 with head) pose-invariant ratios are matched against cached GT ratios.
Confidence is kept attached to the estimator output for the missing-body
shortfall; the hard missing fraction returned for diagnostics is detached.
Pure tensor math; imports no model weights. Body-proportion does NOT
participate in the diffusion/depth ``loss_split`` alternation -- it fires every
step within its timestep window.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def compute_body_proportion_loss(
    gen_ratios: torch.Tensor,
    gen_vis: torch.Tensor,
    ref_ratios: torch.Tensor,
    ref_vis: torch.Tensor,
    vis_threshold: float = 0.2,
    *,
    gen_confidence: Optional[torch.Tensor] = None,
    confidence_temperature: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute ratio matching plus a trainable confidence shortfall.

    ``gen_vis`` is the raw, attached per-ratio confidence used by the ratio L1
    term and, when ``gen_confidence`` is omitted, by the missing-body term.
    ``gen_confidence`` can name that same tensor explicitly for callers that
    need to expose the gradient path. High-confidence reference rows receive a
    zero-at-threshold soft hinge shortfall, with a corrective derivative while
    confidence is below ``vis_threshold``.

    Returns:
        ``(loss_per_sample, missing_fraction)`` -- both ``(B,)``; the first
        carries the gradient and the second is a detached diagnostic.
    """
    combined_vis = torch.min(ref_vis, gen_vis)
    weighted_diff = (gen_ratios - ref_ratios).abs() * combined_vis
    l1 = weighted_diff.sum(dim=-1) / combined_vis.sum(dim=-1).clamp(min=1e-6)

    high_ref = (ref_vis >= 0.5).to(gen_ratios.dtype)
    ref_high_count = high_ref.sum(dim=-1).clamp(min=1.0)
    if gen_confidence is None:
        gen_confidence = gen_vis
    if gen_confidence.shape != gen_ratios.shape:
        raise ValueError(
            "gen_confidence must have the same shape as gen_ratios; "
            f"got {tuple(gen_confidence.shape)} vs {tuple(gen_ratios.shape)}"
        )

    temperature = max(float(confidence_temperature), 1e-6)
    threshold = torch.as_tensor(
        float(vis_threshold),
        dtype=gen_confidence.dtype,
        device=gen_confidence.device,
    )
    # Soft hinge is zero at the threshold and positive below it.
    soft_hinge = F.softplus((threshold - gen_confidence) / temperature)
    soft_hinge = temperature * (soft_hinge - math.log(2.0))
    shortfall = soft_hinge.clamp_min(0.0)
    missing_penalty = (shortfall * high_ref).sum(dim=-1) / ref_high_count
    missing_fraction = (
        (gen_confidence.detach() < float(vis_threshold)).to(gen_ratios.dtype)
        * high_ref
    ).sum(dim=-1) / ref_high_count

    loss_per_sample = l1 + missing_penalty
    return loss_per_sample, missing_fraction.detach()
