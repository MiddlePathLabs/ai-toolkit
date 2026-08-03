"""Body-proportion anchor loss: visibility-weighted L1 of bone-length ratios
plus a missing-keypoint penalty.

The frozen ViTPose perceptor runs on the live x0-decoded pixels and its 8 (or
10 with head) pose-invariant ratios are matched against the cached GT ratios.
Pure tensor math; imports no model weights. Body-proportion does NOT
participate in the diffusion/depth ``loss_split`` alternation -- it fires every
step within its timestep window.
"""
from __future__ import annotations

from typing import Tuple

import torch


def compute_body_proportion_loss(
    gen_ratios: torch.Tensor,
    gen_vis: torch.Tensor,
    ref_ratios: torch.Tensor,
    ref_vis: torch.Tensor,
    vis_threshold: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Body-proportion matching loss for a batch.

    Args:
        gen_ratios: (B, N) generator ratios (gradient flows).
        gen_vis: (B, N) generator ratio visibility weights.
        ref_ratios: (B, N) cached GT ratios.
        ref_vis: (B, N) cached GT ratio visibility weights.
        vis_threshold: below this a predicted ratio counts as "missing".

    Returns:
        ``(loss_per_sample, missing_fraction)`` -- both ``(B,)``; the first
        carries the gradient. The caller applies timestep weighting, per-sample
        loss weights, the valid mask, and reduction.

    The loss is the visibility-weighted L1 of the ratios (a ratio contributes
    only when BOTH ref and gen consider it visible) plus a missing-keypoint
    penalty: the fraction of high-confidence ref ratios the prediction dropped
    below ``vis_threshold``. This penalizes the model for losing body parts it
    was shown, not just for ratio drift.
    """
    combined_vis = torch.min(ref_vis, gen_vis)
    weighted_diff = (gen_ratios - ref_ratios).abs() * combined_vis
    l1 = weighted_diff.sum(dim=-1) / combined_vis.sum(dim=-1).clamp(min=1e-6)  # (B,)

    missing_mask = (ref_vis >= 0.5) & (gen_vis < vis_threshold)
    missing_count = missing_mask.float().sum(dim=-1)
    ref_high_count = (ref_vis >= 0.5).float().sum(dim=-1).clamp(min=1.0)
    missing_fraction = missing_count / ref_high_count  # (B,) in [0, 1]

    loss_per_sample = l1 + missing_fraction
    return loss_per_sample, missing_fraction.detach()
