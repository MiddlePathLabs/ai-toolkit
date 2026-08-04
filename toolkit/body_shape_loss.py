"""Body-shape anchor loss: L1 on 10-dim SMPL betas with a cosine gate.

The frozen HybrIK encoder regresses 10-dim SMPL betas from the live x0-decoded
pixels and they are matched against cached GT betas. Pure tensor math; imports
no model weights. Body-shape does NOT participate in diffusion/depth
``loss_split``.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def compute_body_shape_loss(
    gen_betas: torch.Tensor,
    ref_betas: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-sample L1 over the 10 SMPL betas + cosine similarity (for gating/logging).

    Args:
        gen_betas: (B, 10) generator betas (gradient flows).
        ref_betas: (B, 10) cached GT betas.
    Returns:
        ``(l1_per_sample, cos_per_sample)`` -- both ``(B,)``; l1 carries the
        gradient, cos is detached. The caller applies the timestep weight, the
        min-cos gate, per-sample loss weights, and reduction.
    """
    l1 = (gen_betas - ref_betas).abs().mean(dim=-1)  # (B,)
    cos = F.cosine_similarity(gen_betas, ref_betas, dim=-1).detach()  # (B,)
    return l1, cos
