"""Face-identity anchor loss: bias-corrected ArcFace cosine similarity.

ArcFace embeddings have an inherent cluster bias: even non-face / pure-noise
inputs score ~0.5 against a real face reference. Subtracting the mean
embedding of ~200 noise images and renormalizing collapses that bias so
non-face inputs score ~0 and only genuine identity similarity remains. The
live loss decodes x0 through the VAE under gradient, crops the face, runs
ArcFace, and matches against the cached GT embedding.

Pure tensor math; imports no model weights. Identity does NOT participate in
the diffusion/depth ``loss_split`` alternation.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def bias_corrected_cosine(
    gen_emb: torch.Tensor,
    ref_emb: torch.Tensor,
    mean_emb: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Bias-corrected cosine similarity between generated and reference embeddings.

    Args:
        gen_emb: (B, 512) L2-normalized generated embeddings (gradient flows).
        ref_emb: (B, 512) L2-normalized reference embeddings.
        mean_emb: (512,) optional noise-mean bias direction. When provided, both
            embeddings are centered on it and renormalized before the cosine, so
            the ArcFace cluster bias (~0.5 for non-faces) collapses toward 0.
    Returns:
        (B,) cosine similarity in [-1, 1] (gradient flows through gen_emb).
    """
    if mean_emb is not None:
        m = mean_emb.to(gen_emb.device, dtype=gen_emb.dtype).unsqueeze(0)
        gen_c = F.normalize(gen_emb - m, p=2, dim=-1)
        ref_c = F.normalize(ref_emb - m, p=2, dim=-1)
    else:
        gen_c = gen_emb
        ref_c = ref_emb
    return F.cosine_similarity(gen_c, ref_c, dim=-1)


def compute_identity_loss(
    gen_emb: torch.Tensor,
    ref_emb: torch.Tensor,
    mean_emb: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-sample identity loss = ``1 - bias_corrected_cosine``. Returns (B,).

    The caller applies the timestep weight, the face-presence / SCRFD quality
    gate, the ``min_cos`` floor, per-sample loss weights, and reduction.
    """
    cos = bias_corrected_cosine(gen_emb, ref_emb, mean_emb)
    return 1.0 - cos
