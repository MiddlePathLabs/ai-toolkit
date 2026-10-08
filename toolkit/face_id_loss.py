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


# Sentinel: the training crop for this item cannot be reconstructed (non-bucket
# random crop), so the caller should fall back to the full decoded frame.
UNKNOWN_FRAME = "unknown"


def map_face_bbox_to_training_frame(norm_bbox, file_item):
    """Map a cached face bbox into the training frame, as fractions of it.

    ``norm_bbox`` is ``[x1, y1, x2, y2]`` normalized to the raw (exif-
    transposed) source image, as stored by ``cache_face_identity``. The
    dataloader flips, scales and crops that image before encoding, so the
    face sits elsewhere in the latent the trainer decodes. This replays the
    same deterministic geometry as ``load_and_process_image``:

      * bucketed: flip -> resize to ``scale_to_*`` -> crop ``crop_*``
      * non-bucket: flip -> uniform scale -> center square crop -> resize

    Returns ``[fx1, fy1, fx2, fy2]`` in ``[0, 1]`` of the training frame
    (multiply by the decoded width/height), ``None`` when there is no face
    or the face lies entirely outside the crop, or ``UNKNOWN_FRAME`` for a
    non-bucket random crop whose offset is not recorded.
    """
    if norm_bbox is None:
        return None
    x1, y1, x2, y2 = (float(v) for v in norm_bbox)
    if abs(x1) + abs(y1) + abs(x2) + abs(y2) == 0.0:
        return None

    if getattr(file_item, "flip_x", False):
        x1, x2 = 1.0 - x2, 1.0 - x1
    if getattr(file_item, "flip_y", False):
        y1, y2 = 1.0 - y2, 1.0 - y1

    config = getattr(file_item, "dataset_config", None)
    bucketed = bool(getattr(config, "buckets", True)) if config is not None else True
    stw = getattr(file_item, "scale_to_width", None)
    sth = getattr(file_item, "scale_to_height", None)
    cx = getattr(file_item, "crop_x", None)
    cy = getattr(file_item, "crop_y", None)
    cw = getattr(file_item, "crop_width", None)
    ch = getattr(file_item, "crop_height", None)

    if bucketed and None not in (stw, sth, cx, cy, cw, ch) and cw and ch:
        fx1 = (x1 * float(stw) - float(cx)) / float(cw)
        fx2 = (x2 * float(stw) - float(cx)) / float(cw)
        fy1 = (y1 * float(sth) - float(cy)) / float(ch)
        fy2 = (y2 * float(sth) - float(cy)) / float(ch)
    elif not bucketed and config is not None and getattr(config, "random_crop", False):
        return UNKNOWN_FRAME
    elif not bucketed:
        w = float(getattr(file_item, "width", 0) or 0)
        h = float(getattr(file_item, "height", 0) or 0)
        if w <= 0 or h <= 0:
            return UNKNOWN_FRAME
        side = min(w, h)
        off_x = (w - side) / 2.0
        off_y = (h - side) / 2.0
        fx1 = (x1 * w - off_x) / side
        fx2 = (x2 * w - off_x) / side
        fy1 = (y1 * h - off_y) / side
        fy2 = (y2 * h - off_y) / side
    else:
        # Bucketed item without recorded geometry: the image is used as-is.
        fx1, fy1, fx2, fy2 = x1, y1, x2, y2

    if fx2 <= 0.0 or fy2 <= 0.0 or fx1 >= 1.0 or fy1 >= 1.0:
        return None
    fx1, fy1 = max(0.0, fx1), max(0.0, fy1)
    fx2, fy2 = min(1.0, fx2), min(1.0, fy2)
    if fx2 <= fx1 or fy2 <= fy1:
        return None
    return [fx1, fy1, fx2, fy2]


# ---------------------------------------------------------------------------
# Reference modes (perceptual-fork parity): average / blend / random / multi-ref
# ---------------------------------------------------------------------------

def identity_dataset_key(dataset_config):
    """Key that groups file items into one identity reference pool (per dataset)."""
    return (
        getattr(dataset_config, "folder_path", None)
        or getattr(dataset_config, "dataset_path", None)
        or id(dataset_config)
    )


def identity_reference_modes_active(cfg) -> bool:
    return bool(
        getattr(cfg, "identity_loss_use_average", False)
        or float(getattr(cfg, "identity_loss_average_blend", 0.0) or 0.0) > 0.0
        or getattr(cfg, "identity_loss_use_random", False)
        or int(getattr(cfg, "identity_loss_num_refs", 0) or 0) > 0
    )


def build_identity_reference_tables(embeds_by_key, use_average: bool):
    """Per-dataset ``(averages, pools)`` from cached non-zero embeddings.

    The average is the mean of the L2-normalized embeddings, renormalized.
    With ``use_average`` the fork replaced every embedding with the average
    before building its random pools, so pools then hold only the average.
    """
    averages, pools = {}, {}
    for key, embeds in embeds_by_key.items():
        valid = [e.float() for e in embeds if e is not None and e.abs().sum() > 0]
        if not valid:
            continue
        stacked = torch.stack(valid)
        avg = stacked.mean(dim=0)
        avg = avg / (avg.norm() + 1e-8)
        averages[key] = avg
        pools[key] = avg.unsqueeze(0) if use_average else stacked
    return averages, pools


def identity_reference_cosine(
    gen_emb: torch.Tensor,
    ref_emb: torch.Tensor,
    keys,
    cfg,
    averages,
    pools,
    mean_emb: Optional[torch.Tensor] = None,
    rng=None,
):
    """Bias-corrected cosine against the configured reference(s).

    Mirrors the perceptual fork's order: average (with per-image clean-cos
    targets) or blend, then random replacement, then best-of-N multi-ref.

    Returns ``(cos_sim (V,), clean_cos (V,) or None)``. ``clean_cos`` is set
    only in average mode: each image's clean similarity to its dataset average
    (floored at 0.1), the target for ``max(0, 1 - cos / clean_cos)``.
    """
    import random as _random

    rng = rng or _random
    ref = ref_emb.clone()
    clean_cos = None
    if getattr(cfg, "identity_loss_use_average", False):
        clean = torch.ones(ref.shape[0], device=ref.device, dtype=ref.dtype)
        for j, key in enumerate(keys):
            avg = averages.get(key)
            if avg is None:
                continue
            avg = avg.to(ref.device, dtype=ref.dtype)
            clean[j] = bias_corrected_cosine(ref_emb[j:j + 1], avg.unsqueeze(0), mean_emb).clamp(min=0.1)[0]
            ref[j] = avg
        clean_cos = clean
    else:
        blend = float(getattr(cfg, "identity_loss_average_blend", 0.0) or 0.0)
        if blend > 0.0:
            for j, key in enumerate(keys):
                avg = averages.get(key)
                if avg is None:
                    continue
                mixed = (1.0 - blend) * ref[j] + blend * avg.to(ref.device, dtype=ref.dtype)
                ref[j] = mixed / (mixed.norm() + 1e-8)

    def _random_refs(base):
        out = base.clone()
        for j, key in enumerate(keys):
            pool = pools.get(key)
            if pool is not None and pool.shape[0] > 0:
                out[j] = pool[rng.randint(0, pool.shape[0] - 1)].to(out.device, dtype=out.dtype)
        return out

    if getattr(cfg, "identity_loss_use_random", False) and pools:
        ref = _random_refs(ref)

    cos_sim = bias_corrected_cosine(gen_emb, ref, mean_emb)
    num_refs = int(getattr(cfg, "identity_loss_num_refs", 0) or 0)
    if num_refs > 0 and pools:
        all_cos = [cos_sim]
        for _ in range(num_refs - 1):
            all_cos.append(bias_corrected_cosine(gen_emb, _random_refs(ref), mean_emb))
        cos_sim = torch.stack(all_cos).max(dim=0).values
    return cos_sim, clean_cos
