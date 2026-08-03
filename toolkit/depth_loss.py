"""Depth-anchor loss math + preview rendering (Task 5b).

Scale-and-shift-invariant L1 and multi-scale gradient matching (MiDaS /
Ranftl et al.), plus the combined depth-consistency loss that runs the frozen
DA2 perceptor on the live x0-decoded pixels and matches it against the cached
GT depth. Ported by contract from the perceptual fork's depth_consistency.py.

License notice: the Depth-Anything-V2 HuggingFace weights consumed by the
perceptor argument are CC-BY-NC-4.0 (non-commercial). The weight download is
lazy and lives in toolkit.depth_perceptor; this module holds pure tensor math
plus a PIL preview renderer and imports no model weights itself.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def ssi_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scale-and-shift-invariant L1 (MiDaS / Ranftl et al.).

    Solves ``min_{s,t} ||s*pred + t - target||_2`` in closed form per-sample
    (differentiable in ``pred``), then returns L1 between the aligned pred and
    the target. Returns ``(loss, scale, shift)``; scale/shift are detached.
    """
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    if mask is None:
        mask = torch.ones_like(pred)
    elif mask.dim() == 2:
        mask = mask.unsqueeze(0)
    p = pred.flatten(1)
    g = target.flatten(1)
    m = mask.flatten(1).float()
    n = m.sum(dim=1).clamp_min(1.0)
    mean_p = (p * m).sum(1) / n
    mean_g = (g * m).sum(1) / n
    var_p = (p * p * m).sum(1) / n - mean_p * mean_p
    cov_pg = (p * g * m).sum(1) / n - mean_p * mean_g
    s = cov_pg / var_p.clamp_min(1e-6)
    t = mean_g - s * mean_p
    aligned = s.view(-1, 1, 1) * pred + t.view(-1, 1, 1)
    diff = (aligned - target).abs() * mask
    loss = diff.sum() / mask.sum().clamp_min(1.0)
    return loss, s.detach(), t.detach()


def multiscale_grad_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scales: int = 4,
) -> torch.Tensor:
    """Multi-scale L1 gradient-matching loss (MiDaS)."""
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    if mask is None:
        mask = torch.ones_like(pred)
    elif mask.dim() == 2:
        mask = mask.unsqueeze(0)
    loss = pred.new_zeros(())
    p, g, m = pred, target, mask.float()
    for k in range(scales):
        if k > 0:
            p = F.avg_pool2d(p.unsqueeze(1), 2).squeeze(1)
            g = F.avg_pool2d(g.unsqueeze(1), 2).squeeze(1)
            m = F.avg_pool2d(m.unsqueeze(1), 2).squeeze(1)
        diff = p - g
        mx = m[:, :, 1:] * m[:, :, :-1]
        my = m[:, 1:, :] * m[:, :-1, :]
        dx = (diff[:, :, 1:] - diff[:, :, :-1]).abs() * mx
        dy = (diff[:, 1:, :] - diff[:, :-1, :]).abs() * my
        loss = loss + (dx.sum() / mx.sum().clamp_min(1.0)) + (
            dy.sum() / my.sum().clamp_min(1.0)
        )
    return loss / scales


def compute_depth_consistency_loss(
    encoder,
    x0_pixels: torch.Tensor,
    gt_depth: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    ssi_weight: float = 1.0,
    grad_weight: float = 0.5,
    grad_scales: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full depth-consistency loss for one sample or a batch.

    Args:
        encoder: frozen DA2 perceptor (``DifferentiableDepthEncoder``).
        x0_pixels: ``(B, 3, H, W)`` generator output in ``[0, 1]``.
        gt_depth: ``(B, Hd_gt, Wd_gt)`` cached GT depth (any resolution).
        mask: ``(B, Hm, Wm)`` optional spatial weight in ``[0, 1]``; if None,
            full image.
        ssi_weight, grad_weight, grad_scales: loss composition.

    Returns:
        ``(loss, ssi_component, grad_component, d_pred_detached,
        target_detached)`` -- the first carries the gradient; the rest are
        detached for logging / preview rendering.
    """
    d_pred = encoder(x0_pixels)  # (B, Hd, Wd) fp32, gradient flows

    # Resize GT depth and mask to match the pred grid.
    target = gt_depth
    if target.dim() == 2:
        target = target.unsqueeze(0)
    if target.shape[-2:] != d_pred.shape[-2:]:
        target = F.interpolate(
            target.unsqueeze(1).float(),
            size=d_pred.shape[-2:],
            mode="bilinear",
            align_corners=True,
        ).squeeze(1)
    target = target.to(d_pred.device, dtype=d_pred.dtype)

    if mask is not None:
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.shape[-2:] != d_pred.shape[-2:]:
            mask = F.interpolate(
                mask.unsqueeze(1).float(),
                size=d_pred.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        mask = mask.to(d_pred.device, dtype=d_pred.dtype)

    ssi, s_align, t_align = ssi_l1(d_pred, target, mask)
    # Align d_pred to target's scale before grad-matching so per-sample DA2
    # output magnitude does not leak into |grad d_pred - grad d_gt|. s, t come
    # back detached from ssi_l1, so the gradient w.r.t. d_pred reduces to a
    # constant scalar -- alignment-fit noise stays out of the backward path.
    d_pred_aligned = s_align.view(-1, 1, 1) * d_pred + t_align.view(-1, 1, 1)
    grd = multiscale_grad_loss(d_pred_aligned, target, mask, scales=grad_scales)
    loss = ssi_weight * ssi + grad_weight * grd
    return loss, ssi.detach(), grd.detach(), d_pred.detach(), target.detach()


def render_depth_preview(pred_pil, ref_pil, d_pred, d_gt, mask=None):
    """Render a four-panel ``[GT RGB | GT depth | Pred RGB | Pred depth]`` strip.

    Depth maps are percentile-normalized (p2-p98) to grayscale, then color
    inverted so nearer surfaces appear brighter. With a mask a fifth ``Mask``
    panel is appended (white = included). Phase 2 ships mask_source 'none', so
    the mask panel is not produced on the live path.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    def _depth_to_pil(dep: torch.Tensor, size):
        d = dep.detach().float().cpu().numpy()
        if d.ndim == 3:
            d = d[0]
        lo, hi = np.percentile(d, 2), np.percentile(d, 98)
        dn = np.clip((d - lo) / max(1e-6, (hi - lo)), 0, 1)
        im = Image.fromarray((dn * 255).astype(np.uint8))
        return im.resize(size, Image.BICUBIC)

    W, H = pred_pil.size
    ref_pil = ref_pil.resize((W, H), Image.BICUBIC)
    gt_pil = _depth_to_pil(d_gt, (W, H))
    pred_depth_pil = _depth_to_pil(d_pred, (W, H))

    panels = [
        ("GT RGB", ref_pil),
        ("GT depth", gt_pil.convert("RGB")),
        ("Pred RGB", pred_pil),
        ("Pred depth", pred_depth_pil.convert("RGB")),
    ]

    if mask is not None:
        m = mask.detach().float().cpu().numpy()
        if m.ndim == 3:
            m = m[0]
        m = np.clip(m, 0.0, 1.0)
        mask_pil = (
            Image.fromarray((m * 255).astype(np.uint8))
            .resize((W, H), Image.BICUBIC)
            .convert("RGB")
        )
        panels.append(("Mask", mask_pil))

    combo = Image.new("RGB", (W * len(panels), H), (0, 0, 0))
    for i, (_, panel_pil) in enumerate(panels):
        combo.paste(panel_pil, (W * i, 0))

    draw = ImageDraw.Draw(combo)
    for i, (label, _) in enumerate(panels):
        draw.text((W * i + 4, 4), label, fill=(255, 255, 0))

    return combo
