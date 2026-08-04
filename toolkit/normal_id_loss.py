"""Normal-anchor loss math + preview rendering.

Cosine dissimilarity + L1 per-pixel surface-normal matching (mirrors Sapiens'
own training loss). The frozen Sapiens perceptor runs on the live x0-decoded
pixels and is matched against the cached GT normal map. Pure tensor math plus
a PIL preview renderer; imports no model weights itself.

Normal loss does NOT participate in the diffusion/depth ``loss_split``
alternation -- it fires every step its timestep window is active.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def compute_normal_loss(
    encoder,
    x0_pixels: torch.Tensor,
    gt_normals: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Surface-normal matching loss for a batch.

    Args:
        encoder: frozen Sapiens perceptor (``DifferentiableNormalEncoder``).
        x0_pixels: ``(B, 3, H, W)`` generator output in ``[0, 1]``.
        gt_normals: ``(B, 3, Hg, Wg)`` cached GT unit normals (any resolution;
            resized to the perceptor output grid). All-zero maps mark an item
            with no usable GT (filtered by the caller's valid mask).
        mask: optional ``(B, Hm, Wm)`` spatial weight in ``[0, 1]``. When given,
            the per-pixel cosine/L1 are weighted by it before a per-sample
            *normalized* spatial average (``sum(p*mask)/sum(mask)``), so a
            uniform mask reproduces the plain mean and a body-restriction mask
            focuses the loss on the body region.

    Returns:
        ``(cos_loss, l1_loss, gen_detached, ref_detached)`` -- ``cos_loss`` and
        ``l1_loss`` are per-sample ``(B,)`` tensors that carry the gradient;
        the detached tensors are for logging / preview rendering. The caller
        combines ``cos_loss + l1_loss`` and applies per-sample weights + the
        valid mask + reduction.
    """
    gen = encoder(x0_pixels)  # (B, 3, No, No) unit normals, gradient flows

    ref = gt_normals
    if ref.dim() == 3:
        ref = ref.unsqueeze(0)
    if ref.shape[-2:] != gen.shape[-2:]:
        ref = F.interpolate(
            ref.float(), size=gen.shape[-2:], mode="bilinear", align_corners=False
        )
    ref = ref.to(gen.device, dtype=gen.dtype)

    cos_per_pixel = (ref * gen).sum(dim=1)  # (B, H, W)
    l1_per_pixel = (ref - gen).abs().mean(dim=1)  # (B, H, W)

    if mask is not None:
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.shape[-2:] != cos_per_pixel.shape[-2:]:
            mask = F.interpolate(
                mask.unsqueeze(1).float(), size=cos_per_pixel.shape[-2:],
                mode="nearest",
            ).squeeze(1)
        mask = mask.to(cos_per_pixel.device, dtype=cos_per_pixel.dtype)
        denom = mask.sum(dim=(1, 2)).clamp(min=1e-6)
        cos_mean = (cos_per_pixel * mask).sum(dim=(1, 2)) / denom
        l1_mean = (l1_per_pixel * mask).sum(dim=(1, 2)) / denom
    else:
        cos_mean = cos_per_pixel.mean(dim=(1, 2))  # (B,)
        l1_mean = l1_per_pixel.mean(dim=(1, 2))  # (B,)

    cos_loss = 1.0 - cos_mean  # (B,)
    return cos_loss, l1_mean, gen.detach(), ref.detach()


def render_normal_preview(pred_pil, ref_pil, gen_normals, ref_normals):
    """Render a four-panel ``[GT RGB | GT normal | Pred RGB | Pred normal]`` strip.

    Normal maps are mapped to RGB via ``(n + 1) * 0.5`` ([-1,1] -> [0,1]).
    """
    from PIL import Image, ImageDraw

    def _normal_to_pil(nrm: torch.Tensor, size):
        n = nrm.detach().float().cpu()
        if n.dim() == 4:
            n = n[0]
        rgb = ((n + 1.0) * 0.5).clamp(0, 1)
        im = Image.fromarray((rgb.permute(1, 2, 0).numpy() * 255).astype("uint8"))
        return im.resize(size, Image.BICUBIC)

    W, H = pred_pil.size
    ref_pil = ref_pil.resize((W, H), Image.BICUBIC)
    gt_nrm_pil = _normal_to_pil(ref_normals, (W, H))
    pred_nrm_pil = _normal_to_pil(gen_normals, (W, H))

    panels = [
        ("GT RGB", ref_pil),
        ("GT normal", gt_nrm_pil.convert("RGB")),
        ("Pred RGB", pred_pil),
        ("Pred normal", pred_nrm_pil.convert("RGB")),
    ]

    combo = Image.new("RGB", (W * len(panels), H), (0, 0, 0))
    for i, (_, panel_pil) in enumerate(panels):
        combo.paste(panel_pil, (W * i, 0))

    draw = ImageDraw.Draw(combo)
    for i, (label, _) in enumerate(panels):
        draw.text((W * i + 4, 4), label, fill=(255, 255, 0))

    return combo
