"""Depth-Anything-V2 perceptor wrapper + GT-depth caching pass (Task 5a).

License notice: the Depth-Anything-V2 HuggingFace weights
(depth-anything/Depth-Anything-V2-{Small,Base,Large}-hf) are released under
CC-BY-NC-4.0 (non-commercial use only). The perceptor downloads lazily: the
``DepthAnythingForDepthEstimation`` import lives inside
``DifferentiableDepthEncoder.__init__``, so simply selecting the Krea 2
architecture never pulls DA2 -- only enabling depth (loss_weight > 0) does.

This module is intentionally separate from ``toolkit.depth_consistency``
(Task 2's fingerprint helpers) so the cache-identity contract is not
destabilized by the perceptor / caching implementation. The fingerprint and
safetensors-key helpers are imported from there.

The scale-and-shift-invariant / multi-scale-gradient loss math
(compute_depth_consistency_loss) and the live decode dispatch belong to the
depth loss block (Task 5b) and are not ported here.
"""
from __future__ import annotations

import os
import tempfile
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

from toolkit.depth_consistency import build_depth_cache_fingerprint, depth_cache_key

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DifferentiableDepthEncoder(nn.Module):
    """Frozen Depth-Anything-V2 perceptor with a pure-tensor preprocessor.

    Inputs: ``(B, 3, H, W)`` float tensor in ``[0, 1]`` or ``[-1, 1]``
    (auto-detected). Gradients flow through preprocessing and the full DA2
    forward, so the same instance serves the live depth loss (Task 5b). The HF
    ``DPTImageProcessor`` is intentionally bypassed: it round-trips through
    PIL + numpy and detaches the computation graph.
    """

    def __init__(
        self,
        model_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
        input_size: int = 518,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        grad_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        from transformers import DepthAnythingForDepthEstimation  # lazy import

        self.model = DepthAnythingForDepthEstimation.from_pretrained(
            model_id, torch_dtype=dtype
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        if grad_checkpoint:
            try:
                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                self.model.gradient_checkpointing_enable()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.input_size = input_size
        if device is not None:
            self.to(device)

    def _aspect_preserving_hw(self, h: int, w: int):
        if h >= w:
            new_h = self.input_size
            new_w = max(14, int(round(w * self.input_size / h / 14)) * 14)
        else:
            new_w = self.input_size
            new_h = max(14, int(round(h * self.input_size / w / 14)) * 14)
        return new_h, new_w

    def preprocess(self, pixels: torch.Tensor) -> torch.Tensor:
        if float(pixels.min().item()) < -0.05:
            pixels = (pixels + 1.0) * 0.5
        pixels = pixels.clamp(0.0, 1.0)
        _, _, h, w = pixels.shape
        new_h, new_w = self._aspect_preserving_hw(h, w)
        x = F.interpolate(
            pixels,
            size=(new_h, new_w),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        x = x.to(self.mean.dtype)
        x = (x - self.mean) / self.std
        return x.to(next(self.model.parameters()).dtype)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """Return ``(B, Hd, Wd)`` float32 depth. Gradients flow if input has grad."""
        x = self.preprocess(pixels)
        out = self.model(pixel_values=x)
        return out.predicted_depth.float()


def gaussian_blur_2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Differentiable depthwise Gaussian blur for ``(B, C, H, W)`` tensors.

    Identity passthrough when ``sigma <= 0``. Kernel size is
    ``2*ceil(3*sigma) + 1``; padding is ``reflect``. Built in fp32 and cast to
    the input dtype so bf16/fp16 callers stay numerically safe. Matches the
    live depth-loss blur so pred-depth (blurred) and GT-depth (blurred) align.
    """
    if sigma is None or sigma <= 0:
        return x
    import math

    radius = max(1, int(math.ceil(3.0 * float(sigma))))
    k = 2 * radius + 1
    coords = torch.arange(k, device=x.device, dtype=torch.float32) - radius
    g = torch.exp(-(coords ** 2) / (2.0 * float(sigma) * float(sigma)))
    g = g / g.sum()
    kernel_1d = g.view(1, 1, 1, k)
    kernel_2d = (kernel_1d * kernel_1d.transpose(-1, -2)).to(x.dtype)
    c = x.shape[1]
    kernel = kernel_2d.expand(c, 1, k, k).contiguous()
    x_padded = F.pad(x, (radius, radius, radius, radius), mode="reflect")
    return F.conv2d(x_padded, kernel, groups=c)


def _apply_dataloader_transform(img, file_item):
    """PIL -> PIL: mirror the dataloader's deterministic bucket transform.

    Applies recorded flips + bucket resize + crop when bucket params are
    attached to ``file_item`` (the same values that feed the cache fingerprint,
    so the cached GT depth is taken from exactly the pixels the trainer sees).
    Falls back to the input image unchanged when params are missing.
    """
    from PIL import Image as _PILImage

    if getattr(file_item, "flip_x", False):
        img = img.transpose(_PILImage.FLIP_LEFT_RIGHT)
    if getattr(file_item, "flip_y", False):
        img = img.transpose(_PILImage.FLIP_TOP_BOTTOM)
    stw = getattr(file_item, "scale_to_width", None)
    sth = getattr(file_item, "scale_to_height", None)
    cx = getattr(file_item, "crop_x", None)
    cy = getattr(file_item, "crop_y", None)
    cw = getattr(file_item, "crop_width", None)
    ch = getattr(file_item, "crop_height", None)
    if None in (stw, sth, cx, cy, cw, ch):
        return img
    img = img.resize((int(stw), int(sth)), _PILImage.BICUBIC)
    img = img.crop((int(cx), int(cy), int(cx) + int(cw), int(cy) + int(ch)))
    return img


def _depth_cache_path(file_item, fingerprint: str) -> str:
    """Locate the safetensors cache file for one (image, transform) pair.

    Follows the existing latent-cache convention (image-dir-local ``_latent_cache``
    subdir), keyed by the fingerprint so each bucket transform gets its own file.
    """
    img_dir = os.path.dirname(file_item.path)
    cache_dir = os.path.join(img_dir, "_latent_cache")
    filename_no_ext = os.path.splitext(os.path.basename(file_item.path))[0]
    return os.path.join(cache_dir, f"{filename_no_ext}_{fingerprint}.safetensors")


def _depth_cache_hit(cache_path: str, key: str) -> bool:
    """Header-only hit check: True when the file exists and ``key`` is present.

    Reads only the safetensors header (a small JSON blob), never the tensor
    bytes, so the up-front pass stays cheap. Returns False on a corrupt or
    unreadable file so the caller recomputes.
    """
    if not os.path.exists(cache_path):
        return False
    try:
        with safe_open(cache_path, framework="pt", device="cpu") as f:
            return key in f.keys()
    except Exception:  # noqa: BLE001 -- corrupt/unreadable cache -> recompute
        return False


def _atomic_save_file(save_data: dict, cache_path: str) -> None:
    """Windows-safe atomic safetensors write (temp file + ``os.replace``).

    safetensors.save_file uses memory-mapped I/O on Windows. Rewriting a file
    that was read earlier in the same process can hit OSError 1224
    ("user-mapped section open"). Writing to a temp file in the same directory
    and renaming sidesteps that.
    """
    cache_dir = os.path.dirname(cache_path) or "."
    os.makedirs(cache_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".safetensors", dir=cache_dir)
    os.close(fd)
    try:
        save_file(save_data, tmp_path)
        os.replace(tmp_path, cache_path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def cache_depth_gt(
    file_items: List,
    config,
    *,
    encoder: DifferentiableDepthEncoder,
    arch: str,
    vae_id: str,
    device: Optional[torch.device] = None,
    roundtrip_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> None:
    """Stamp every file item with its depth-cache metadata and persist GT depth.

    For each item: compute the fingerprint (Task 2 helper, supplied ``arch`` and
    ``vae_id``), derive the cache path + ``depth_gt_{fingerprint}`` key, and
    stamp them onto the item -- the exact attributes Task 3's
    ``DepthCachingFileItemDTOMixin`` reads. On a cache miss, run the VAE
    round-trip callback (Krea: encode_images/decode_latents; generic:
    vae.encode/scaling_factor) -> optional pre-DA2 blur -> frozen DA2 forward ->
    write the safetensors. On a header hit, skip the recompute.

    The resident tensor is never held on the shared file-list item; the worker
    re-reads lazily via ``get_depth_gt()``.
    """
    from PIL import Image
    from PIL.ImageOps import exif_transpose
    import numpy as np

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    blur_sigma = float(getattr(config, "pixel_blur_sigma", 0.0) or 0.0)
    hits = 0
    misses = 0
    zero_depth = 0

    for file_item in tqdm(file_items, desc="Caching GT depth maps"):
        fingerprint = build_depth_cache_fingerprint(
            file_item, config, arch=arch, vae_id=vae_id
        )
        key = depth_cache_key(fingerprint)
        cache_path = _depth_cache_path(file_item, fingerprint)

        # stamp for the worker (hit or miss -- both point at the same file)
        file_item._depth_cache_path = cache_path
        file_item._depth_cache_key = key
        file_item.is_depth_cached = True
        file_item.depth_gt = None

        if _depth_cache_hit(cache_path, key):
            hits += 1
            continue

        misses += 1
        raw_pil = exif_transpose(Image.open(file_item.path)).convert("RGB")
        pil_image = _apply_dataloader_transform(raw_pil, file_item)
        arr = torch.from_numpy(
            np.asarray(pil_image, dtype=np.float32) / 255.0
        ).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            if roundtrip_fn is not None:
                arr = roundtrip_fn(arr)
            if blur_sigma > 0:
                arr = gaussian_blur_2d(arr, blur_sigma)
            depth = encoder(arr)[0].cpu().to(torch.float16)

        if float(depth.abs().sum().item()) < 1e-6:
            zero_depth += 1

        _atomic_save_file({key: depth}, cache_path)

    if hits or misses:
        print(
            f"  - GT depth cache: {hits} reused, {misses} computed "
            f"(key {key!r}). An all-computed run means a transform/perceptor/"
            f"arch/vae mismatch or a missing cache file."
        )
    if zero_depth > 0:
        print(f"  - Warning: zero depth for {zero_depth}/{len(file_items)} images")
