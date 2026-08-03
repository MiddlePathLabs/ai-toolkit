"""ViTPose body-proportion perceptor + GT-ratio caching.

A frozen ViTPose-Plus-Base pose estimator produces 17 COCO keypoints; from
those we compute 8 pose-invariant bone-length ratios (10 with head) that
characterize a person's proportions independent of pose. Gradients flow
through a differentiable affine warp (replicating the HF processor) into the
ViTPose forward and a vendored soft-argmax heatmap decode, so the same
instance serves the live body-proportion loss.

Only the body-PROPORTION auxiliary loss is ported here. The source module
also contained an SMPL body-shape CONDITIONING feature (HMR2 /
BodyIDExtractor / BodyIDProjector) and video/5D paths -- both intentionally
excluded. The MediaPipe person detector is also dropped: it was used only for
presence detection, but ViTPose is always run on the full image regardless, so
the encoder naturally returns a zero vector (filtered by the loss) when no
body is present. This removes the onnxruntime dependency.

The ``dsntnn`` soft-argmax is vendored inline (MIT, attributed) rather than
added as a dependency -- it is ~10 lines of standard coordinate-expectation
math. Loss math + preview live in ``toolkit.body_proportion_loss``.
"""
from __future__ import annotations

import os
import tempfile
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

NUM_BODY_RATIOS = 8
NUM_HEAD_RATIOS = 2
# Cache version key encodes include_head so changing it invalidates the cache.
CACHE_VERSION_KEY_BODY = "body_proportion_v3"
CACHE_VERSION_KEY_HEAD = "body_proportion_v4_head"


# ---------------------------------------------------------------------------
# Vendored soft-argmax (DSNT). Originally from the `dsntnn` package
# (https://github.com/anibali/dsntnn, MIT license). ~10 lines of coordinate
# expectation; vendored to avoid a single-purpose pip dependency.
# ---------------------------------------------------------------------------

def _normalized_linspace(length, dtype, device):
    # Cell-centered linspace from -1 to 1 (length=4 -> [-0.75, -0.25, 0.25, 0.75]).
    first = -(length - 1.0) / length
    return torch.arange(length, dtype=dtype, device=device) * (2.0 / length) + first


def _soft_argmax_2d(heatmaps: torch.Tensor) -> torch.Tensor:
    """Differentiable spatial-to-numerical transform for ``(B, K, H, W)`` maps.

    Expects heatmaps already normalized to a spatial distribution (each keypoint
    map sums to 1 over HxW). Returns ``(B, K, 2)`` coordinates in ``[-1, 1]``,
    ``(x, y)`` order.
    """
    B, K, H, W = heatmaps.shape
    ys = _normalized_linspace(H, heatmaps.dtype, heatmaps.device).view(1, 1, H, 1)
    xs = _normalized_linspace(W, heatmaps.dtype, heatmaps.device).view(1, 1, 1, W)
    cy = (heatmaps * ys).sum(dim=(2, 3))  # (B, K)
    cx = (heatmaps * xs).sum(dim=(2, 3))  # (B, K)
    return torch.stack([cx, cy], dim=-1)  # (B, K, 2) in (x, y)


# ---------------------------------------------------------------------------
# ViTPose body-proportion encoder
# ---------------------------------------------------------------------------

class DifferentiableBodyProportionEncoder(nn.Module):
    """Frozen ViTPose-Plus-Base encoder for the body-proportion anchor loss.

    Produces 8 pose-invariant bone-length ratios (10 with head). The cache
    ``encode()`` uses the HF processor; the training ``forward()`` replicates
    that affine warp differentiably so gradients reach the decoded pixels.
    """

    # COCO 17 keypoint indices
    L_SHOULDER, R_SHOULDER = 5, 6
    L_ELBOW, R_ELBOW = 7, 8
    L_WRIST, R_WRIST = 9, 10
    L_HIP, R_HIP = 11, 12
    L_KNEE, R_KNEE = 13, 14
    L_ANKLE, R_ANKLE = 15, 16

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    INPUT_SIZE = (256, 192)  # (height, width) ViTPose input
    VIS_THRESHOLD = 0.2

    def __init__(self, device: Optional[torch.device] = None):
        super().__init__()
        from transformers import VitPoseForPoseEstimation, VitPoseImageProcessor

        self.processor = VitPoseImageProcessor.from_pretrained(
            "usyd-community/vitpose-plus-base"
        )
        self.model = VitPoseForPoseEstimation.from_pretrained(
            "usyd-community/vitpose-plus-base",
            torch_dtype=torch.float16,
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.register_buffer(
            "_img_mean", torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "_img_std", torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )
        if device is not None:
            self.to(device)

    @staticmethod
    def _heatmaps_to_coords(heatmaps: torch.Tensor) -> torch.Tensor:
        """Integral-regression coords for ViTPose's unnormalized ~Gaussian maps.

        Raw ViTPose heatmaps sum to ~20 (not a distribution), so a plain
        centroid is meaningless (~800px off). Clamp >= 0, normalize each
        keypoint map to a spatial distribution, then take its expectation.
        Matches the argmax peak to ~3px while staying differentiable.
        """
        hm = heatmaps.clamp(min=0)
        hm = hm / hm.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        return _soft_argmax_2d(hm)

    @staticmethod
    def _compute_ratios(keypoints, visibilities, ref_ratios=None, include_head=False):
        """Pose-invariant bone-length ratios from COCO keypoints.

        Args:
            keypoints: (B, 17, 2) coordinates.
            visibilities: (B, 17) confidence in [0, 1].
            ref_ratios: (B, N) optional cached ratios; low-confidence ratios are
                replaced with these (detached, zero gradient).
            include_head: add head ratios (nose-to-shoulder, ear-to-ear).
        Returns:
            (ratios (B, N), ratio_vis (B, N)) with N = 8 or 10.
        """
        kp = keypoints
        vis = visibilities
        threshold = DifferentiableBodyProportionEncoder.VIS_THRESHOLD

        def dist(i, j):
            return (kp[:, i] - kp[:, j]).pow(2).sum(-1).clamp(min=1e-6).sqrt()

        def min_vis(*indices):
            return torch.stack([vis[:, i] for i in indices], dim=-1).min(dim=-1).values

        upper_arm = (dist(5, 7) + dist(6, 8)) / 2
        forearm = (dist(7, 9) + dist(8, 10)) / 2
        thigh = (dist(11, 13) + dist(12, 14)) / 2
        shin = (dist(13, 15) + dist(14, 16)) / 2

        shoulder_mid = (kp[:, 5] + kp[:, 6]) / 2
        hip_mid = (kp[:, 11] + kp[:, 12]) / 2
        torso = (shoulder_mid - hip_mid).pow(2).sum(-1).clamp(min=1e-6).sqrt()
        shoulder_w = dist(5, 6)
        hip_w = dist(11, 12)

        height = (torso + thigh + shin).clamp(min=1e-4)

        ratio_list = [
            upper_arm / height,
            forearm / height,
            thigh / height,
            shin / height,
            torso / height,
            shoulder_w / hip_w.clamp(min=1e-4),
            upper_arm / forearm.clamp(min=1e-4),
            thigh / shin.clamp(min=1e-4),
        ]
        vis_list = [
            min_vis(5, 6, 7, 8),
            min_vis(7, 8, 9, 10),
            min_vis(11, 12, 13, 14),
            min_vis(13, 14, 15, 16),
            min_vis(5, 6, 11, 12),
            min_vis(5, 6, 11, 12),
            min_vis(5, 6, 7, 8, 9, 10),
            min_vis(11, 12, 13, 14, 15, 16),
        ]

        if include_head:
            head_height = (kp[:, 0] - shoulder_mid).pow(2).sum(-1).clamp(min=1e-6).sqrt()
            ratio_list.append(head_height / height)
            vis_list.append(min_vis(0, 5, 6))
            head_width = dist(3, 4)
            ratio_list.append(head_width / shoulder_w.clamp(min=1e-4))
            vis_list.append(min_vis(3, 4, 5, 6))

        ratios = torch.stack(ratio_list, dim=-1)
        ratio_vis = torch.stack(vis_list, dim=-1)

        if ref_ratios is not None:
            low_conf = ratio_vis < threshold
            ratios = torch.where(low_conf, ref_ratios.detach(), ratios)
            ratio_vis = torch.where(low_conf, torch.zeros_like(ratio_vis), ratio_vis)

        return ratios, ratio_vis

    @torch.no_grad()
    def encode(self, pil_image, include_head=False) -> torch.Tensor:
        """Cache path: PIL -> (2*N,) ratios+vis CPU tensor (zeros if no body)."""
        img = pil_image.convert("RGB")
        w, h = img.size
        boxes = [[[0.0, 0.0, float(w), float(h)]]]  # full image as bbox

        inputs = self.processor(images=img, boxes=boxes, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(
            device=next(self.model.parameters()).device,
            dtype=next(self.model.parameters()).dtype,
        )
        outputs = self.model(
            pixel_values, dataset_index=torch.tensor([0], device=pixel_values.device)
        )
        heatmaps = outputs.heatmaps.float()  # (1, 17, 64, 48)
        coords = self._heatmaps_to_coords(heatmaps)  # (1, 17, 2)
        confidence = heatmaps.flatten(2).max(dim=2).values  # (1, 17)
        ratios, ratio_vis = self._compute_ratios(coords, confidence, include_head=include_head)
        n = ratios.shape[-1]
        if float(ratio_vis.mean().item()) < 0.1:
            return torch.zeros(n * 2)
        return torch.cat([ratios.squeeze(0), ratio_vis.squeeze(0)], dim=0).cpu()  # (2*N,)

    def forward(self, pixels: torch.Tensor, ref_ratios=None, include_head: bool = False):
        """Differentiable training path.

        Args:
            pixels: (B, 3, H, W) in [0, 1].
            ref_ratios: (B, N) cached reference ratios for low-conf fallback.
            include_head: add head ratios.
        Returns:
            (ratios (B, N), ratio_vis (B, N)).
        """
        from transformers.models.vitpose.image_processing_vitpose import (
            box_to_center_and_scale, get_warp_matrix,
        )

        pixels = pixels.float()
        B, C, H, W = pixels.shape
        out_w, out_h = self.INPUT_SIZE[1], self.INPUT_SIZE[0]  # (192, 256)

        all_kp, all_vis = [], []
        with torch.amp.autocast("cuda", enabled=False):
            for i in range(B):
                sample = pixels[i:i + 1]  # (1, 3, H, W)
                bbox_coco = [0, 0, W, H]  # full image, COCO (x, y, w, h)
                center, scale = box_to_center_and_scale(
                    bbox_coco, out_w, out_h, normalize_factor=200.0, padding_factor=1.25
                )
                warp_mat = get_warp_matrix(
                    0, center * 2.0,
                    np.array([out_w - 1, out_h - 1], dtype=np.float32),
                    scale * 200.0,
                )
                # Replicate the HF processor's affine via grid_sample (differentiable).
                M = np.vstack([warp_mat, [0, 0, 1]])
                M_inv = np.linalg.inv(M)
                S_in = np.array([[2.0 / (W - 1), 0, -1], [0, 2.0 / (H - 1), -1], [0, 0, 1]])
                S_out_inv = np.array(
                    [[(out_w - 1) / 2.0, 0, (out_w - 1) / 2.0],
                     [0, (out_h - 1) / 2.0, (out_h - 1) / 2.0],
                     [0, 0, 1]]
                )
                theta_np = (S_in @ M_inv @ S_out_inv)[:2, :]
                theta = torch.from_numpy(theta_np).float().unsqueeze(0).to(sample.device)
                grid = F.affine_grid(theta, (1, C, out_h, out_w), align_corners=True)
                sample = F.grid_sample(
                    sample, grid, align_corners=True, mode="bilinear", padding_mode="zeros"
                )

                mean = self._img_mean.to(sample.device, sample.dtype)
                std = self._img_std.to(sample.device, sample.dtype)
                sample = (sample - mean) / std

                model_dtype = next(self.model.parameters()).dtype
                heatmaps = self.model(
                    sample.to(model_dtype),
                    dataset_index=torch.tensor([0], device=sample.device),
                ).heatmaps.float()
                coords = self._heatmaps_to_coords(heatmaps)
                # Detach peak confidence to avoid "green dot" artifacts (source comment).
                confidence = heatmaps.flatten(2).max(dim=2).values.detach()
                all_kp.append(coords)
                all_vis.append(confidence)

        keypoints = torch.cat(all_kp, dim=0)  # (B, 17, 2)
        visibilities = torch.cat(all_vis, dim=0)  # (B, 17)
        return self._compute_ratios(
            keypoints, visibilities, ref_ratios=ref_ratios, include_head=include_head
        )


# ---------------------------------------------------------------------------
# GT-ratio caching (stamps file items for lazy worker reads)
# ---------------------------------------------------------------------------

def _bp_cache_path(file_item) -> str:
    img_dir = os.path.dirname(file_item.path)
    cache_dir = os.path.join(img_dir, "_body_proportion_cache")
    filename_no_ext = os.path.splitext(os.path.basename(file_item.path))[0]
    return os.path.join(cache_dir, f"{filename_no_ext}_bodyprop.safetensors")


def _bp_cache_hit(cache_path: str, key: str, version_key: str) -> bool:
    if not os.path.exists(cache_path):
        return False
    try:
        with safe_open(cache_path, framework="pt", device="cpu") as f:
            return key in f.keys() and version_key in f.keys()
    except Exception:  # noqa: BLE001
        return False


def _atomic_save_file(save_data: dict, cache_path: str) -> None:
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


def cache_body_proportion(
    file_items: List,
    config,
    *,
    encoder: DifferentiableBodyProportionEncoder,
) -> None:
    """Stamp every file item with its body-proportion cache metadata and persist GT ratios.

    GT ratios are computed from the RAW source image (full-image ViTPose) and are
    transform-independent, so the cache is one file per image + a version key
    (which encodes ``include_head``). The resident tensor is never held on the
    shared file-list item; the worker re-reads lazily via ``get_body_proportion_gt()``.
    """
    from PIL import Image
    from PIL.ImageOps import exif_transpose

    include_head = bool(getattr(config, "include_head", False))
    version_key = CACHE_VERSION_KEY_HEAD if include_head else CACHE_VERSION_KEY_BODY
    key = "body_proportion_gt"
    hits, misses, no_body = 0, 0, 0

    for file_item in tqdm(file_items, desc="Caching GT body proportions"):
        cache_path = _bp_cache_path(file_item)
        file_item._bp_cache_path = cache_path
        file_item._bp_cache_key = key
        file_item.is_body_proportion_cached = True
        file_item.body_proportion_gt = None

        if _bp_cache_hit(cache_path, key, version_key):
            hits += 1
            continue

        misses += 1
        pil_image = exif_transpose(Image.open(file_item.path)).convert("RGB")
        prop_tensor = encoder.encode(pil_image, include_head=include_head)  # (2*N,)

        if float(prop_tensor.abs().sum().item()) < 1e-6:
            no_body += 1

        _atomic_save_file(
            {key: prop_tensor, version_key: torch.ones(1)},
            cache_path,
        )

    if hits or misses:
        print(
            f"  - GT body-proportion cache: {hits} reused, {misses} computed "
            f"(key {key!r}, include_head={include_head})."
        )
    if no_body > 0:
        print(f"  - Warning: no body detected in {no_body}/{len(file_items)} images")
