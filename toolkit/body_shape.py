"""HybrIK SMPL body-shape perceptor + GT-beta caching.

A frozen ResNet-34 + FC head (ported from HybrIK) regresses 10-dim SMPL betas
from a 256x256 body crop. The live loss decodes the predicted x0 through the
VAE under gradient, runs the HybrIK encoder, and matches its 10 betas against
the cached GT via L1 (with a cosine gate). Gradients flow through the
differentiable square-crop + resize + backbone + head.

Only the body-SHAPE auxiliary loss (SMPL betas) is ported here -- distinct
from body-proportion (ViTPose ratios, already ported). Dependencies:
torchvision.models (ResNet-34, present). The HybrIK checkpoint
(``hybrik_resnet34.pth``) is Google-Drive-only (no HF mirror); ``gdown``
auto-downloads it if installed, otherwise the constructor raises a clean
FileNotFoundError naming the manual fetch.
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
import torchvision.models as models

BETA_DIM = 10
INPUT_SIZE = 256
CACHE_VERSION_KEY = "body_shape_v1"
GDRIVE_ID = "19ktHbERz0Un5EzJYZBdzdzTrFyd9gLCx"


class DifferentiableBodyShapeEncoder(nn.Module):
    """Frozen HybrIK ResNet-34 + FC head regressing 10-dim SMPL betas."""

    BETA_DIM = 10
    INPUT_SIZE = 256

    def __init__(self, device: Optional[torch.device] = None):
        super().__init__()
        resnet = models.resnet34(weights=None)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.decshape = nn.Linear(1024, self.BETA_DIM)
        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)

        self.register_buffer("init_shape", torch.zeros(1, self.BETA_DIM))
        # HybrIK's own normalization constants (NOT canonical ImageNet).
        self.register_buffer("img_mean", torch.tensor([0.406, 0.457, 0.480]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.225, 0.224, 0.229]).view(1, 3, 1, 1))

        self._load_pretrained()
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)
        if device is not None:
            self.to(device)

    def _load_pretrained(self):
        """Download and load HybrIK ResNet-34 weights (Google Drive via gdown)."""
        search_paths = [
            os.path.expanduser("~/.cache/hybrik/hybrik_resnet34.pth"),
            "/tmp/hybrik_resnet34.pth",
        ]
        ckpt_path = None
        for p in search_paths:
            if os.path.exists(p):
                ckpt_path = p
                break

        if ckpt_path is None:
            cache_dir = os.path.expanduser("~/.cache/hybrik")
            os.makedirs(cache_dir, exist_ok=True)
            ckpt_path = os.path.join(cache_dir, "hybrik_resnet34.pth")
            try:
                import gdown
                gdown.download(id=GDRIVE_ID, output=ckpt_path, quiet=False)
            except Exception as e:
                raise FileNotFoundError(
                    f"HybrIK ResNet-34 weights not found. Download from Google Drive "
                    f"(ID: {GDRIVE_ID}) to {ckpt_path}.\n"
                    f"Install gdown for automatic download: pip install gdown"
                ) from e

        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        key_map = {
            "preact.conv1.weight": "conv1.weight",
            "preact.bn1.weight": "bn1.weight",
            "preact.bn1.bias": "bn1.bias",
            "preact.bn1.running_mean": "bn1.running_mean",
            "preact.bn1.running_var": "bn1.running_var",
            "preact.bn1.num_batches_tracked": "bn1.num_batches_tracked",
        }

        new_sd = {}
        for old_key, tensor in sd.items():
            if old_key.startswith(("smpl.", "deconv_", "final_layer", "deccam", "decphi")):
                continue
            if old_key == "init_shape":
                new_sd["init_shape"] = tensor.unsqueeze(0) if tensor.dim() == 1 else tensor
                continue
            if old_key == "init_cam":
                continue
            if old_key in key_map:
                new_sd[key_map[old_key]] = tensor
            elif old_key.startswith("preact."):
                new_sd[old_key.replace("preact.", "")] = tensor
            elif old_key.startswith(("fc1.", "fc2.", "decshape.", "drop1.", "drop2.")):
                new_sd[old_key] = tensor

        result = self.load_state_dict(new_sd, strict=False)
        unexpected_missing = [k for k in result.missing_keys if k not in ("img_mean", "img_std")]
        if unexpected_missing:
            print(f"  [body_shape] Warning: missing keys: {unexpected_missing}")

    def _backbone(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)

    def _predict_betas(self, features):
        # No ReLU between FC layers (HybrIK preserves negative activations).
        x = self.drop1(self.fc1(features))
        x = self.drop2(self.fc2(x))
        return self.decshape(x) + self.init_shape

    @staticmethod
    def _square_crop(pil_image, bbox=None):
        """Square aspect-preserving crop around the person (HybrIK scale_mult=1.25)."""
        w, h = pil_image.size
        if bbox is not None:
            x1, y1, x2, y2 = [float(v) for v in bbox]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            bw = x2 - x1
            bh = y2 - y1
        else:
            cx, cy = w / 2, h / 2
            bw, bh = float(w), float(h)
        size = max(bw, bh) * 1.25
        half = size / 2
        x1 = max(0, int(cx - half))
        y1 = max(0, int(cy - half))
        x2 = min(w, int(cx + half))
        y2 = min(h, int(cy + half))
        return pil_image.crop((x1, y1, x2, y2))

    @torch.no_grad()
    def encode(self, pil_image, person_bbox=None) -> torch.Tensor:
        """PIL -> (10,) beta tensor on CPU."""
        crop = self._square_crop(pil_image, person_bbox)
        tensor = torch.from_numpy(np.array(crop)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        tensor = F.interpolate(
            tensor, size=(self.INPUT_SIZE, self.INPUT_SIZE),
            mode="bilinear", align_corners=False,
        )
        device = next(self.parameters()).device
        tensor = tensor.to(device)
        tensor = (tensor - self.img_mean) / self.img_std
        features = self._backbone(tensor)
        betas = self._predict_betas(features)
        return betas.squeeze(0).cpu()

    def forward(self, pixels: torch.Tensor, person_bboxes: Optional[List] = None) -> torch.Tensor:
        """Differentiable training path: (B,3,H,W) in [0,1] -> (B,10) betas."""
        pixels = pixels.float()
        if person_bboxes is not None:
            crops = []
            for i in range(pixels.shape[0]):
                bbox = person_bboxes[i]
                if bbox is not None:
                    ph, pw = pixels.shape[2], pixels.shape[3]
                    x1, y1, x2, y2 = bbox
                    bw, bh = x2 - x1, y2 - y1
                    cx_bbox = (x1 + x2) / 2
                    cy_bbox = (y1 + y2) / 2
                    size = max(bw, bh) * 1.25
                    half = size / 2
                    cx1 = max(0, int(round(float(cx_bbox - half))))
                    cy1 = max(0, int(round(float(cy_bbox - half))))
                    cx2 = min(pw, int(round(float(cx_bbox + half))))
                    cy2 = min(ph, int(round(float(cy_bbox + half))))
                    if cx2 > cx1 and cy2 > cy1:
                        crop = pixels[i:i + 1, :, cy1:cy2, cx1:cx2]
                    else:
                        crop = pixels[i:i + 1]
                else:
                    crop = pixels[i:i + 1]
                crop = F.interpolate(
                    crop, size=(self.INPUT_SIZE, self.INPUT_SIZE),
                    mode="bilinear", align_corners=False,
                )
                crops.append(crop)
            pixels = torch.cat(crops, dim=0)
        else:
            _, _, h, w = pixels.shape
            if h != w:
                s = min(h, w)
                y_off = (h - s) // 2
                x_off = (w - s) // 2
                pixels = pixels[:, :, y_off:y_off + s, x_off:x_off + s]
            pixels = F.interpolate(
                pixels, size=(self.INPUT_SIZE, self.INPUT_SIZE),
                mode="bilinear", align_corners=False,
            )

        pixels = (pixels - self.img_mean) / self.img_std
        with torch.amp.autocast("cuda", enabled=False):
            features = self._backbone(pixels)
            betas = self._predict_betas(features)
        return betas


# ---------------------------------------------------------------------------
# GT-beta caching (stamps file items for lazy worker reads)
# ---------------------------------------------------------------------------

def _body_shape_cache_path(file_item) -> str:
    img_dir = os.path.dirname(file_item.path)
    cache_dir = os.path.join(img_dir, "_body_shape_cache")
    filename_no_ext = os.path.splitext(os.path.basename(file_item.path))[0]
    return os.path.join(cache_dir, f"{filename_no_ext}_bodyshape.safetensors")


def _body_shape_cache_hit(cache_path: str, key: str) -> bool:
    if not os.path.exists(cache_path):
        return False
    try:
        with safe_open(cache_path, framework="pt", device="cpu") as f:
            return key in f.keys() and CACHE_VERSION_KEY in f.keys()
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


def cache_body_shape(
    file_items: List,
    config,
    *,
    encoder: DifferentiableBodyShapeEncoder,
) -> None:
    """Stamp every file item with its body-shape cache metadata and persist GT betas.

    GT betas come from the raw source image (transform-independent). No person
    bbox is used (full-image square crop); the resident tensor is never held on
    the file-list item.
    """
    from PIL import Image
    from PIL.ImageOps import exif_transpose

    key = "body_shape_embedding"
    hits, misses, no_body = 0, 0, 0

    for file_item in tqdm(file_items, desc="Caching GT body shapes"):
        cache_path = _body_shape_cache_path(file_item)
        file_item._body_shape_cache_path = cache_path
        file_item._body_shape_cache_key = key
        file_item.is_body_shape_cached = True
        file_item.body_shape_gt = None

        if _body_shape_cache_hit(cache_path, key):
            hits += 1
            continue

        misses += 1
        pil_image = exif_transpose(Image.open(file_item.path)).convert("RGB")
        betas = encoder.encode(pil_image)  # (10,)

        if float(betas.abs().sum().item()) < 1e-6:
            no_body += 1

        _atomic_save_file(
            {key: betas, CACHE_VERSION_KEY: torch.ones(1)},
            cache_path,
        )

    if hits or misses:
        print(f"  - GT body-shape cache: {hits} reused, {misses} computed (key {key!r}).")
    if no_body > 0:
        print(f"  - Warning: zero body shape for {no_body}/{len(file_items)} images")
