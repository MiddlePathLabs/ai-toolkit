"""ArcFace face-identity perceptor + GT-embedding caching.

A frozen ArcFace (w600k_r50, from the InsightFace ``buffalo_l`` pack) produces a
512-d identity embedding. The live loss decodes the predicted x0 through the
VAE under gradient, crops the face region, and matches its ArcFace embedding
against the cached GT embedding via cosine similarity. Gradients flow through
the differentiable face crop + the onnx2torch GraphModule forward.

Only the face-IDENTITY auxiliary loss is ported here. The source module also
contained face-token CONDITIONING (FaceIDProjector / VisionFaceProjector), a
landmark shape loss (MediaPipe FaceMesh), face suppression, and video/5D
paths -- all intentionally excluded.

Dependencies (NOT in requirements_base.txt -- install manually, see
requirements_perceptual.txt): insightface, onnx2torch, onnxruntime-gpu. The
CUDA-provider shadowing hazard (insightface pulls CPU onnxruntime which
clobbers onnxruntime-gpu) requires uninstalling plain onnxruntime after
install. All heavy imports are lazy so selecting Krea never imports them.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

IDENTITY_EMBED_DIM = 512
CACHE_VERSION_KEY = "face_identity_v1"


def _require_face_deps():
    """Lazy-import the face-identity deps with a clean ImportError.

    Raising here (at first use) rather than at module import keeps the module
    importable when the deps are absent, so the rest of the trainer works.
    """
    try:
        import onnx2torch  # noqa: F401
        from insightface.app import FaceAnalysis  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "face-identity anchor requires insightface + onnx2torch + onnxruntime-gpu.\n"
            "Install (note the onnxruntime shadowing fix):\n"
            "  pip install insightface onnx2torch onnxruntime-gpu\n"
            "  pip uninstall -y onnxruntime\n"
            "  pip install --no-deps onnxruntime-gpu\n"
            "Then verify: python -c \"import onnxruntime as ort; "
            "print(ort.get_available_providers())\" lists CUDAExecutionProvider."
        ) from e


class FaceIDExtractor:
    """InsightFace detection + recognition (buffalo_l) for GT caching.

    Detects the largest face and returns its 512-d normed embedding + bbox.
    Used only in the no-grad cache pass; the live loss uses
    DifferentiableFaceEncoder (the onnx2torch path).
    """

    def __init__(self, model_name: str = "buffalo_l", device_id: int = 0):
        _require_face_deps()
        from insightface.app import FaceAnalysis
        self.app = FaceAnalysis(
            name=model_name,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=device_id, det_size=(640, 640))
        self._warn_if_cpu_only()

    def _warn_if_cpu_only(self):
        active = set()
        for m in self.app.models.values():
            try:
                active.update(m.session.get_providers())
            except Exception:  # noqa: BLE001
                pass
        if "CUDAExecutionProvider" not in active:
            try:
                import onnxruntime as ort
                avail = ort.get_available_providers()
            except Exception:  # noqa: BLE001
                avail = ["<unknown>"]
            bar = "!" * 80
            print(
                f"\n{bar}\n[face_id] WARNING: InsightFace is running on CPU "
                f"(active providers: {sorted(active)}).\n"
                f"          onnxruntime available providers: {avail}\n"
                f"          Face-embedding caching will be EXTREMELY slow. Usually the\n"
                f"          CPU-only `onnxruntime` package is shadowing `onnxruntime-gpu`,\n"
                f"          or cuDNN is not on the library path. Fix:\n"
                f"            pip uninstall -y onnxruntime && "
                f"pip install --no-deps onnxruntime-gpu\n{bar}\n"
            )

    def _get_largest_face(self, faces):
        return sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )[0]

    def _detect(self, image: np.ndarray):
        # RetinaFace anchors miss faces that fill the frame; retry on a 25%-
        # padded copy and subtract the pad offset so bboxes stay in-image.
        faces = self.app.get(image)
        if len(faces) > 0:
            return faces, 0
        h, w = image.shape[:2]
        pad = max(h, w) // 4
        padded = np.full((h + 2 * pad, w + 2 * pad, 3), 128, dtype=image.dtype)
        padded[pad:pad + h, pad:pad + w] = image
        faces = self.app.get(padded)
        if len(faces) == 0:
            return [], 0
        for f in faces:
            f.bbox = f.bbox - np.array([pad, pad, pad, pad], dtype=f.bbox.dtype)
        return faces, pad

    def extract_from_pil_with_bbox(self, pil_image) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (512-d embedding, [x1,y1,x2,y2] bbox) or (None, None)."""
        import cv2
        pil_image = pil_image.convert("RGB")
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        faces, _ = self._detect(image)
        if len(faces) == 0:
            return None, None
        face = self._get_largest_face(faces)
        return face.normed_embedding.astype(np.float32), face.bbox.astype(np.float32)


def _crop_face_pil(pil_image, bbox, padding: float = 0.15):
    """PIL face crop with fractional padding, clipped to image bounds."""
    from PIL import Image
    w, h = pil_image.size
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw, bh = x2 - x1, y2 - y1
    cx1 = max(0, int(round(x1 - bw * padding)))
    cy1 = max(0, int(round(y1 - bh * padding)))
    cx2 = min(w, int(round(x2 + bw * padding)))
    cy2 = min(h, int(round(y2 + bh * padding)))
    if cx2 <= cx1 or cy2 <= cy1:
        return pil_image
    return pil_image.crop((cx1, cy1, cx2, cy2))


class DifferentiableFaceEncoder(nn.Module):
    """Frozen ArcFace (w600k_r50 via onnx2torch) for the identity anchor loss.

    The cache ``encode()`` takes a PIL face crop; the training ``forward()``
    takes a (B,3,H,W) [0,1] batch + per-sample bboxes and crops differentiably.
    Both pad to square, resize to 112x112, flip RGB->BGR, normalize to [-1,1].
    """

    def __init__(self, model_name: str = "buffalo_l", device: Optional[torch.device] = None):
        super().__init__()
        _require_face_deps()
        import onnx2torch
        onnx_path = os.path.join(
            os.path.expanduser("~"), ".insightface", "models", model_name, "w600k_r50.onnx"
        )
        if not os.path.exists(onnx_path):
            print(f"  [face_id] ArcFace model not found, downloading via InsightFace...")
            try:
                from insightface.app import FaceAnalysis
                app = FaceAnalysis(name=model_name, providers=["CPUExecutionProvider"])
                app.prepare(ctx_id=-1, det_size=(160, 160))
                del app
            except Exception as e:
                raise FileNotFoundError(
                    f"ArcFace ONNX model not found at {onnx_path}. "
                    f"Install insightface for automatic download:\n"
                    f"  pip install insightface onnx2torch onnxruntime-gpu\n"
                    f"  pip uninstall -y onnxruntime && pip install --no-deps onnxruntime-gpu\n"
                    f"Or manually download the buffalo_l model pack to ~/.insightface/models/"
                ) from e
        # onnx2torch.safe_shape_inference writes a temp file next to the source
        # ONNX -> PermissionError on Windows when the model dir has a live handle.
        # Staging the ONNX into a temp dir avoids the contention.
        with tempfile.TemporaryDirectory() as _td:
            _local_onnx = os.path.join(_td, os.path.basename(onnx_path))
            shutil.copy2(onnx_path, _local_onnx)
            self.model = onnx2torch.convert(_local_onnx)
        self.model.eval()
        self.model.requires_grad_(False)
        if device is not None:
            self.to(device)

    @torch.no_grad()
    def encode(self, pil_image) -> torch.Tensor:
        """PIL face crop -> (512,) L2-normalized CPU embedding."""
        tensor = torch.from_numpy(np.array(pil_image)).permute(2, 0, 1).float().unsqueeze(0)
        _, _, th, tw = tensor.shape
        if tw != th:  # pad to square
            diff = abs(tw - th)
            if tw > th:
                tensor = F.pad(tensor, (0, 0, diff // 2, diff - diff // 2), mode="constant", value=0)
            else:
                tensor = F.pad(tensor, (diff // 2, diff - diff // 2, 0, 0), mode="constant", value=0)
        tensor = F.interpolate(tensor, size=(112, 112), mode="bilinear", align_corners=False)
        tensor = tensor.squeeze(0).flip(0)  # RGB -> BGR
        tensor = (tensor - 127.5) / 127.5  # [0,255] -> [-1,1]
        tensor = tensor.unsqueeze(0).to(next(self.model.parameters()).device)
        emb = self.model(tensor)
        return F.normalize(emb, p=2, dim=-1).squeeze(0).cpu()

    def forward(self, pixels: torch.Tensor, bboxes: Optional[List] = None,
                return_crops: bool = False):
        """Differentiable training path.

        Args:
            pixels: (B, 3, H, W) in [0, 1] RGB.
            bboxes: list of [x1,y1,x2,y2] per item in pixels coords, or None
                entries to fall back to the full image.
            return_crops: also return the (B,3,112,112) RGB crops fed to ArcFace.
        Returns:
            (B, 512) L2-normalized embeddings [, (B,3,112,112) crops].
        """
        if bboxes is not None:
            crops = []
            for i in range(pixels.shape[0]):
                bbox = bboxes[i]
                if bbox is not None:
                    ph, pw = pixels.shape[2], pixels.shape[3]
                    x1, y1, x2, y2 = bbox
                    bw, bh = x2 - x1, y2 - y1
                    pad_w, pad_h = bw * 0.15, bh * 0.15
                    cx1 = max(0, int(round(float(x1 - pad_w))))
                    cy1 = max(0, int(round(float(y1 - pad_h))))
                    cx2 = min(pw, int(round(float(x2 + pad_w))))
                    cy2 = min(ph, int(round(float(y2 + pad_h))))
                    if cx2 > cx1 and cy2 > cy1:
                        crop = pixels[i:i + 1, :, cy1:cy2, cx1:cx2]
                        _, _, ch, cw = crop.shape
                        if cw != ch:  # pad to square
                            diff = abs(cw - ch)
                            if cw > ch:
                                crop = F.pad(crop, (0, 0, diff // 2, diff - diff // 2), mode="constant", value=0)
                            else:
                                crop = F.pad(crop, (diff // 2, diff - diff // 2, 0, 0), mode="constant", value=0)
                    else:
                        crop = pixels[i:i + 1]
                else:
                    crop = pixels[i:i + 1]
                crop = F.interpolate(crop, size=(112, 112), mode="bilinear", align_corners=False)
                crops.append(crop)
            pixels = torch.cat(crops, dim=0)
        else:
            pixels = F.interpolate(pixels, size=(112, 112), mode="bilinear", align_corners=False)

        rgb_crops = pixels.detach() if return_crops else None
        pixels = pixels.flip(1)  # RGB -> BGR
        pixels = (pixels * 255.0 - 127.5) / 127.5  # [0,1] -> [-1,1]
        emb = self.model(pixels)
        emb = F.normalize(emb, p=2, dim=-1)
        if return_crops:
            return emb, rgb_crops
        return emb


# ---------------------------------------------------------------------------
# GT-identity caching (stamps file items for lazy worker reads)
# ---------------------------------------------------------------------------

def _face_cache_path(file_item) -> str:
    img_dir = os.path.dirname(file_item.path)
    cache_dir = os.path.join(img_dir, "_face_id_cache")
    filename_no_ext = os.path.splitext(os.path.basename(file_item.path))[0]
    return os.path.join(cache_dir, f"{filename_no_ext}_identity.safetensors")


def _face_cache_hit(cache_path: str, key: str) -> bool:
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


def cache_face_identity(
    file_items: List,
    config,
    *,
    encoder: DifferentiableFaceEncoder,
) -> None:
    """Stamp every file item with identity-cache metadata and persist GT embeddings.

    Detects the largest face (InsightFace), crops it, and runs the differentiable
    ArcFace encoder (no_grad) to get the GT target embedding -- the same
    preprocessing the live loss uses, so a perfect reconstruction scores cosine
    ~1.0. Stores the 512-d embedding + the face bbox (original-image coords).
    No-face images store a zero embedding + no bbox (filtered by the loss). GT is
    transform-independent (from the raw source image).
    """
    from PIL import Image
    from PIL.ImageOps import exif_transpose

    detector = FaceIDExtractor(model_name=getattr(config, "face_model", "buffalo_l"))
    key = "identity_embedding"
    bbox_key = "face_bbox"
    hits, misses, no_face = 0, 0, 0

    for file_item in tqdm(file_items, desc="Caching GT face identities"):
        cache_path = _face_cache_path(file_item)
        file_item._face_cache_path = cache_path
        file_item._face_cache_key = key
        file_item._face_bbox_key = bbox_key
        file_item.is_face_identity_cached = True
        file_item.identity_embedding = None
        file_item.face_bbox = None

        if _face_cache_hit(cache_path, key):
            hits += 1
            continue

        misses += 1
        pil_image = exif_transpose(Image.open(file_item.path)).convert("RGB")
        img_w, img_h = pil_image.size
        emb_np, bbox_np = detector.extract_from_pil_with_bbox(pil_image)

        if emb_np is None:
            no_face += 1
            save_data = {
                key: torch.zeros(IDENTITY_EMBED_DIM, dtype=torch.float32),
                bbox_key: torch.zeros(4, dtype=torch.float32),
                CACHE_VERSION_KEY: torch.ones(1),
            }
        else:
            face_crop = _crop_face_pil(pil_image, bbox_np, padding=0.15)
            emb_tensor = encoder.encode(face_crop)  # (512,)
            # Store bbox in normalized [0,1] coords (x1,y1,x2,y2)/img_size so the
            # loss can map it to the decoded x0 resolution without knowing the
            # bucket transform. Approximation: ignores aggressive crops, but the
            # SCRFD quality gate + 15% crop padding tolerate mild misalignment.
            norm_bbox = np.array([
                bbox_np[0] / img_w, bbox_np[1] / img_h,
                bbox_np[2] / img_w, bbox_np[3] / img_h,
            ], dtype=np.float32)
            save_data = {
                key: emb_tensor,
                bbox_key: torch.from_numpy(norm_bbox),
                CACHE_VERSION_KEY: torch.ones(1),
            }

        _atomic_save_file(save_data, cache_path)

    del detector
    torch.cuda.empty_cache()
    if hits or misses:
        print(
            f"  - GT face-identity cache: {hits} reused, {misses} computed (key {key!r})."
        )
    if no_face > 0:
        print(f"  - Warning: no face detected in {no_face}/{len(file_items)} images")
