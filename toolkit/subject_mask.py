"""Auto-masking pipeline: YOLO + SAM 2 + SegFormer-clothes.

Extracts per-image `person`, `body`, and `clothing` binary masks for use with
region-aware training losses. Mirrors the reference pipeline in
`scripts/profile_full_pipeline.py`. SegFormer is the primary source of truth;
SAM is loaded for a reference silhouette but not intersected into the final
masks (SAM drops pixels on low-contrast boundaries, SegFormer is semantic).

Phase 1: caching only. The resulting masks are attached to FileItemDTO but
are not consumed by any loss. See `toolkit/config_modules.SubjectMaskConfig`.
"""

import hashlib
import json
import os
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file, load_file
from tqdm import tqdm

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import FileItemDTO
    from toolkit.config_modules import SubjectMaskConfig


# ============================================================
# Config constants — copied verbatim from scripts/profile_full_pipeline.py
# ============================================================

SAM_HF_IDS: Dict[str, str] = {
    "tiny":      "facebook/sam2.1-hiera-tiny",
    "small":     "facebook/sam2.1-hiera-small",
    "base_plus": "facebook/sam2.1-hiera-base-plus",
    "large":     "facebook/sam2.1-hiera-large",
}

SEGFORMER_ID = "mattmdjaga/segformer_b2_clothes"

# "Body" = identity-relevant human parts we want to preserve.
# Hair is included because it's part of identity.
BODY_CLASSES = {"Hair", "Face", "Left-arm", "Right-arm", "Left-leg", "Right-leg"}
CLOTHING_CLASSES = {"Hat", "Sunglasses", "Upper-clothes", "Skirt", "Pants",
                    "Dress", "Belt", "Left-shoe", "Right-shoe", "Bag", "Scarf"}

CACHE_VERSION_KEY = "subject_mask_v3"  # transformed-source cache with full identity
CACHE_VERSION_KEY_MATCHED = "subject_mask_v4"  # + identity-matched subject/others masks

# Module-level insightface detector shared across per-dataset cache calls so
# the ONNX session loads once per run. Released via release_face_detector().
_FACE_DETECTOR: Optional['object'] = None


def _get_face_detector():
    global _FACE_DETECTOR
    if _FACE_DETECTOR is None:
        from toolkit.face_id import FaceIDExtractor
        _FACE_DETECTOR = FaceIDExtractor()
    return _FACE_DETECTOR


def release_face_detector():
    """Free the shared insightface detector (call after all mask caching)."""
    global _FACE_DETECTOR
    _FACE_DETECTOR = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_subject_reference(file_items: List['FileItemDTO']) -> torch.Tensor:
    """Average ArcFace embedding of the solo-subject dataset's faces.

    Detects faces on the raw (exif-transposed, untransformed) source images —
    a solo dataset has exactly one face per image, and the largest-face pick
    is the subject. Returns a L2-normalized (512,) float32 tensor (same math
    as the face-id reference tables: mean of normalized embeddings,
    renormalized). Raises when no image yields a face.
    """
    from PIL import Image
    from PIL.ImageOps import exif_transpose

    detector = _get_face_detector()
    embeds: List[torch.Tensor] = []
    no_face = 0
    for file_item in tqdm(file_items, desc="Building subject identity reference"):
        try:
            raw_pil = exif_transpose(Image.open(file_item.path)).convert('RGB')
            found = detector.detect_all_with_embeddings(raw_pil)
        except Exception as e:
            print(f"  - Warning: identity reference skipped unreadable {file_item.path}: {e}")
            no_face += 1
            continue
        if not found:
            no_face += 1
            continue
        emb, _bbox = found[0]  # area-sorted: largest face = subject in a solo set
        embeds.append(torch.from_numpy(emb.astype(np.float32)))
    if not embeds:
        raise ValueError(
            "subject_mask.subject_identity_dataset: no faces detected in any "
            "image of the identity dataset — cannot build the reference embedding."
        )
    if no_face > 0:
        print(f"  - Subject identity reference: {no_face}/{len(file_items)} images had no detectable face (skipped)")
    stacked = torch.stack(embeds)
    avg = stacked.mean(dim=0)
    avg = avg / (avg.norm() + 1e-8)
    print(f"  - Subject identity reference: averaged {len(embeds)} face embeddings")
    return avg.float()


# ============================================================
# Mask post-processing
# ============================================================


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    try:
        from scipy.ndimage import binary_fill_holes
        return binary_fill_holes(mask.astype(bool)).astype(np.uint8)
    except Exception:
        return mask.astype(np.uint8)


def _skin_probability(image_rgb_u8: np.ndarray) -> np.ndarray:
    """Soft skin-tone probability map in [0, 1] from a uint8 RGB image.

    Uses the classical YCrCb thresholds (Cr ∈ [133, 173], Cb ∈ [77, 127])
    blurred for a smooth probability. Tone-agnostic across most skin tones;
    over-dark or over-light skin can fall outside the range. Cheap (~ms
    per frame) and adds no model dependencies.
    """
    try:
        import cv2
        ycc = cv2.cvtColor(image_rgb_u8, cv2.COLOR_RGB2YCrCb)
        cr = ycc[..., 1].astype(np.float32)
        cb = ycc[..., 2].astype(np.float32)
        m = ((cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)).astype(np.float32)
        return cv2.GaussianBlur(m, (7, 7), 0)
    except Exception:
        return np.zeros(image_rgb_u8.shape[:2], dtype=np.float32)


def _smooth_mask(mask: np.ndarray, close_radius: int = 3, do_fill: bool = True) -> np.ndarray:
    """Clean stippling: morphological closing + hole fill.

    close_radius: pixel radius of the structuring disk. 3-5 works well at 1MP.
    """
    try:
        from scipy.ndimage import (binary_closing, binary_fill_holes,
                                   generate_binary_structure, iterate_structure)
        m = mask.astype(bool)
        struct = iterate_structure(generate_binary_structure(2, 2), close_radius)
        m = binary_closing(m, structure=struct)
        if do_fill:
            m = binary_fill_holes(m)
        return m.astype(bool)
    except Exception:
        return mask.astype(bool)


# ============================================================
# Extractor
# ============================================================


def _box_overlap(a, b):
    """Return (iou, inter_over_min_area) for two [x1,y1,x2,y2] boxes."""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    iou = inter / union if union > 0 else 0.0
    over_min = inter / min(area_a, area_b) if min(area_a, area_b) > 0 else 0.0
    return iou, over_min


def _merge_person_boxes(boxes, iou_threshold=0.55, contain_threshold=0.85):
    """Greedy-merge heavily-overlapping person boxes into their union box.

    YOLO frequently splits one large close-up person into overlapping partial
    detections; counting those as separate people would trigger isolation and
    produce a partial subject region (verified failure on a close-up still).
    Merging on IoU OR on inter-over-minor-area (nested fragments) collapses
    split detections while leaving genuinely disjoint people separate.
    """
    merged = [list(map(float, b)) for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                iou, over_min = _box_overlap(merged[i], merged[j])
                if iou >= iou_threshold or over_min >= contain_threshold:
                    bi, bj = merged[i], merged[j]
                    merged[i] = [min(bi[0], bj[0]), min(bi[1], bj[1]),
                                 max(bi[2], bj[2]), max(bi[3], bj[3])]
                    merged.pop(j)
                    changed = True
                    break
            if changed:
                break
    return merged


def _resolve_subject_match(face_boxes, cos, boxes, img_w, img_h,
                           threshold, margin, isolate_fn):
    """Pure decision core of the identity match (model-free, unit-testable).

    face_boxes: [x1,y1,x2,y2] arrays parallel to ``cos``.
    cos: cosine similarity of each face to the subject reference.
    boxes: YOLO person boxes for the image (heavily-overlapping boxes are
        merged first — split detections of one person are not rivals).
    isolate_fn: callable(person_box) -> (H, W) bool silhouette or None (SAM).

    Accepts a face iff its cosine is >= ``threshold`` and, with rival faces,
    it beats the runner-up by >= ``margin``. Region semantics: ``None`` means
    the whole person parse is the subject — ONLY when there is a single face
    AND at most one person box. A faceless companion (facing away, occluded,
    cropped) still counts as a rival person, so one detected face with two
    person boxes must isolate rather than label everyone the subject.
    """
    n = len(face_boxes)
    if n == 0:
        return {"matched": False, "region": None, "face_count": 0,
                "best_cos": 0.0, "second_cos": 0.0}
    boxes = _merge_person_boxes(boxes)
    order = np.argsort(cos)[::-1]
    best = int(order[0])
    best_cos = float(cos[best])
    second_cos = float(cos[order[1]]) if n > 1 else -1.0
    ok = best_cos >= threshold and (n == 1 or (best_cos - second_cos) >= margin)
    if not ok:
        return {"matched": False, "region": None, "face_count": n,
                "best_cos": best_cos, "second_cos": second_cos}

    # Single face AND at most one merged person box -> unambiguous, the whole
    # person parse is the subject. More faces OR more person boxes means a
    # rival person exists even if their face wasn't detected (facing away,
    # occluded, cropped) — isolate the matched face's person.
    if n == 1 and len(boxes) <= 1:
        # Unambiguous: the person parse in this image is the subject.
        return {"matched": True, "region": None, "face_count": 1,
                "best_cos": best_cos, "second_cos": second_cos}

    # Rival person present (second face, or a faceless second person box):
    # isolate the subject via the YOLO person box containing her face, then
    # the SAM silhouette prompted with that box.
    face_box = face_boxes[best]
    fx = (face_box[0] + face_box[2]) / 2.0
    fy = (face_box[1] + face_box[3]) / 2.0
    containing = [b for b in boxes
                  if b[0] <= fx <= b[2] and b[1] <= fy <= b[3]]
    if containing:
        # Tightest containing box (smallest area) avoids swallowing a
        # neighboring person when boxes overlap.
        person_box = min(containing, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    else:
        # Face outside every person box (odd crop) — pad the face box to a
        # rough upper-body prompt for SAM.
        fw = max(face_box[2] - face_box[0], 1.0)
        fh = max(face_box[3] - face_box[1], 1.0)
        person_box = [
            max(0.0, fx - 1.2 * fw), max(0.0, face_box[1] - 0.5 * fh),
            min(float(img_w), fx + 1.2 * fw), min(float(img_h), face_box[3] + 3.0 * fh),
        ]
    def _rectangle_region():
        # Padded person-box rectangle the caller intersects with the person
        # parse - cruder than SAM but functional.
        pad = 0.05 * max(img_h, img_w)
        x1 = max(0, int(person_box[0] - pad)); y1 = max(0, int(person_box[1] - pad))
        x2 = min(img_w, int(person_box[2] + pad)); y2 = min(img_h, int(person_box[3] + pad))
        rect = np.zeros((img_h, img_w), dtype=bool)
        rect[y1:y2, x1:x2] = True
        return rect

    def _face_cover(sil):
        fy1 = max(0, int(face_box[1])); fy2 = min(img_h, int(face_box[3]))
        fx1 = max(0, int(face_box[0])); fx2 = min(img_w, int(face_box[2]))
        if fy2 <= fy1 or fx2 <= fx1:
            return 1.0
        return float(sil[fy1:fy2, fx1:fx2].mean())

    silhouette = isolate_fn(person_box)
    isolation = "sam" if silhouette is not None else "rectangle"
    if silhouette is None:
        silhouette = _rectangle_region()
    # Quality gate: the isolated region must actually contain the matched
    # face. A patchy SAM silhouette (e.g. prompted with a near-full-frame box
    # after a spurious extra detection on a close-up) would otherwise split
    # the subject herself into subject+others. A failing SAM mask retries
    # with the rectangle; if that also misses the face the match FAILS so
    # both identity weights no-op for the image. Returning region=None here
    # is not an option - a rival person exists in this branch, and None
    # (whole person parse = subject) would label the rival as the subject.
    face_cover = _face_cover(silhouette)
    if face_cover < 0.5 and isolation == "sam":
        silhouette = _rectangle_region()
        isolation = "rectangle"
        face_cover = _face_cover(silhouette)
    if face_cover < 0.5:
        return {"matched": False, "region": None, "face_count": n,
                "best_cos": best_cos, "second_cos": second_cos,
                "isolation": "failed", "face_cover": face_cover}
    return {"matched": True, "region": silhouette, "face_count": n,
            "best_cos": best_cos, "second_cos": second_cos,
            "isolation": isolation}


class SubjectMaskExtractor:
    """Lazy-loads YOLO, SAM 2, and SegFormer once per instance.

    `.extract(pil_image)` returns a dict with keys:
        person:   np.bool_ (H, W) — body + clothing (pure SegFormer, smoothed)
        body:     np.bool_ (H, W) — hair/face/arms/legs (identity-relevant)
        clothing: np.bool_ (H, W) — upper/pants/skirt/dress/shoes/bag/etc

    All masks are at the ORIGINAL image resolution.
    """

    def __init__(self, config: 'SubjectMaskConfig'):
        self.config = config
        dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        self.dtype = dtype_map.get(config.dtype, torch.float16)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Lazy imports — keep import cost out of general toolkit import graph.
        from ultralytics import YOLO
        from transformers import (AutoConfig, AutoModelForSemanticSegmentation,
                                  Sam2Model, Sam2Processor, SegformerImageProcessor)

        # YOLO (person detector, COCO class 0)
        self.yolo = YOLO(config.yolo_ckpt)
        # warmup
        try:
            self.yolo.predict(np.zeros((640, 480, 3), dtype=np.uint8),
                              verbose=False, device=0 if self.device == "cuda" else "cpu")
        except Exception:
            # Non-fatal: GPU warmup can fail on some environments; real call will surface errors
            pass

        # SAM 2 (kept for debug / future; not intersected into final masks in Phase 1)
        sam_id = SAM_HF_IDS.get(config.sam_size, SAM_HF_IDS["small"])
        self.sam_processor = Sam2Processor.from_pretrained(sam_id)
        self.sam = Sam2Model.from_pretrained(sam_id, torch_dtype=self.dtype).to(self.device).eval()

        # SegFormer-clothes (primary source of truth for body/clothing semantics).
        # We do the resize ourselves (aspect-preserving, longest side =
        # ``segformer_res``) and pass ``do_resize=False`` at call time. The
        # processor's default ``{height,width}`` resize forces a square,
        # which severely distorts tall/wide images at high resolutions and
        # *decreases* mask accuracy as ``segformer_res`` grows.
        self.seg_processor = SegformerImageProcessor.from_pretrained(SEGFORMER_ID)
        self.seg = AutoModelForSemanticSegmentation.from_pretrained(
            SEGFORMER_ID, dtype=self.dtype
        ).to(self.device).eval()
        self.seg_cfg = AutoConfig.from_pretrained(SEGFORMER_ID)

        # Precompute body/clothing class id sets from the SegFormer config
        self._body_ids = {i for i, name in self.seg_cfg.id2label.items()
                          if name in BODY_CLASSES}
        self._clothing_ids = {i for i, name in self.seg_cfg.id2label.items()
                              if name in CLOTHING_CLASSES}

    # ------------------------------------------------------------------ #
    # Per-stage
    # ------------------------------------------------------------------ #

    def _run_yolo(self, pil_image, primary_only: Optional[bool] = None):
        """Return a list of [x1,y1,x2,y2] boxes sorted by area desc."""
        img_np = np.array(pil_image)
        device_arg = 0 if self.device == "cuda" else "cpu"
        results = self.yolo.predict(img_np, classes=[0], conf=self.config.yolo_conf,
                                    verbose=False, device=device_arg)
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        boxes = r.boxes.xyxy.cpu().numpy().tolist()
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        order = np.argsort(areas)[::-1]
        boxes = [boxes[i] for i in order]
        keep = self.config.primary_only if primary_only is None else primary_only
        if keep:
            boxes = boxes[:1]
        return boxes

    def _sam_box_mask(self, pil_image, box) -> Optional[np.ndarray]:
        """SAM 2 box-prompted person silhouette (H, W) bool, or None on failure.

        SAM is loaded for exactly this purpose: given the person box containing
        the identity-matched face, it separates that person from adjacent
        people better than a rectangle intersection. ``pred_masks`` comes back
        as (1, 1, num_heads, h, w) low-res logits — the highest-IoU head must
        be selected (the first head is frequently a partial/alternative mask).
        """
        try:
            W, H = pil_image.size
            inputs = self.sam_processor(
                images=pil_image,
                input_boxes=[[ [float(v) for v in box] ]],
                return_tensors="pt",
            )
            inputs = {k: (v.to(self.device, dtype=self.dtype) if v.is_floating_point()
                          else v.to(self.device))
                      for k, v in inputs.items()}
            with torch.inference_mode():
                outputs = self.sam(**inputs)
            best = int(outputs.iou_scores.flatten().argmax())
            pred = outputs.pred_masks
            if pred.dim() == 5:      # (B, prompts, heads, h, w)
                logits = pred[0, 0, best]
            elif pred.dim() == 4:    # (B, heads, h, w)
                logits = pred[0, best]
            else:
                logits = pred[best]
            logits = logits.float()[None, None]
            if logits.shape[-2:] != (H, W):
                logits = F.interpolate(
                    logits, size=(H, W), mode="bilinear", align_corners=False
                )
            return (logits[0, 0] > 0.0).cpu().numpy().astype(bool)
        except Exception:
            return None

    def _match_subject(self, pil_image, subject_reference: torch.Tensor, boxes):
        """Identity-match the subject's face; return the match result dict.

        Runs face detection/embeddings here, then delegates acceptance and
        region resolution to :func:`_resolve_subject_match` (pure,
        unit-testable). Accepts a face iff its cosine to the reference is
        >= match_threshold and (with rival faces) beats the runner-up by
        >= match_margin. Returns:
            matched: bool
            region:  np.bool_ (H, W) covering ONLY the subject's person, or
                     None when unambiguous (single face AND at most one
                     person box — the whole person mask IS the subject)
            face_count / best_cos / second_cos: for logging
        """
        cfg_thresh = float(getattr(self.config, 'match_threshold', 0.40))
        cfg_margin = float(getattr(self.config, 'match_margin', 0.10))
        detector = _get_face_detector()
        faces = detector.detect_all_with_embeddings(pil_image)
        W, H = pil_image.size
        cos: list = []
        face_boxes = []
        if faces:
            embs = np.stack([e for e, _b in faces]).astype(np.float32)
            ref = subject_reference.detach().cpu().numpy().astype(np.float32)
            cos = (embs @ ref).tolist()  # both sides L2-normalized
            face_boxes = [b for _e, b in faces]
        return _resolve_subject_match(
            face_boxes, cos, boxes, W, H, cfg_thresh, cfg_margin,
            lambda person_box: self._sam_box_mask(pil_image, person_box),
        )

    def _run_segformer(self, pil_image) -> np.ndarray:
        """Return (H, W) int32 class map at original image resolution.

        Aspect-preserving: longest side resized to ``segformer_res``, the
        other side rounded to a multiple of 32 (SegFormer's stride). Square
        forcing produced large drops in mask coverage on tall portraits at
        high resolutions because the horizontal upscale fed the model
        interpolation artifacts at scales it was never trained on.
        """
        from PIL import Image

        target = int(self.config.segformer_res)
        W, H = pil_image.size
        if H >= W:
            new_h = target
            new_w = max(32, int(round(W * target / H / 32)) * 32)
        else:
            new_w = target
            new_h = max(32, int(round(H * target / W / 32)) * 32)
        pil_in = pil_image if (new_w, new_h) == (W, H) else pil_image.resize(
            (new_w, new_h), Image.BICUBIC
        )

        inputs = self.seg_processor(
            images=pil_in, return_tensors="pt", do_resize=False,
        ).to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        with torch.inference_mode():
            logits = self.seg(**inputs).logits
            up = F.interpolate(logits.float(),
                               size=(H, W),
                               mode="bilinear", align_corners=False)
            # Skin-tone bias: where the image looks like skin in YCrCb,
            # add a positive bias to body-class logits so close-call
            # clothing/body pixels tip into body. Disabled at 0 (default).
            bias = float(getattr(self.config, "skin_bias", 0.0))
            if bias > 0.0 and len(self._body_ids) > 0:
                skin = _skin_probability(np.asarray(pil_image, dtype=np.uint8))
                skin_t = torch.from_numpy(skin).to(up.device, dtype=up.dtype)
                skin_t = skin_t.unsqueeze(0).unsqueeze(0)
                body_idx = torch.tensor(
                    sorted(self._body_ids), device=up.device, dtype=torch.long,
                )
                up[:, body_idx] = up[:, body_idx] + bias * skin_t
            class_map = up.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)
        return class_map

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #

    def extract(self, pil_image, subject_reference: Optional[torch.Tensor] = None) -> Dict[str, np.ndarray]:
        """Extract {person, body, clothing} bool masks at original resolution.

        Also returns the raw SegFormer ``class_map`` (int32 class ids) and the
        list of YOLO ``boxes`` so callers (e.g. debug preview rendering) can
        visualize detector inputs alongside masks. When ``subject_reference``
        is provided, additionally returns ``subject_match`` — the identity
        match result dict from :meth:`_match_subject` whose ``region``
        covers only the matched person (None = the whole person mask is the
        subject, e.g. single-face single-person images).

        SAM is run only for multi-face identity matching (person isolation);
        SegFormer is primary source of truth for the base masks.
        """
        # YOLO for detection (unused in final base mask but kept to signal
        # "no subject" and to host the subject's person box for matching).
        matching = subject_reference is not None
        boxes = self._run_yolo(pil_image, primary_only=(not matching))

        # SegFormer parsing is the semantic source of truth
        class_map = self._run_segformer(pil_image)

        body_parse = np.isin(class_map, list(self._body_ids))
        clothing_parse = np.isin(class_map, list(self._clothing_ids))

        body_mask = _smooth_mask(body_parse, close_radius=int(self.config.body_close_radius))
        clothing_mask = _smooth_mask(clothing_parse, close_radius=2)
        # person = body ∪ clothing (pure SegFormer), then closed + hole-filled.
        person_mask = _smooth_mask(body_mask | clothing_mask, close_radius=3)

        # True dilation grows the outer boundary (closing only fills holes),
        # so this is the knob users reach for when they want a padded mask.
        dilate_r = int(getattr(self.config, 'mask_dilate_radius', 0))
        if dilate_r > 0:
            try:
                from scipy.ndimage import (binary_dilation,
                                            generate_binary_structure,
                                            iterate_structure)
                struct = iterate_structure(
                    generate_binary_structure(2, 2), dilate_r,
                )
                person_mask = binary_dilation(person_mask, structure=struct)
            except Exception:
                pass

        result = {
            "person": person_mask.astype(np.bool_),
            "body": body_mask.astype(np.bool_),
            "clothing": clothing_mask.astype(np.bool_),
            "class_map": class_map.astype(np.int32),
            "boxes": boxes,
        }
        if matching:
            result["subject_match"] = self._match_subject(pil_image, subject_reference, boxes)
        return result

    def cleanup(self):
        """Free GPU memory held by loaded models."""
        try:
            del self.yolo
        except Exception:
            pass
        try:
            del self.sam
            del self.sam_processor
        except Exception:
            pass
        try:
            del self.seg
            del self.seg_processor
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================
# Debug preview rendering
# ============================================================


def _overlay_mask(image_rgb: np.ndarray, mask: np.ndarray, color, alpha: float = 0.55) -> np.ndarray:
    """Blend a binary mask onto an RGB image with a solid color + yellow outline."""
    out = image_rgb.astype(np.float32).copy()
    m = mask[..., None].astype(np.float32)
    color_layer = np.array(color, dtype=np.float32)
    out = out * (1 - alpha * m) + color_layer * alpha * m
    try:
        from scipy.ndimage import binary_dilation
        border = binary_dilation(mask.astype(bool), iterations=2) & (~mask.astype(bool))
        out[border] = np.array([255, 255, 0])
    except Exception:
        pass
    return np.clip(out, 0, 255).astype(np.uint8)


def _colormap_from_classes(class_map: np.ndarray, n_classes: int) -> np.ndarray:
    """Render a class map to an RGB color image using a deterministic palette."""
    rng = np.random.RandomState(7)
    pal = np.zeros((n_classes, 3), dtype=np.uint8)
    for i in range(1, n_classes):
        pal[i] = rng.randint(40, 230, 3)
    return pal[class_map.astype(np.int32)]


def _render_preview_tile_from_cache(
    pil_image,
    person_t: torch.Tensor,
    body_t: torch.Tensor,
    clothing_t: torch.Tensor,
    subject_t: Optional[torch.Tensor] = None,
    others_t: Optional[torch.Tensor] = None,
    col_width: int = 380,
):
    """4-panel (6 with identity matching) tile from cached bool masks: image |
    person | body | clothing [| subject | others].

    Used for upfront QC previews on cache hit, where the SegFormer ``class_map``
    isn't stored. Mirrors :func:`_render_preview_tile` minus the parse colormap.
    """
    from PIL import Image, ImageDraw, ImageFont
    img_np = np.array(pil_image)
    H, W = img_np.shape[:2]

    def _bool_to_np(t: torch.Tensor) -> np.ndarray:
        # _overlay_mask expects 0/1 (it does alpha*m); 0/255 saturates the blend.
        m = t.detach().cpu().to(torch.uint8).numpy()
        if m.shape != (H, W):
            # Resize via 0/255 so PIL.NEAREST works on visible values, then rebinarize.
            m255 = (m * 255)
            m = np.array(Image.fromarray(m255).resize((W, H), Image.NEAREST))
            m = (m > 127).astype(np.uint8)
        return m

    person = _bool_to_np(person_t)
    body = _bool_to_np(body_t)
    clothing = _bool_to_np(clothing_t)

    ov_person = _overlay_mask(img_np, person, (100, 180, 255))
    ov_body = _overlay_mask(img_np, body, (255, 120, 80))
    ov_clothing = _overlay_mask(img_np, clothing, (120, 255, 120))
    panels = [img_np, ov_person, ov_body, ov_clothing]
    labels = ["Original", "Person", "Body (hair+face+limbs)", "Clothing"]
    if subject_t is not None or others_t is not None:
        ov_subject = _overlay_mask(img_np, _bool_to_np(subject_t) if subject_t is not None
                                   else np.zeros((H, W), np.uint8), (255, 80, 80))
        ov_others = _overlay_mask(img_np, _bool_to_np(others_t) if others_t is not None
                                  else np.zeros((H, W), np.uint8), (80, 220, 255))
        panels += [ov_subject, ov_others]
        labels += ["Subject (identity match)", "Others"]

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    resized = []
    for a in panels:
        r = col_width / a.shape[1]
        new_h = int(a.shape[0] * r)
        resized.append(np.array(Image.fromarray(a).resize((col_width, new_h), Image.BILINEAR)))
    h_max = max(a.shape[0] for a in resized)
    label_h = 26
    canvas = Image.new("RGB", (col_width * len(panels) + 8 * (len(panels) - 1),
                               h_max + label_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for a, lbl in zip(resized, labels):
        canvas.paste(Image.fromarray(a), (x, label_h))
        draw.text((x + 6, 6), lbl, fill=(230, 230, 230), font=font)
        x += col_width + 8
    return canvas


def _render_preview_tile(pil_image, masks: Dict[str, np.ndarray], n_classes: int,
                         subject_person: Optional[np.ndarray] = None,
                         others_person: Optional[np.ndarray] = None,
                         col_width: int = 380):
    """5-panel (7 with identity matching) tile: image | person | body | clothing
    [| subject | others] | parse colormap.

    Returns a PIL Image ready to save.
    """
    from PIL import Image, ImageDraw, ImageFont
    img_np = np.array(pil_image)

    person = masks["person"].astype(np.uint8)
    body = masks["body"].astype(np.uint8)
    clothing = masks["clothing"].astype(np.uint8)
    class_map = masks["class_map"]

    ov_person = _overlay_mask(img_np, person, (100, 180, 255))
    ov_body = _overlay_mask(img_np, body, (255, 120, 80))
    ov_clothing = _overlay_mask(img_np, clothing, (120, 255, 120))
    color_map = _colormap_from_classes(class_map, n_classes)
    parse_blend = (img_np.astype(np.float32) * 0.5 + color_map.astype(np.float32) * 0.5)
    parse_blend = np.clip(parse_blend, 0, 255).astype(np.uint8)

    panels = [img_np, ov_person, ov_body, ov_clothing]
    labels = ["Original", "Person", "Body (hair+face+limbs)", "Clothing"]
    if subject_person is not None or others_person is not None:
        H, W = img_np.shape[:2]
        ov_subject = _overlay_mask(img_np, subject_person if subject_person is not None
                                   else np.zeros((H, W), np.uint8), (255, 80, 80))
        ov_others = _overlay_mask(img_np, others_person if others_person is not None
                                  else np.zeros((H, W), np.uint8), (80, 220, 255))
        panels += [ov_subject, ov_others]
        labels += ["Subject (identity match)", "Others"]
    panels.append(parse_blend)
    labels.append("Parse colormap")

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    resized = []
    for a in panels:
        r = col_width / a.shape[1]
        new_h = int(a.shape[0] * r)
        resized.append(np.array(Image.fromarray(a).resize((col_width, new_h), Image.BILINEAR)))
    h_max = max(a.shape[0] for a in resized)
    label_h = 26
    canvas = Image.new("RGB", (col_width * len(panels) + 8 * (len(panels) - 1),
                               h_max + label_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for a, lbl in zip(resized, labels):
        canvas.paste(Image.fromarray(a), (x, label_h))
        draw.text((x + 6, 6), lbl, fill=(230, 230, 230), font=font)
        x += col_width + 8
    return canvas


# ============================================================
# Cache helper
# ============================================================


def _downsample_bool(mask: np.ndarray, target_hw: int) -> torch.Tensor:
    """Nearest-neighbor downsample a bool mask to (target_hw, target_hw).

    Returns a torch.bool tensor on CPU.
    """
    # Use torch.nn.functional.interpolate with nearest-exact to get stable
    # downsampling; work in uint8 so we don't fall back to float rounding.
    t = torch.from_numpy(mask.astype(np.uint8)).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    t = F.interpolate(t, size=(target_hw, target_hw), mode="nearest")
    return (t.squeeze(0).squeeze(0) > 0.5).to(torch.bool)


def _resize_bool(mask: np.ndarray, out_h: int, out_w: int) -> torch.Tensor:
    """Nearest-neighbor resize a bool mask to (out_h, out_w). CPU torch.bool."""
    t = torch.from_numpy(mask.astype(np.uint8)).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(out_h, out_w), mode="nearest")
    return (t.squeeze(0).squeeze(0) > 0.5).to(torch.bool)


def _apply_dataloader_transform(
    img,  # PIL.Image.Image in RGB
    file_item: 'FileItemDTO',
):
    """Mirror the image geometry used by ``load_and_process_image``.

    Subject extraction runs before workers fetch samples, so this helper must
    apply both bucket and nonbucket geometry before the mask is resized. The
    nonbucket path intentionally uses center geometry; random crops cannot be
    represented by one static per-file mask and should use bucketed datasets.
    """
    from PIL import Image as _Image
    from torchvision import transforms

    config = getattr(file_item, "dataset_config", None)
    is_bucketed = bool(getattr(config, "buckets", False))
    if getattr(file_item, 'flip_x', False):
        img = img.transpose(_Image.FLIP_LEFT_RIGHT)
    if getattr(file_item, 'flip_y', False):
        img = img.transpose(_Image.FLIP_TOP_BOTTOM)

    stw = getattr(file_item, 'scale_to_width', None)
    sth = getattr(file_item, 'scale_to_height', None)
    cx = getattr(file_item, 'crop_x', None)
    cy = getattr(file_item, 'crop_y', None)
    cw = getattr(file_item, 'crop_width', None)
    ch = getattr(file_item, 'crop_height', None)
    if is_bucketed and None not in (stw, sth, cx, cy, cw, ch):
        img = img.resize((int(stw), int(sth)), _Image.BICUBIC)
        return img.crop((int(cx), int(cy), int(cx) + int(cw), int(cy) + int(ch)))

    if config is None:
        return img
    scale = float(getattr(config, "scale", 1.0) or 1.0)
    img = img.resize(
        (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
        _Image.BICUBIC,
    )
    crop_size = min(img.size)
    if getattr(config, "random_crop", False):
        crop_size = min(crop_size, int(getattr(config, "resolution", crop_size)))
    img = transforms.CenterCrop(crop_size)(img)
    resolution = int(getattr(config, "resolution", crop_size))
    return img.resize((resolution, resolution), _Image.BICUBIC)


def _mask_output_hw(file_item: 'FileItemDTO', fallback_hw: int) -> tuple:
    """Return mask geometry matching the effective training representation."""
    config = getattr(file_item, "dataset_config", None)
    if not bool(getattr(config, "buckets", False)):
        return int(fallback_hw), int(fallback_hw)
    cw = getattr(file_item, 'crop_width', None)
    ch = getattr(file_item, 'crop_height', None)
    if cw is not None and ch is not None:
        return int(ch), int(cw)
    return int(fallback_hw), int(fallback_hw)

def _mask_cache_identity(file_item: 'FileItemDTO', config, out_h: int, out_w: int,
                         subject_reference: Optional[torch.Tensor] = None) -> str:
    """Hash every representation-changing source and transform field.

    Masks are extracted after the dataloader transform, so crop/flip identity
    must be part of the filename. A source-content digest prevents a
    same-size/same-mtime replacement from reusing an aligned-looking mask.
    Identity-matched masks also key on the match recipe, a digest of the
    reference embedding, and the YOLO detector settings that shape person
    isolation — re-running with a different subject, threshold, or detector
    cannot reuse the old subject/others split. Detector settings stay OUT of
    the non-matching recipe (base masks are pure SegFormer) so those caches
    don't churn on detector tweaks.
    """
    from toolkit.dataloader_mixins import content_fingerprint

    if subject_reference is not None:
        ref_digest = hashlib.sha256(
            subject_reference.detach().cpu().numpy().astype(np.float32).tobytes()
        ).hexdigest()[:16]
    else:
        ref_digest = ""
    source_digest = getattr(file_item, "source_content_fingerprint", None)
    if not source_digest:
        source_digest = content_fingerprint(file_item.path)
    identity = {
        "cache_version": CACHE_VERSION_KEY,
        "source": source_digest,
        "flip_x": bool(getattr(file_item, "flip_x", False)),
        "flip_y": bool(getattr(file_item, "flip_y", False)),
        "scale_to": [
            getattr(file_item, "scale_to_width", None),
            getattr(file_item, "scale_to_height", None),
        ],
        "crop": [
            getattr(file_item, "crop_x", None),
            getattr(file_item, "crop_y", None),
            getattr(file_item, "crop_width", None),
            getattr(file_item, "crop_height", None),
        ],
        "output": [int(out_h), int(out_w)],
        "config": {
            "buckets": bool(
                getattr(getattr(file_item, "dataset_config", None), "buckets", False)
            ),
            "scale": float(
                getattr(getattr(file_item, "dataset_config", None), "scale", 1.0)
            ),
            "random_crop": bool(
                getattr(getattr(file_item, "dataset_config", None), "random_crop", False)
            ),
            "random_scale": bool(
                getattr(getattr(file_item, "dataset_config", None), "random_scale", False)
            ),
            "resolution": int(
                getattr(getattr(file_item, "dataset_config", None), "resolution", 0)
            ),
            "cache_resolution": int(getattr(config, "cache_resolution", 0)),
            "body_close_radius": int(getattr(config, "body_close_radius", 0)),
            "mask_dilate_radius": int(getattr(config, "mask_dilate_radius", 0)),
            "skin_bias": float(getattr(config, "skin_bias", 0.0)),
            "primary_only": bool(getattr(config, "primary_only", True)),
            "sam_size": str(getattr(config, "sam_size", "")),
            "segformer_res": int(getattr(config, "segformer_res", 0)),
            "identity_matching": {
                "threshold": float(getattr(config, "match_threshold", 0.40)),
                "margin": float(getattr(config, "match_margin", 0.10)),
                "reference": ref_digest,
            },
        },
    }
    if ref_digest:
        # Matched masks depend on the YOLO detector (the subject's person box
        # or the fallback prompt derive from it); re-key when it changes.
        identity["config"]["identity_matching"]["yolo_conf"] = float(
            getattr(config, "yolo_conf", 0.25))
        identity["config"]["identity_matching"]["yolo_ckpt"] = str(
            getattr(config, "yolo_ckpt", ""))
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:24]


def cache_subject_masks(
    file_items: List['FileItemDTO'],
    config: 'SubjectMaskConfig',
    preview_dir: Optional[str] = None,
    subject_reference: Optional[torch.Tensor] = None,
) -> None:
    """Extract and cache subject masks for all file items.

    Caches each image's masks to:
        {image_dir}/_face_id_cache/{stem}_subject_masks_{H}x{W}_{identity}.safetensors
    where ``identity`` includes source content, effective flip/scale/crop,
    output geometry, mask extraction recipe, and (when identity matching is
    on) the match recipe + reference-embedding digest. This is one file per
    fully transformed representation, so same-shaped crops cannot alias.

    Keys: ``person``, ``body``, ``clothing`` (uint8 0/255), plus a version
    sentinel. When ``subject_reference`` is provided, additionally
    ``subject_person`` / ``others_person`` (uint8 0/255; zeros when no face
    matched — weights then no-op for that image) and the v4 sentinel.
        file_item.subject_mask  : torch.bool (H_c, W_c)
        file_item.body_mask     : torch.bool (H_c, W_c)
        file_item.clothing_mask : torch.bool (H_c, W_c)
        file_item.subject_person_mask / others_person_mask (matching only)
    where ``(H_c, W_c)`` is the effective transformed output geometry (bucket
    crop dimensions when available, otherwise ``cache_resolution`` square).

    Skips extraction if cache exists and has the matching version key.

    Args:
        preview_dir: If provided AND ``config.save_debug_previews`` is True, a
            preview ``{stem}.png`` tile is written here for each extracted image.
            Callers (SDTrainer) should pass the job output folder
            (``save_root/subject_mask_previews``) so previews don't end up
            inside the image dataset itself. If None, previews are skipped even
            when the flag is on.
        subject_reference: L2-normalized ArcFace reference embedding of the
            training subject (see :func:`build_subject_reference`). When given,
            each image's faces are matched against it and the cached
            subject/others masks isolate the matched person.
    """
    from PIL import Image
    from PIL.ImageOps import exif_transpose

    for file_item in file_items:
        dataset_config = getattr(file_item, "dataset_config", None)
        if (
            not bool(getattr(dataset_config, "buckets", False))
            and bool(getattr(dataset_config, "random_crop", False))
        ):
            raise ValueError(
                "Subject-mask caching with nonbucket random crops/scales is "
                "unsupported because one static mask cannot follow each "
                f"per-sample transform: {file_item.path}"
            )
    target_hw = int(config.cache_resolution)
    matching_on = subject_reference is not None
    version_key = CACHE_VERSION_KEY_MATCHED if matching_on else CACHE_VERSION_KEY
    required_keys = ['person', 'body', 'clothing']
    if matching_on:
        required_keys += ['subject_person', 'others_person']

    # Determine whether we can skip loading the extractor altogether (all cached).
    extractor: Optional[SubjectMaskExtractor] = None
    empty_count = 0
    cached_count = 0
    extracted_count = 0
    # Identity-match bookkeeping (extraction path only; hits replay cached results).
    match_multi_lines: List[str] = []
    match_failed_lines: List[str] = []
    match_single_count = 0
    match_unmatched_count = 0
    match_noface_count = 0
    match_isolation_failed_count = 0

    pbar = tqdm(file_items, desc="Caching subject masks")
    for file_item in pbar:
        img_dir = os.path.dirname(file_item.path)
        cache_dir = os.path.join(img_dir, '_face_id_cache')
        stem = os.path.splitext(os.path.basename(file_item.path))[0]
        # Per-bucket cache filename. With multi-resolution training the same
        # image lands in different bucket dims across datasets, so the cache
        # must be keyed by shape — otherwise the first dataset's mask shape
        # gets loaded for every other resolution's items and torch.stack fails
        # at collate time.
        out_h, out_w = _mask_output_hw(file_item, fallback_hw=target_hw)
        identity = _mask_cache_identity(file_item, config, out_h, out_w,
                                        subject_reference=subject_reference)
        cache_path = os.path.join(
            cache_dir,
            f'{stem}_subject_masks_{out_h}x{out_w}_{identity}.safetensors',
        )

        # ------------------------------------------------------------- cache hit
        if os.path.exists(cache_path):
            try:
                data = load_file(cache_path)
            except Exception:
                data = {}
            has_keys = all(k in data for k in required_keys)
            has_version = version_key in data
            # Body smoothing radius is baked into the cached masks — invalidate
            # when the config value changes so users can iterate on it.
            cached_bcr = int(data['body_close_radius'].item()) if 'body_close_radius' in data else 2
            radius_match = cached_bcr == int(config.body_close_radius)
            if has_keys and has_version and radius_match:
                person = (data['person'].clone() > 127).to(torch.bool)
                body = (data['body'].clone() > 127).to(torch.bool)
                clothing = (data['clothing'].clone() > 127).to(torch.bool)
                file_item.subject_mask = person
                file_item.body_mask = body
                file_item.clothing_mask = clothing
                if matching_on:
                    file_item.subject_person_mask = (data['subject_person'].clone() > 127).to(torch.bool)
                    file_item.others_person_mask = (data['others_person'].clone() > 127).to(torch.bool)

                # Upfront QC preview: render from cache if missing on disk.
                # Skips when the .png already exists so re-runs are free.
                if getattr(config, 'save_debug_previews', False) and preview_dir:
                    os.makedirs(preview_dir, exist_ok=True)
                    preview_path = os.path.join(preview_dir, f'{stem}.png')
                    if not os.path.exists(preview_path):
                        try:
                            raw_pil = exif_transpose(Image.open(file_item.path)).convert('RGB')
                            pil_image = _apply_dataloader_transform(raw_pil, file_item)
                            tile = _render_preview_tile_from_cache(
                                pil_image, person, body, clothing,
                                subject_t=file_item.subject_person_mask if matching_on else None,
                                others_t=file_item.others_person_mask if matching_on else None,
                            )
                            tile.save(preview_path)
                        except Exception as e:
                            print(f"  -  Warning: failed to render preview for {stem}: {e}")
                cached_count += 1
                pbar.set_postfix(hit=cached_count, miss=extracted_count)
                continue  # cache hit — no need to run models

        # ------------------------------------------------------------- cache miss
        if extractor is None:
            extractor = SubjectMaskExtractor(config)

        # v3: extract masks from the *dataloader-transformed* pixels so cached
        # masks align with the training tensor (and thus latent grid). Applies
        # the same flip → resize → crop chain as
        # toolkit/dataloader_mixins.load_and_process_image (lines 774-793).
        raw_pil = exif_transpose(Image.open(file_item.path)).convert('RGB')
        pil_image = _apply_dataloader_transform(raw_pil, file_item)
        masks = extractor.extract(pil_image, subject_reference=subject_reference)

        # Cache at training-tensor dimensions (crop_w, crop_h) when known.
        # No square downsample — preserves aspect ratio so F.interpolate to
        # the latent grid at training time is a straight resize.
        # (out_h, out_w computed at the top of the loop for the cache path.)
        person_t = _resize_bool(masks['person'], out_h, out_w)
        body_t = _resize_bool(masks['body'], out_h, out_w)
        clothing_t = _resize_bool(masks['clothing'], out_h, out_w)

        # Identity-matched split: subject region vs everyone else. A failed or
        # faceless match leaves BOTH masks empty so the loss weights no-op for
        # that image instead of guessing.
        subject_np: Optional[np.ndarray] = None
        others_np: Optional[np.ndarray] = None
        if matching_on:
            match = masks.get('subject_match')
            if match is not None and match['matched']:
                if match['region'] is None:
                    subject_np = masks['person']
                else:
                    subject_np = masks['person'] & match['region']
                others_np = masks['person'] & ~subject_np
                if match['region'] is not None:
                    match_multi_lines.append(
                        f"{stem}: {match['face_count']} faces, subject cos="
                        f"{match['best_cos']:.2f} (2nd {match['second_cos']:.2f})"
                        f" [{match.get('isolation', 'sam')}]"
                    )
                else:
                    match_single_count += 1
            else:
                if match is not None and match.get('isolation') == 'failed':
                    # Cosine-accepted but un-isolatable: both masks stay empty
                    # so neither identity weight applies to a guessed person.
                    match_isolation_failed_count += 1
                    match_failed_lines.append(
                        f"{stem}: isolation FAILED (face cover "
                        f"{match.get('face_cover', 0.0):.2f}) -> identity weights no-op"
                    )
                elif match is None or match['face_count'] == 0:
                    match_noface_count += 1
                else:
                    match_unmatched_count += 1
                    match_failed_lines.append(
                        f"{stem}: {match['face_count']} faces, best cos="
                        f"{match['best_cos']:.2f} (2nd {match['second_cos']:.2f})"
                    )
        subject_t = _resize_bool(subject_np, out_h, out_w) if subject_np is not None \
            else torch.zeros((out_h, out_w), dtype=torch.bool)
        others_t = _resize_bool(others_np, out_h, out_w) if others_np is not None \
            else torch.zeros((out_h, out_w), dtype=torch.bool)

        if not person_t.any():
            empty_count += 1

        file_item.subject_mask = person_t
        file_item.body_mask = body_t
        file_item.clothing_mask = clothing_t
        if matching_on:
            file_item.subject_person_mask = subject_t
            file_item.others_person_mask = others_t

        os.makedirs(cache_dir, exist_ok=True)
        save_data = {
            'person': (person_t.to(torch.uint8) * 255),
            'body': (body_t.to(torch.uint8) * 255),
            'clothing': (clothing_t.to(torch.uint8) * 255),
            'body_close_radius': torch.tensor([float(config.body_close_radius)]),
            version_key: torch.ones(1),
        }
        if matching_on:
            save_data['subject_person'] = (subject_t.to(torch.uint8) * 255)
            save_data['others_person'] = (others_t.to(torch.uint8) * 255)
        save_file(save_data, cache_path)
        extracted_count += 1
        pbar.set_postfix(hit=cached_count, miss=extracted_count)

        # Optional: write a preview tile for visual inspection.
        # Only if both the flag is on AND a target directory was provided —
        # we never write inside the image dataset, so the caller must decide
        # where previews live.
        if getattr(config, 'save_debug_previews', False) and preview_dir:
            os.makedirs(preview_dir, exist_ok=True)
            preview_path = os.path.join(preview_dir, f'{stem}.png')
            try:
                tile = _render_preview_tile(
                    pil_image, masks,
                    n_classes=extractor.seg_cfg.num_labels,
                    subject_person=(subject_np.astype(np.uint8) if subject_np is not None
                                    else (np.zeros(masks['person'].shape, dtype=np.uint8)
                                          if matching_on else None)),
                    others_person=(others_np.astype(np.uint8) if others_np is not None
                                   else (np.zeros(masks['person'].shape, dtype=np.uint8)
                                         if matching_on else None)),
                )
                tile.save(preview_path)
            except Exception as e:
                # Preview failures are non-fatal — the cache is the real artifact.
                print(f"  -  Warning: failed to render preview for {stem}: {e}")

    # Free VRAM held by models (only loaded if we had cache misses)
    if extractor is not None:
        extractor.cleanup()
        del extractor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(
        f"  -  Subject masks: {cached_count} cache hit"
        f"{'s' if cached_count != 1 else ''}, "
        f"{extracted_count} extracted"
    )
    if empty_count > 0:
        print(f"  -  Warning: empty subject mask for {empty_count}/{len(file_items)} images")
    if matching_on:
        total_matched = match_single_count + len(match_multi_lines)
        print(
            f"  -  Identity matching: {total_matched} matched "
            f"({match_single_count} whole-person, {len(match_multi_lines)} isolated), "
            f"{match_unmatched_count} unmatched, {match_noface_count} no-face"
            + (f", {match_isolation_failed_count} isolation-failed"
               if match_isolation_failed_count else "")
        )
        for line in match_multi_lines:
            print(f"    - {line}")
        for line in match_failed_lines:
            print(f"    - WARNING no subject match: {line}")
