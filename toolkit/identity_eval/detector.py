"""Face detection for identity evaluation.

Detection runs on the raw ``cv2.imread`` output (BGR). The legacy eval
scripts fed RGB (channel-swapped) images to insightface; insightface
expects BGR and swaps internally (``blobFromImage(..., swapRB=True)``), so
BGR passthrough is the correct input and gives ~9% more same/impostor
separation (see docs/identity_eval.md). A regression test pins this.

Faces are ordered by bbox area, descending. That order defines the stable
``index`` used by face picks in ``selection.json``; it never auto-selects
a subject for scoring (the scripts' biggest-face shortcut is deliberately
not copied).
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .models import ARCFACE_DET_FILE, find_arcface, find_yunet, sha256_file

DET_SIZE = (640, 640)
DET_THRESHOLD = 0.5
# Bumped when the face ordering / caching format changes so cache entries
# are not silently reused across incompatible orders.
FACE_ORDER_VERSION = 2

# CUDA ORT arena cap: enough for SCRFD/ArcFace inference while never
# starving a training job on the same GPU.
CUDA_ARENA_LIMIT_BYTES = 2 * 1024 * 1024 * 1024


def ort_cuda_providers(gpu_id: int = 0) -> list:
    """CUDA-first provider list capped to a small arena, targeting the
    policy-selected GPU (must match the GPU free VRAM was checked on)."""
    cuda_options = {
        'device_id': int(gpu_id),
        'gpu_mem_limit': CUDA_ARENA_LIMIT_BYTES,
        'arena_extend_strategy': 'kSameAsRequested',
    }
    return [('CUDAExecutionProvider', cuda_options), 'CPUExecutionProvider']


@dataclass
class DetectedFace:
    index: int
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    kps: np.ndarray  # (5, 2)
    det_score: float

    def area(self) -> float:
        return max(0.0, (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1]))

    def to_dict(self) -> dict:
        return {
            'index': self.index,
            'bbox': [round(float(v), 2) for v in self.bbox],
            'kps': [[round(float(x), 2), round(float(y), 2)] for x, y in self.kps],
            'det_score': round(float(self.det_score), 4),
        }


@dataclass
class DetectionResult:
    faces: List[DetectedFace] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.faces)


class FaceDetector:
    """SCRFD (buffalo_l) detection, falling back to YuNet when buffalo_l
    is absent so SFace-only mode still works. Both backends receive BGR."""

    def __init__(
        self,
        device: str = 'cpu',
        det_size: Tuple[int, int] = DET_SIZE,
        insightface_root: Optional[str] = None,
        models_dir: Optional[str] = None,
        gpu_id: int = 0,
    ):
        self.device = device
        self.gpu_id = int(gpu_id)
        self.det_size = tuple(int(v) for v in det_size)
        self._arcface_status = find_arcface(insightface_root)
        self._yunet_status = find_yunet(models_dir) if models_dir else find_yunet(os.path.join(os.getcwd(), 'models', 'face'))
        self.backend = None
        self._det = None
        self.det_error: Optional[str] = None

        if self._arcface_status.available:
            try:
                self._init_scrfd()
                self.backend = 'scrfd'
            except Exception as e:  # noqa: BLE001 -- fall through to YuNet
                self.det_error = f'SCRFD load failed: {e}'
        if self.backend is None and self._yunet_status.available:
            try:
                self._init_yunet()
                self.backend = 'yunet'
                if self.det_error:
                    # keep the SCRFD failure visible alongside the fallback
                    self._yunet_status.notes.append(self.det_error)
            except Exception as e:  # noqa: BLE001
                self.det_error = (self.det_error or '') + f' YuNet load failed: {e}'
        if self.backend is None:
            reason = self.det_error or 'no detector available: install buffalo_l or YuNet (docs/identity_eval.md)'
            raise RuntimeError(reason)

    # -- backend construction ------------------------------------------------

    def _ort_providers(self):
        if self.device == 'cuda':
            return ort_cuda_providers(self.gpu_id)
        return ['CPUExecutionProvider']

    def _init_scrfd(self):
        from insightface.model_zoo import model_zoo

        det_path = os.path.join(self._arcface_status.path, ARCFACE_DET_FILE)
        self._det = model_zoo.get_model(
            det_path,
            providers=self._ort_providers(),
        )
        ctx_id = 0 if self.device == 'cuda' else -1
        self._det.prepare(ctx_id, input_size=self.det_size, det_thresh=DET_THRESHOLD)

    def _init_yunet(self):
        import cv2

        self._det = cv2.FaceDetectorYN.create(
            str(self._yunet_status.path),
            '',
            self.det_size,
            score_threshold=DET_THRESHOLD,
        )

    # -- signature -------------------------------------------------------------

    @property
    def det_sig(self) -> str:
        """Cache signature: detector weights + det size + face order logic."""
        if self.backend == 'scrfd':
            model_hash = sha256_file(os.path.join(self._arcface_status.path, ARCFACE_DET_FILE))
            tag = f'scrfd:{model_hash}'
        else:
            tag = f'yunet:{self._yunet_status.file_hash}'
        return f'v{FACE_ORDER_VERSION}:{tag}:{self.det_size[0]}x{self.det_size[1]}'

    # -- detection ---------------------------------------------------------------

    def detect(self, img_bgr: np.ndarray) -> DetectionResult:
        """Detect every face in a BGR image, sorted by area (desc)."""
        if self.backend == 'scrfd':
            bboxes, kpss = self._det.detect(img_bgr, max_num=0, metric='default')
            faces = []
            if bboxes is not None and len(bboxes):
                order = sorted(range(len(bboxes)), key=lambda i: -_area(bboxes[i, 0:4]))
                for index, i in enumerate(order):
                    faces.append(
                        DetectedFace(
                            index=index,
                            bbox=tuple(float(v) for v in bboxes[i, 0:4]),
                            kps=np.asarray(kpss[i], dtype=np.float32) if kpss is not None else np.zeros((5, 2), dtype=np.float32),
                            det_score=float(bboxes[i, 4]),
                        )
                    )
            return DetectionResult(faces=faces)

        import cv2

        ok, rows = self._det.detect(img_bgr)
        faces = []
        if ok and rows is not None and len(rows):
            order = sorted(range(len(rows)), key=lambda i: -_area(rows[i, 0:4]))
            for index, i in enumerate(order):
                row = rows[i]
                x, y, w, h = (float(v) for v in row[0:4])
                kps = np.asarray(row[4:14], dtype=np.float32).reshape(5, 2)
                faces.append(DetectedFace(index=index, bbox=(x, y, x + w, y + h), kps=kps, det_score=float(row[14])))
        return DetectionResult(faces=faces)


def _area(bbox) -> float:
    return max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
