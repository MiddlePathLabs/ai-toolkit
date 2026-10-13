"""ArcFace (w600k_r50) embedding via insightface.

Loads only the recognition model from the buffalo_l directory (detection
is handled by :mod:`toolkit.identity_eval.detector`). Images must be BGR
(``cv2.imread`` output); ``ArcFaceONNX`` swaps to RGB internally via
``blobFromImage(..., swapRB=True)``.
"""

import os
from typing import Optional

import numpy as np

from .detector import DetectedFace
from .models import ARCFACE_REC_FILE, find_arcface

ARCFACE_DIM = 512


class ArcFaceEmbedder:
    def __init__(self, device: str = 'cpu', insightface_root: Optional[str] = None):
        self.status = find_arcface(insightface_root)
        if not self.status.available:
            raise RuntimeError(self.status.reason)
        from insightface.model_zoo import model_zoo

        providers = ['CPUExecutionProvider']
        if device == 'cuda':
            cuda_options = {
                'device_id': 0,
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                'arena_extend_strategy': 'kSameAsRequested',
            }
            providers = [('CUDAExecutionProvider', cuda_options), 'CPUExecutionProvider']
        self._model = model_zoo.get_model(os.path.join(self.status.path, ARCFACE_REC_FILE), providers=providers)
        ctx_id = 0 if device == 'cuda' else -1
        self._model.prepare(ctx_id)
        self.dim = int(np.prod(self._model.output_shape[1:]))

    @property
    def model_sig(self) -> str:
        return f'arcface:{self.status.file_hash}'

    def embed(self, img_bgr: np.ndarray, face: DetectedFace) -> np.ndarray:
        """Aligned, L2-normalized embedding for one detected face."""
        aimg = _norm_crop(img_bgr, face.kps, image_size=self._model.input_size[0])
        embedding = self._model.get_feat(aimg).flatten().astype(np.float64)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding


def _norm_crop(img, landmark, image_size=112):
    """insightface's arcface-template alignment (BGR in, BGR out)."""
    from insightface.utils import face_align

    return face_align.norm_crop(img, landmark=np.asarray(landmark, dtype=np.float32), image_size=int(image_size))
