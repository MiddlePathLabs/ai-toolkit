"""SFace (opencv_zoo, Apache-2.0) embedding via OpenCV DNN.

Runs on CPU by default; the CUDA backend is used only when the installed
OpenCV build actually has it (pip ``opencv-python`` wheels usually do not
-- checked, not assumed).

Landmark order: both SCRFD and YuNet use the person-relative order
[right eye, left eye, nose, right mouth corner, left mouth corner] (for a
non-mirrored photo: image-left eye first). The conversion below is a
straight passthrough of the SCRFD landmarks into the YuNet row; a
real-model regression test asserts the resulting ``alignCrop`` matches
insightface's ``norm_crop`` geometry (see
testing/integration/test_identity_eval.py).
"""

from typing import Optional

import cv2
import numpy as np

from .detector import DetectedFace
from .models import find_sface

SFACE_DIM = 128


def yunet_row(face: DetectedFace) -> np.ndarray:
    """15-element YuNet-format row from a detected face.

    ``[x, y, w, h, re_x, re_y, le_x, le_y, n_x, n_y, rm_x, rm_y, lm_x, lm_y, score]``
    """
    x1, y1, x2, y2 = face.bbox
    kps = np.asarray(face.kps, dtype=np.float32).reshape(5, 2)
    return np.array(
        [x1, y1, x2 - x1, y2 - y1, *kps.flatten(), face.det_score],
        dtype=np.float32,
    )


def cuda_target_available() -> bool:
    try:
        targets = cv2.dnn.getAvailableTargets(cv2.dnn.DNN_BACKEND_OPENCV)
        return int(cv2.dnn.DNN_TARGET_CUDA) in [int(t) for t in targets]
    except Exception:  # noqa: BLE001 -- CPU-only builds may not expose CUDA symbols
        return False


class SFaceEmbedder:
    def __init__(self, models_dir: str, device: str = 'cpu', gpu_id: int = 0):
        self.status = find_sface(models_dir)
        if not self.status.available:
            raise RuntimeError(self.status.reason)
        self.device = 'cpu'
        if device == 'cuda' and gpu_id != 0:
            # OpenCV DNN's CUDA target has no per-device selection and always
            # uses device 0; never silently evaluate on an unchecked GPU.
            self.status.notes.append(
                f'OpenCV DNN CUDA always targets device 0; SFace runs on CPU (selected GPU is {gpu_id})'
            )
        elif device == 'cuda' and cuda_target_available():
            self.device = 'cuda'
        elif device == 'cuda':
            self.status.notes.append('CUDA DNN target unavailable in this OpenCV build; SFace runs on CPU')
        backend_id = cv2.dnn.DNN_BACKEND_OPENCV
        target_id = cv2.dnn.DNN_TARGET_CUDA if self.device == 'cuda' else cv2.dnn.DNN_TARGET_CPU
        self._recognizer = cv2.FaceRecognizerSF.create(str(self.status.path), '', backend_id, target_id)
        self.dim = SFACE_DIM

    @property
    def model_sig(self) -> str:
        # Weights-hash only: the embedding model itself is device-independent
        # for cache purposes, and a device suffix would needlessly invalidate
        # embeddings when auto mode flips between CUDA and CPU.
        return f'sface:{self.status.file_hash}'

    def embed(self, img_bgr: np.ndarray, face: DetectedFace) -> np.ndarray:
        """Aligned, L2-normalized embedding for one detected face."""
        aligned = self._recognizer.alignCrop(img_bgr, yunet_row(face))
        feature = np.asarray(self._recognizer.feature(aligned), dtype=np.float64).flatten()
        # some OpenCV builds (4.11 windows wheel) omit the normL2 binding;
        # plain L2 normalization is the same math
        norm = np.linalg.norm(feature)
        if norm > 0:
            feature = feature / norm
        return feature
