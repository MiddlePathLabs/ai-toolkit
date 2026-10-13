"""Model file discovery and status reporting.

Nothing in this module ever downloads weights. Every model is looked up on
disk and reported as available or missing with the expected location, so the
UI can tell the user exactly what to place where.

- ArcFace stack: SCRFD ``det_10g.onnx`` + ``w600k_r50.onnx`` from the
  insightface home (``~/.insightface/models/buffalo_l``). InsightFace
  weights are non-commercial research terms -- fork-only, never bundled.
- SFace: ``face_recognition_sface_2021dec.onnx`` (opencv_zoo, Apache-2.0),
  user-supplied in the identity models dir (default ``models/face``).
- YuNet (optional, SFace-only mode detection): ``face_detection_yunet_2023mar.onnx``
  (opencv_zoo, MIT), same dir.
"""

import hashlib
import os
from dataclasses import dataclass, field
from typing import Optional

ARCFACE_DIR_NAME = 'buffalo_l'
ARCFACE_DET_FILE = 'det_10g.onnx'
ARCFACE_REC_FILE = 'w600k_r50.onnx'
SFACE_FILE = 'face_recognition_sface_2021dec.onnx'
YUNET_FILE = 'face_detection_yunet_2023mar.onnx'

# Extensions accepted for user-supplied reference images (mirrors the
# reference-upload route, which re-encodes everything to .jpg anyway).
REF_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp')

_SHA_CACHE: dict = {}


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    """Content hash of a file, memoized by (path, mtime, size)."""
    try:
        stat = os.stat(path)
        cache_key = (path, stat.st_mtime_ns, stat.st_size)
    except OSError:
        return hashlib.sha256(path.encode('utf-8')).hexdigest()
    cached = _SHA_CACHE.get(cache_key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    value = digest.hexdigest()
    _SHA_CACHE[cache_key] = value
    return value


@dataclass
class ModelStatus:
    name: str
    available: bool
    path: Optional[str] = None
    file_hash: Optional[str] = None
    reason: Optional[str] = None
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        out = {'name': self.name, 'available': self.available, 'reason': self.reason, 'notes': self.notes}
        if self.available:
            out['path'] = self.path
            out['hash'] = self.file_hash
        return out


def find_arcface(insightface_root: Optional[str] = None) -> ModelStatus:
    """Locate the buffalo_l SCRFD + ArcFace pair without downloading.

    ``insightface.models.buffalo_l`` is the default insightface home
    layout; ``insightface_root`` lets callers (and tests) point elsewhere.
    """
    root = insightface_root or os.path.join(os.path.expanduser('~'), '.insightface')
    model_dir = os.path.join(root, 'models', ARCFACE_DIR_NAME)
    missing = [name for name in (ARCFACE_DET_FILE, ARCFACE_REC_FILE) if not os.path.isfile(os.path.join(model_dir, name))]
    if missing:
        return ModelStatus(
            name='arcface',
            available=False,
            path=model_dir,
            reason=f'missing {", ".join(missing)} in {model_dir}. ArcFace (insightface buffalo_l) '
            'weights are not bundled; install them per docs/identity_eval.md.',
        )
    return ModelStatus(
        name='arcface',
        available=True,
        path=model_dir,
        file_hash=sha256_file(os.path.join(model_dir, ARCFACE_REC_FILE)),
    )


def find_sface(models_dir: str) -> ModelStatus:
    path = os.path.join(models_dir, SFACE_FILE)
    if not os.path.isfile(path):
        return ModelStatus(
            name='sface',
            available=False,
            path=path,
            reason=f'{SFACE_FILE} not found in {models_dir}. SFace (opencv_zoo, Apache-2.0) '
            'is user-supplied; see docs/identity_eval.md.',
        )
    return ModelStatus(name='sface', available=True, path=path, file_hash=sha256_file(path))


def find_yunet(models_dir: str) -> ModelStatus:
    path = os.path.join(models_dir, YUNET_FILE)
    if not os.path.isfile(path):
        return ModelStatus(
            name='yunet',
            available=False,
            path=path,
            reason=f'{YUNET_FILE} not found in {models_dir} (optional; only needed for '
            'SFace-only mode when buffalo_l is absent).',
        )
    return ModelStatus(name='yunet', available=True, path=path, file_hash=sha256_file(path))
