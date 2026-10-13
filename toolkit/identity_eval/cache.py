"""Content-addressed embedding cache under ``output/<job>/identity_eval/emb_cache``.

One JSON file per image, keyed by the image content hash plus the detector
signature (detector weights hash + det size + face-order version). Each
face keeps its detection geometry and per-model embeddings stored under
that model's own signature, so swapping one model file (or running
ArcFace-only before SFace is installed) reuses everything else and only
recomputes what changed.
"""

import hashlib
import json
import os
from typing import Optional

import numpy as np

from .detector import DetectionResult

CACHE_VERSION = 1


def _atomic_write_json(path: str, payload) -> None:
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle)
    os.replace(tmp, path)


class EmbeddingCache:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _entry_path(self, image_sha: str, det_sig: str) -> str:
        key = hashlib.sha256(f'{image_sha}|{det_sig}'.encode('utf-8')).hexdigest()
        return os.path.join(self.root, f'{key}.json')

    def load(self, image_sha: str, det_sig: str) -> Optional[dict]:
        """Load a cache entry, or None when missing / stale / unreadable."""
        path = self._entry_path(image_sha, det_sig)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                entry = json.load(handle)
        except (OSError, ValueError):
            return None
        if not isinstance(entry, dict):
            return None
        if entry.get('version') != CACHE_VERSION or entry.get('det_sig') != det_sig or entry.get('image_sha') != image_sha:
            return None
        return entry

    def save(self, entry: dict) -> None:
        _atomic_write_json(self._entry_path(entry['image_sha'], entry['det_sig']), entry)

    # -- entry construction -----------------------------------------------------

    @staticmethod
    def entry_from_detection(image_sha: str, det_sig: str, detection: DetectionResult) -> dict:
        return {
            'version': CACHE_VERSION,
            'image_sha': image_sha,
            'det_sig': det_sig,
            'faces': [face.to_dict() for face in detection.faces],
        }

    @staticmethod
    def get_embedding(face_entry: dict, model_sig: str) -> Optional[np.ndarray]:
        embeddings = face_entry.get('embeddings') or {}
        vec = embeddings.get(model_sig)
        if vec is None:
            return None
        arr = np.asarray(vec, dtype=np.float64)
        if arr.size == 0:
            return None
        return arr

    @staticmethod
    def set_embedding(face_entry: dict, model_sig: str, embedding: np.ndarray) -> None:
        face_entry.setdefault('embeddings', {})[model_sig] = [round(float(v), 7) for v in embedding]
