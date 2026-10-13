"""Scoring math for identity evaluation.

Mirrors the mean-of-per-reference-cosine method used by the existing
ad-hoc arm evaluation scripts so numbers sit on a familiar scale, with one
fix: raw cosines only (never percentages) and no cross-model averaging.
ArcFace and SFace scores are never combined or compared numerically; their
scales differ.

Conventions:
- ``R`` is an (n, d) matrix of L2-normalized reference embeddings, one row
  per reference face (per model).
- ``g`` is a normalized sample-face embedding.
- primary  = mean(R @ g)   -- mean of per-reference raw cosines
- max      = max(R @ g)    -- best single reference (near-copy hint)
- ref band = LOO           -- mean over references of each reference's mean
  cosine to the other references; a same-person baseline for the primary,
  not an upper bound (a generated embedding near the reference mean can
  score above it).
"""

import math
import os
import re
from typing import Optional

import numpy as np

# Sample filenames: <ts>__<step:09d>_<prompt:02d>_seed-<n>_steps-<n>.<ext>
# with an optional RAW_ prefix (dual-save raw/EMA weights).
SAMPLE_FILENAME_RE = re.compile(
    r'^(RAW_)?(\d+)__(\d+)_(\d+)(?:_seed-(-?\d+)_steps-(\d+))?\.[^.]+$'
)


def parse_sample_filename(path: str) -> Optional[dict]:
    """Extract step / prompt index / seed from a sample filename."""
    filename = os.path.basename(path.replace('\\', '/'))
    match = SAMPLE_FILENAME_RE.match(filename)
    if not match:
        return None
    try:
        step = int(match.group(3))
        prompt_idx = int(match.group(4))
        seed = int(match.group(5)) if match.group(5) is not None else None
    except ValueError:
        return None
    return {
        'filename': filename,
        'step': step,
        'prompt_idx': prompt_idx,
        'seed': seed,
        'is_raw': bool(match.group(1)),
    }


def mean_cosine(refs: np.ndarray, embedding: np.ndarray) -> float:
    """Primary score: mean of per-reference raw cosines."""
    sims = refs @ embedding
    return float(np.mean(sims))


def max_cosine(refs: np.ndarray, embedding: np.ndarray) -> float:
    """Secondary score: best single-reference cosine."""
    return float(np.max(refs @ embedding))


def loo_band(refs: np.ndarray) -> Optional[float]:
    """Reference leave-one-out band.

    Mean over references of each reference's mean cosine to all other
    references. With fewer than two references there is no band.
    """
    if refs is None or len(refs) < 2:
        return None
    sims = refs @ refs.T
    n = sims.shape[0]
    total = float(np.sum(sims) - np.trace(sims))
    return total / (n * (n - 1))


def round3(value: Optional[float]) -> Optional[float]:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return None
    return round(float(value), 3)


def score_face(refs: Optional[np.ndarray], embedding: np.ndarray) -> dict:
    """Primary + max scores against one reference set; None when no refs."""
    if refs is None or len(refs) == 0:
        return {'mean': None, 'max': None}
    return {
        'mean': round3(mean_cosine(refs, embedding)),
        'max': round3(max_cosine(refs, embedding)),
    }
