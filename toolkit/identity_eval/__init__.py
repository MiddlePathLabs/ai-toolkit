"""Identity evaluation for generated samples (fork-only feature).

Scores sample faces against reference faces with two independent models
(ArcFace via insightface buffalo_l, SFace via OpenCV DNN). All recognition
logic lives here; ``ui_scripts/identity_eval.py`` is a thin CLI over this
module so the same code can later replace the ad-hoc eval scripts.

Licensing: model weights are user-supplied and never downloaded by this
code. See ``docs/identity_eval.md`` for the paths, licenses and the
SFace-only permissive mode.
"""

from .cache import EmbeddingCache
from .checkpoint_compare import (
    build_checkpoint_report,
    convergence_summary,
    extract_prompt_texts,
    load_categories,
)
from .detector import DetectedFace, FaceDetector
from .device import choose_device, resolve_device
from .drift import detect_drift
from .refs import (
    discover_training_refs,
    load_selection,
    save_selection,
)
from .runner import IdentityEvalRunner
from .scoring import (
    loo_band,
    max_cosine,
    mean_cosine,
    parse_sample_filename,
)

__all__ = [
    'DetectedFace',
    'EmbeddingCache',
    'FaceDetector',
    'IdentityEvalRunner',
    'build_checkpoint_report',
    'choose_device',
    'convergence_summary',
    'detect_drift',
    'discover_training_refs',
    'extract_prompt_texts',
    'load_categories',
    'load_selection',
    'loo_band',
    'max_cosine',
    'mean_cosine',
    'parse_sample_filename',
    'resolve_device',
    'save_selection',
]
