"""Lightweight toolkit package namespace.

Admission is imported by ``python -m toolkit.admission`` before any training
dependencies are needed. Keep this module free of third-party imports; the
Hugging Face progress hook is exposed lazily for the real training bootstrap.
"""
import importlib
import logging
import os

# No telemetry.  onnxruntime's Linux builds from 1.29 send a device ID and
# machine details to Microsoft unless ORT_DISABLE_TELEMETRY=1 (it is read when
# onnxruntime first imports — always later than this package, whose modules
# import it lazily for face_id / pose controls); Hugging Face usage reporting
# goes off with HF_HUB_DISABLE_TELEMETRY=1.  An existing value always wins.
# Mirrors Fizgig v9.2.0 (fe07178c).
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def force_hf_hub_progress_bars() -> None:
    """Keep Hub download bars visible when training logs are not a TTY."""
    hf_tqdm = importlib.import_module("huggingface_hub.utils.tqdm")
    original_is_tqdm_disabled = hf_tqdm.is_tqdm_disabled
    if getattr(original_is_tqdm_disabled, "_aitk_forced_progress", False):
        return

    def is_tqdm_disabled(log_level):
        disabled = original_is_tqdm_disabled(log_level)
        # UI jobs log to files, so stderr is not a TTY. Keep HF download bars
        # visible while preserving normal carriage-return progress updates.
        if disabled is None and log_level != logging.NOTSET:
            return False
        return disabled

    is_tqdm_disabled._aitk_forced_progress = True
    hf_tqdm.is_tqdm_disabled = is_tqdm_disabled
