"""Reference discovery and selection persistence.

Training references are discovered from the job's stored config: datasets
with ``is_reg``/``prior_reg`` false, using ``dataset_path`` when set (the
trainer's precedence) else ``folder_path``. Control and mask folders are
excluded, mirroring the dataloader's image extensions rather than the
legacy scripts' ``*.jpg`` glob.

Selections live in ``output/<job>/identity_eval/selection.json``: per-image
exclusion toggles and explicit face picks for multi-face references (a
multi-face reference never enters the reference set without one).
"""

import json
import os
import re
from typing import List, Optional

# Mirrors toolkit.data_loader.image_extensions (the dataloader accepts more
# than .jpg; the eval scripts' *.jpg glob was a known gap).
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp')

SELECTION_VERSION = 1

_CONTROL_OR_MASK_KEY = re.compile(r'^(control_path|mask_path)(_\d+)?$')


def canonical_ref_key(path: str) -> str:
    """Canonical selection key for a training-ref path.

    Contract shared with the UI (``normalizeRefPath`` in
    ``ui/src/utils/identityEval.ts``): lowercase, forward slashes,
    normalized. Selection writes, UI reads and worker lookups all use
    this exact form so exclusions and face picks survive the round-trip
    on case-insensitive (Windows) and case-sensitive platforms alike.
    """
    return os.path.normcase(os.path.normpath(path)).replace(os.sep, '/')


def _normalize(path: str) -> str:
    return canonical_ref_key(path)


def _iter_dataset_images(folder: str) -> List[str]:
    found: List[str] = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for name in files:
            if name.startswith('.'):
                continue
            if name.lower().endswith(IMAGE_EXTENSIONS):
                found.append(os.path.join(root, name))
    return sorted(found, key=_normalize)


def discover_training_refs(job_config: Optional[dict]) -> dict:
    """Enumerate training-reference images from a stored job config.

    Accepts either the UI envelope ``{"job": ..., "config": {...}}`` or the
    config object itself. Returns per-dataset info, the flat image list and
    dataset-level warnings (never raises on bad input).
    """
    out = {'datasets': [], 'images': [], 'warnings': []}
    if not isinstance(job_config, dict):
        out['warnings'].append('job config is missing or not an object')
        return out
    config = job_config.get('config', job_config)
    processes = config.get('process') if isinstance(config, dict) else None
    if not processes or not isinstance(processes, list):
        out['warnings'].append('job config has no process list')
        return out
    datasets = processes[0].get('datasets') if isinstance(processes[0], dict) else None
    if not datasets:
        out['warnings'].append('job config has no datasets; nothing to discover')
        return out

    excluded_dirs = set()
    all_images = []
    for dataset in datasets:
        if not isinstance(dataset, dict):
            continue
        name = dataset.get('dataset_path') or dataset.get('name') or dataset.get('folder_path') or '<unnamed>'
        entry = {'dataset': name, 'path': None, 'images': 0, 'skipped': None, 'warnings': []}
        if dataset.get('is_reg'):
            entry['skipped'] = 'is_reg'
            out['datasets'].append(entry)
            continue
        if dataset.get('prior_reg'):
            entry['skipped'] = 'prior_reg'
            out['datasets'].append(entry)
            continue
        folder_path = dataset.get('folder_path')
        dataset_path = dataset.get('dataset_path')
        if folder_path and dataset_path and _normalize(folder_path) != _normalize(dataset_path):
            entry['warnings'].append(
                f"dataset has both folder_path ('{folder_path}') and dataset_path ('{dataset_path}') "
                'pointing at different paths; training uses dataset_path and would ignore folder_path'
            )
        path = dataset_path or folder_path
        if not path:
            entry['skipped'] = 'no path'
            out['datasets'].append(entry)
            continue
        path = os.path.abspath(path)
        for key, value in dataset.items():
            if _CONTROL_OR_MASK_KEY.match(key) and isinstance(value, str) and value.strip():
                excluded_dirs.add(_normalize(value))
        if path.lower().endswith('.json'):
            entry['warnings'].append(
                f'json dataset_path is not supported for identity references: {path}'
            )
            out['datasets'].append(entry)
            continue
        if not os.path.isdir(path):
            entry['warnings'].append(f'dataset folder not found: {path}')
            out['datasets'].append(entry)
            continue
        entry['path'] = path
        images = _iter_dataset_images(path)
        all_images.extend(images)
        entry['images'] = len(images)
        out['datasets'].append(entry)

    if excluded_dirs:
        kept = []
        for image in all_images:
            normalized = _normalize(image)
            # canonical keys use forward slashes, so the child-of check does too
            if any(normalized == d or normalized.startswith(d + '/') for d in excluded_dirs):
                continue
            kept.append(image)
        out['excluded_by_control_mask'] = len(all_images) - len(kept)
        all_images = kept
    seen = set()
    unique_images = []
    for image in all_images:
        normalized = _normalize(image)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_images.append(image)
    out['images'] = sorted(unique_images, key=_normalize)
    return out


def selection_path(identity_dir: str) -> str:
    return os.path.join(identity_dir, 'selection.json')


def default_selection() -> dict:
    return {'version': SELECTION_VERSION, 'training_refs': {}, 'external_refs': {}}


def load_selection(path: str) -> dict:
    """Load selection.json tolerantly; a corrupt file resets to defaults."""
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            selection = json.load(handle)
        if not isinstance(selection, dict) or selection.get('version') != SELECTION_VERSION:
            return default_selection()
        selection.setdefault('training_refs', {})
        selection.setdefault('external_refs', {})
        return selection
    except (OSError, ValueError):
        return default_selection()


def save_selection(path: str, selection: dict) -> None:
    import datetime

    selection['version'] = SELECTION_VERSION
    selection['updated_at'] = datetime.datetime.now().isoformat(timespec='seconds')
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as handle:
        json.dump(selection, handle, indent=2)
    os.replace(tmp, path)


def prune_selection(selection: dict, training_paths: List[str], external_names: List[str]) -> dict:
    """Drop entries for images that no longer exist; returns a new dict."""
    training_keys = {_normalize(p) for p in training_paths}
    external_keys = set(external_names)
    pruned = {
        'version': SELECTION_VERSION,
        'training_refs': {k: v for k, v in selection.get('training_refs', {}).items() if _normalize(k) in training_keys},
        'external_refs': {k: v for k, v in selection.get('external_refs', {}).items() if k in external_keys},
    }
    return pruned
