"""Translate perceptual-fork (ai-toolkit-perceptual) config layout to ours.

The perceptual fork keeps every perceptual anchor inside ``face_id:``
(``body_proportion_loss_weight``, ``body_shape_loss_min_cos``,
``vae_anchor_model_path``, ...). This fork splits them into their own process
sections (``body_proportion:``, ``body_shape:``, ``normal_id:``,
``vae_anchor:``) with un-prefixed keys. Config classes read keys with
``kwargs.get``, so a fork-style YAML would otherwise load with those losses
silently disabled.

Pure Python with no torch import: used by both ``toolkit.admission`` (UI/API)
and ``BaseSDTrainProcess`` (runtime) so the two always agree.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

# face_id legacy key -> (our section, our key)
LEGACY_FACE_ID_KEY_MAP: dict[str, tuple[str, str]] = {
    'body_proportion_loss_weight': ('body_proportion', 'loss_weight'),
    'body_proportion_loss_min_t': ('body_proportion', 'loss_min_t'),
    'body_proportion_loss_max_t': ('body_proportion', 'loss_max_t'),
    'body_proportion_include_head': ('body_proportion', 'include_head'),
    'body_shape_loss_weight': ('body_shape', 'loss_weight'),
    'body_shape_loss_min_t': ('body_shape', 'loss_min_t'),
    'body_shape_loss_max_t': ('body_shape', 'loss_max_t'),
    'body_shape_loss_min_cos': ('body_shape', 'loss_min_cos'),
    'normal_loss_weight': ('normal_id', 'loss_weight'),
    'normal_loss_min_t': ('normal_id', 'loss_min_t'),
    'normal_loss_max_t': ('normal_id', 'loss_max_t'),
    'vae_anchor_loss_weight': ('vae_anchor', 'loss_weight'),
    'vae_anchor_loss_min_t': ('vae_anchor', 'loss_min_t'),
    'vae_anchor_loss_max_t': ('vae_anchor', 'loss_max_t'),
    'vae_anchor_model_path': ('vae_anchor', 'vae_model_path'),
}

# Keys FaceIDConfig actually reads (pinned against the class in tests).
SUPPORTED_FACE_ID_KEYS = frozenset({
    'identity_loss_weight',
    'identity_loss_min_t',
    'identity_loss_max_t',
    'identity_loss_min_cos',
    'face_model',
    'identity_loss_decoded_det_threshold',
    'identity_loss_use_average',
    'identity_loss_average_blend',
    'identity_loss_use_random',
    'identity_loss_num_refs',
})

# Dataset-level keys the perceptual fork reads that this fork does not.
UNSUPPORTED_DATASET_KEYS = frozenset({
    'depth_as_control',
    'depth_model_id',
    'diffusion_loss_weight',
    'diffusion_loss_min_t',
    'diffusion_loss_max_t',
    'face_suppression_weight',
    'face_suppression_expand',
    'face_suppression_soft',
    'landmark_loss_weight',
    'latent_perceptual_loss_weight',
    'latent_perceptual_loss_min_t',
    'latent_perceptual_loss_max_t',
})

# Process sections the perceptual fork reads that this fork does not.
UNSUPPORTED_PROCESS_SECTIONS = frozenset({'body_id'})


DEPTH_PREVIEW_ONLY_MESSAGE = (
    "depth_consistency.preview_only is true with a positive depth weight: depth "
    "will NOT train (preview_only disables the depth loss for every sample here; "
    "the perceptual fork only added previews of zero-weight samples). Set "
    "preview_only: false to train depth."
)

IDENTITY_MODE_MESSAGE = (
    "identity loss is active but face_id.identity_loss_use_average is not set: "
    "using per-image reference embeddings (this fork's default). The perceptual "
    "fork defaulted to the dataset-average reference. Set "
    "identity_loss_use_average explicitly to silence this."
)


def _positive(value: Any) -> bool:
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return bool(value)


@dataclass
class LegacyConfigReport:
    process: dict
    # (legacy key, "section.key", value)
    moved: list[tuple[str, str, Any]] = field(default_factory=list)
    # (legacy key, "section.key", legacy value, existing section value)
    conflicts: list[tuple[str, str, Any, Any]] = field(default_factory=list)
    # face_id keys this fork does not read
    ignored_face_id: list[str] = field(default_factory=list)
    # (dataset index, key)
    ignored_dataset: list[tuple[int, str]] = field(default_factory=list)
    ignored_sections: list[str] = field(default_factory=list)
    # depth_consistency present but relying on defaults that differ from the fork
    depth_default_keys: list[str] = field(default_factory=list)
    # depth preview_only with a positive depth weight: nothing trains here,
    # while the perceptual fork still trained the weighted samples
    depth_preview_only_blocks_weight: bool = False
    # identity loss active but the reference mode is implicit (here: per-image;
    # perceptual fork default: dataset average)
    identity_mode_implicit: bool = False

    @property
    def has_notes(self) -> bool:
        return bool(
            self.moved or self.conflicts or self.ignored_face_id
            or self.ignored_dataset or self.ignored_sections or self.depth_default_keys
            or self.depth_preview_only_blocks_weight or self.identity_mode_implicit
        )

    def messages(self) -> list[str]:
        out = []
        if self.moved:
            out.append(
                "perceptual-fork face_id keys translated: "
                + ", ".join(f"face_id.{k} -> {dst}" for k, dst, _ in self.moved)
            )
        for k, dst, legacy, existing in self.conflicts:
            out.append(
                f"face_id.{k}={legacy!r} conflicts with {dst}={existing!r}; set only one"
            )
        if self.ignored_face_id:
            out.append(
                "face_id keys not supported by this fork and IGNORED: "
                + ", ".join(sorted(self.ignored_face_id))
            )
        if self.ignored_dataset:
            out.append(
                "dataset keys not supported by this fork and IGNORED: "
                + ", ".join(f"datasets[{i}].{k}" for i, k in self.ignored_dataset)
            )
        if self.ignored_sections:
            out.append(
                "process sections not supported by this fork and IGNORED: "
                + ", ".join(sorted(self.ignored_sections))
            )
        if self.depth_default_keys:
            out.append(
                "depth_consistency omits "
                + ", ".join(self.depth_default_keys)
                + "; this fork defaults to loss_weight=0.0 / mask_source='none' "
                "(the perceptual fork defaulted to 0.1 / 'subject'). Set them "
                "explicitly to get the fork's behavior."
            )
        if self.depth_preview_only_blocks_weight:
            out.append(DEPTH_PREVIEW_ONLY_MESSAGE)
        if self.identity_mode_implicit:
            out.append(IDENTITY_MODE_MESSAGE)
        return out


def normalize_perceptual_config(process: Mapping[str, Any]) -> LegacyConfigReport:
    """Return a translated deep copy of ``process`` plus a report.

    Legacy ``face_id`` keys are moved into their own sections. A legacy key
    whose target is already set to a different value is a conflict and is left
    untouched (callers must reject the config). Never mutates the input.
    """
    normalized = copy.deepcopy(process)
    report = LegacyConfigReport(process=normalized)
    if not isinstance(normalized, dict):
        return report

    face_id = normalized.get('face_id')
    if isinstance(face_id, Mapping):
        for key in list(face_id.keys()):
            if key in LEGACY_FACE_ID_KEY_MAP:
                section_name, target_key = LEGACY_FACE_ID_KEY_MAP[key]
                section = normalized.get(section_name)
                if section is None:
                    section = {}
                    normalized[section_name] = section
                elif not isinstance(section, dict):
                    # Malformed target section; leave it to config parsing.
                    continue
                value = face_id[key]
                dst = f"{section_name}.{target_key}"
                if target_key in section and section[target_key] != value:
                    report.conflicts.append((key, dst, value, section[target_key]))
                    continue
                section[target_key] = value
                del face_id[key]
                report.moved.append((key, dst, value))
            elif key not in SUPPORTED_FACE_ID_KEYS:
                report.ignored_face_id.append(key)

    for name in UNSUPPORTED_PROCESS_SECTIONS:
        if normalized.get(name) is not None:
            report.ignored_sections.append(name)

    datasets = normalized.get('datasets')
    if isinstance(datasets, list):
        for index, dataset in enumerate(datasets):
            if isinstance(dataset, Mapping):
                for key in dataset.keys():
                    if key in UNSUPPORTED_DATASET_KEYS:
                        report.ignored_dataset.append((index, key))

    dataset_maps = [d for d in datasets if isinstance(d, Mapping)] if isinstance(datasets, list) else []

    depth = normalized.get('depth_consistency')
    if depth is not None:
        depth_map = depth if isinstance(depth, Mapping) else {}
        report.depth_default_keys = [
            k for k in ('loss_weight', 'mask_source') if k not in depth_map
        ]
        if _truthy(depth_map.get('preview_only', False)):
            report.depth_preview_only_blocks_weight = _positive(depth_map.get('loss_weight')) or any(
                _positive(d.get('depth_loss_weight')) for d in dataset_maps
            )

    face_map = normalized.get('face_id') if isinstance(normalized.get('face_id'), Mapping) else {}
    identity_active = _positive(face_map.get('identity_loss_weight')) or any(
        _positive(d.get('identity_loss_weight')) for d in dataset_maps
    )
    report.identity_mode_implicit = identity_active and 'identity_loss_use_average' not in face_map

    return report
