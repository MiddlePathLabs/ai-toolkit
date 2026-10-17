# Copyright 2026 Peter Neill
# SPDX-License-Identifier: Apache-2.0
# Adapted from Fizgig v8.4.0 src/fizgig/families/multi_person.py:
# subject(), fuse(), and dop_caption(). Toolkit adaptations use explicit classes,
# validated mappings, preset overrides, and unchanged missing-trigger DOP captions.
"""Opt-in identity-isolation configuration and boundary-aware caption rewriting.

Fusion joins text; it makes no guarantee about any model's tokenization.
Configuration parsing performs no file reads, downloads, or directory creation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from typing import Any

_PRESETS = {
    "off": (False, False, False, 1, 1.0),
    "dop_only": (False, False, True, 1, 1.0),
    "full": (True, True, True, 2, 2.0),
}


def _boolean(raw: dict, key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"identity_isolation.{key} must be a boolean")
    return value


def _integer(raw: dict, key: str, default: int, minimum: int = 1) -> int:
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"identity_isolation.{key} must be an integer >= {minimum}")
    return value


def _positive(raw: dict, key: str, default: float) -> float:
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"identity_isolation.{key} must be finite and positive")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(
            f"identity_isolation.{key} must be finite and positive"
        ) from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"identity_isolation.{key} must be finite and positive")
    return value


def _text(raw: dict, key: str, default: str = "") -> str:
    value = raw.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"identity_isolation.{key} must be a nonempty string")
    return value.strip()


def _boundary(text: str) -> re.Pattern:
    return re.compile(r"(?<!\w)" + re.escape(text) + r"(?!\w)", re.IGNORECASE)


def _subject(raw: dict) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise ValueError("identity_isolation.subjects entries must be mappings")
    trigger = _text(raw, "trigger")
    noun = re.sub(r"^an?\s+", "", _text(raw, "class_name"), flags=re.I).lower()
    if not re.fullmatch(r"[^\W\d_]+", noun) or noun in ("a", "an"):
        raise ValueError("identity_isolation.class_name must be a plain one-word noun")
    phrase = ("an " if noun[0] in "aeiou" else "a ") + noun
    return {
        "folder_path": _text(raw, "folder_path"),
        "trigger": trigger,
        "class_name": noun,
        "phrase": phrase,
        "fused": re.sub(r"\s+", "", trigger) + noun,
    }


def _validate_subjects(subjects: list[dict[str, str]]) -> None:
    folders: set[str] = set()
    triggers: set[str] = set()
    fused_words: set[str] = set()
    for subject in subjects:
        folder = os.path.normcase(os.path.abspath(subject["folder_path"]))
        if folder in folders:
            raise ValueError(
                "identity_isolation.subjects contains duplicate source folders"
            )
        folders.add(folder)
        trigger, fused = subject["trigger"].casefold(), subject["fused"].casefold()
        if trigger in triggers or any(
            _boundary(trigger).fullmatch(existing) for existing in triggers
        ):
            raise ValueError("identity_isolation.subjects contains duplicate triggers")
        if fused in fused_words or any(
            _boundary(fused).fullmatch(existing) for existing in fused_words
        ):
            raise ValueError("identity_isolation.subjects contains fused collisions")
        triggers.add(trigger)
        fused_words.add(fused)

    # Replacement outputs must never become another mapping's input. Class
    # phrases are outputs too: preservation must not reintroduce an identity.
    for source in subjects:
        trigger_pattern = _boundary(source["trigger"])
        fused_pattern = _boundary(source["fused"])
        for target in subjects:
            if trigger_pattern.search(target["fused"]) or trigger_pattern.search(
                target["phrase"]
            ):
                raise ValueError(
                    "identity_isolation.subjects contains a cross-mapping alias"
                )
            if fused_pattern.search(target["phrase"]):
                raise ValueError(
                    "identity_isolation.subjects contains a class/fused alias"
                )
            if source is not target and fused_pattern.search(target["fused"]):
                raise ValueError(
                    "identity_isolation.subjects contains cascading fused aliases"
                )


class IsolationConfig:
    """Resolve presets once, then apply pure caption transforms.

    ``off`` is inert unless a component is explicitly enabled. This object does
    not change legacy DOP settings. ``raw`` and ``as_dict()`` contain resolved,
    JSON-compatible settings suitable for propagating to dataset configurations.
    """

    def __init__(self, raw: dict | None = None):
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError("identity_isolation must be a mapping or null")
        self.preset: str = _text(raw, "preset", "off").lower()
        if self.preset not in _PRESETS:
            raise ValueError("identity_isolation.preset must be off, dop_only, or full")
        fusion, composites, dop, interval, weight = _PRESETS[self.preset]
        self.fused_captions: bool = _boolean(raw, "fused_captions", fusion)
        self.composites: bool = _boolean(raw, "composites", composites)
        self.dop_enabled: bool = _boolean(raw, "dop_enabled", dop)
        self.composite_count: int = _integer(raw, "composite_count", 16)
        self.composite_lr_multiplier: float = _positive(
            raw, "composite_lr_multiplier", 2.0
        )
        self.dop_interval: int = _integer(raw, "dop_interval", interval)
        self.dop_weight: float = _positive(raw, "dop_weight", weight)
        self.balance_subjects: bool = _boolean(raw, "balance_subjects", False)
        self.stranger_asset_dir: str = _text(
            raw, "stranger_asset_dir", "data/identity_isolation/strangers"
        )
        self.generated_dir: str = _text(
            raw, "generated_dir", "data/identity_isolation/generated"
        )
        self.seed: int = _integer(raw, "seed", 42, minimum=0)
        self.active: bool = (
            self.fused_captions
            or self.composites
            or self.dop_enabled
            or self.balance_subjects
        )
        subjects = raw.get("subjects", [])
        if not isinstance(subjects, list):
            raise ValueError("identity_isolation.subjects must be a list")
        if len(subjects) > 4:
            raise ValueError("identity_isolation supports at most 4 subjects")
        if (
            self.fused_captions or self.composites or self.balance_subjects
        ) and not 1 <= len(subjects) <= 4:
            raise ValueError(
                "identity_isolation requires 1 to 4 subjects for captions, composites, or balancing"
            )
        if not self.active:
            if any(not isinstance(subject, dict) for subject in subjects):
                raise ValueError("identity_isolation.subjects entries must be mappings")
            subjects = []
        self.subjects: list[dict[str, str]] = [
            _subject(subject) for subject in subjects
        ]
        _validate_subjects(self.subjects)
        self._training_rules: list[tuple[re.Pattern, str]] = []
        self._preservation_rules: list[tuple[re.Pattern, str]] = []
        self._build_rules()
        self.raw: dict[str, Any] = {
            "preset": self.preset,
            "subjects": [dict(subject) for subject in self.subjects],
            "fused_captions": self.fused_captions,
            "composites": self.composites,
            "composite_count": self.composite_count,
            "composite_lr_multiplier": self.composite_lr_multiplier,
            "dop_enabled": self.dop_enabled,
            "dop_interval": self.dop_interval,
            "dop_weight": self.dop_weight,
            "balance_subjects": self.balance_subjects,
            "stranger_asset_dir": self.stranger_asset_dir,
            "generated_dir": self.generated_dir,
            "seed": self.seed,
        }
        canonical = json.dumps(
            self.raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        self.signature: str = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _build_rules(self) -> None:
        subjects = sorted(
            self.subjects,
            key=lambda subject: (
                -len(subject["trigger"]),
                subject["trigger"].casefold(),
            ),
        )
        for subject in subjects:
            trigger, phrase, fused = (
                subject["trigger"],
                subject["phrase"],
                subject["fused"],
            )
            for original in (f"{trigger}, {phrase}", f"{trigger} {phrase}", trigger):
                self._training_rules.append((_boundary(original), fused))
            for original in (
                f"{trigger}, {phrase}",
                f"{trigger} {phrase}",
                fused,
                trigger,
            ):
                self._preservation_rules.append((_boundary(original), phrase))

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.raw,
            "subjects": [dict(subject) for subject in self.raw["subjects"]],
        }

    def training_caption(self, text: str) -> str:
        if not self.fused_captions:
            return text
        for pattern, replacement in self._training_rules:
            text = pattern.sub(lambda _match, value=replacement: value, text)
        return text

    def preservation_caption(self, text: str) -> str:
        if not self.active:
            return text
        for pattern, replacement in self._preservation_rules:
            text = pattern.sub(lambda _match, value=replacement: value, text)
        return text

    def caption_preview(self, text: str) -> dict[str, str]:
        return {
            "original": text,
            "training": self.training_caption(text),
            "preservation": self.preservation_caption(text),
        }
