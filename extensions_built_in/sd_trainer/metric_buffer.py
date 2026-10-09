"""Metric aggregation buffer used by the SDTrainer (ported from the
perceptual fork's metrics overhaul).

Pre-overhaul, each metric in SDTrainer was stored as a single
``self._last_<name>`` scalar that got *overwritten* on every microbatch.
With ``gradient_accumulation_steps > 1`` this meant every metric except
``loss`` reflected only the **final** microbatch of a step.

``MetricBuffer`` accumulates a weighted running mean across microbatches
(``add_scalar`` / ``flush_scalars``). It also carries the fork's per-sample
breakdown machinery (``add_per_sample`` / ``flush_per_sample``,
``MetricValue``) for future use — the current sqlite schema has no
value_text column to persist the JSON payloads into, so the trainer only
wires the scalar means for now.

The buffer is **display-only** — it never participates in any loss tensor
or gradient computation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


class MetricValue(float):
    """Float-with-payload used to ship a per-sample breakdown alongside the
    metric scalar through ``loss_dict`` without breaking any existing
    consumer. Subclasses ``float`` so all arithmetic and formatting just
    work; the ``breakdown`` attribute carries the JSON-serialisable payload.
    """

    breakdown: Dict[str, Any]

    def __new__(cls, value: float, breakdown: Dict[str, Any]):
        inst = super().__new__(cls, value)
        inst.breakdown = breakdown
        return inst

    def __repr__(self):  # pragma: no cover — cosmetic
        return f"MetricValue({float(self)!r}, breakdown_keys={list(self.breakdown.keys())})"


class _ScalarSlot:
    __slots__ = ("sum", "weight")

    def __init__(self) -> None:
        self.sum: float = 0.0
        self.weight: float = 0.0

    def add(self, value: float, weight: float) -> None:
        if weight <= 0.0:
            return
        self.sum += float(value) * float(weight)
        self.weight += float(weight)

    def mean(self) -> Optional[float]:
        if self.weight <= 0.0:
            return None
        return self.sum / self.weight


class _PerSampleSlot:
    """Holds every sample seen for a metric until flush; flush decides which
    K to retain (top-K by deviation from the mean)."""

    __slots__ = ("samples",)

    def __init__(self) -> None:
        self.samples: List[Dict[str, Any]] = []

    def add(self, value: float, t: Optional[float], sample_tag: Optional[str]) -> None:
        entry: Dict[str, Any] = {"value": float(value)}
        if t is not None:
            entry["t"] = float(t)
        if sample_tag is not None:
            entry["sample"] = str(sample_tag)
        self.samples.append(entry)

    def finalize(self, cap: int) -> Dict[str, Any]:
        n = len(self.samples)
        if n == 0:
            return {"samples": [], "n": 0, "mean": None, "std": None}
        values = [s["value"] for s in self.samples]
        mean = sum(values) / n
        var = sum((v - mean) ** 2 for v in values) / n if n > 1 else 0.0
        std = math.sqrt(var)
        if n > cap:
            ordered = sorted(self.samples, key=lambda s: abs(s["value"] - mean), reverse=True)
            kept = ordered[:cap]
        else:
            kept = list(self.samples)
        return {"samples": kept, "n": n, "mean": mean, "std": std}


class MetricBuffer:
    """Accumulates scalar means and per-sample breakdowns across microbatches.

    Call ``reset()`` (or rely on ``flush_*`` clearing state) once per
    optimizer step. Non-finite and ``None`` values are dropped silently.
    """

    def __init__(self, per_sample_cap: int = 16) -> None:
        self._scalars: Dict[str, _ScalarSlot] = {}
        self._per_sample: Dict[str, _PerSampleSlot] = {}
        self._per_sample_cap = int(per_sample_cap)

    def add_scalar(self, name: str, value: Any, weight: float = 1.0) -> None:
        if value is None:
            return
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(v):
            return
        if weight is None or weight <= 0.0:
            return
        slot = self._scalars.get(name)
        if slot is None:
            slot = _ScalarSlot()
            self._scalars[name] = slot
        slot.add(v, float(weight))

    def add_per_sample(self, name: str, value: Any, t: Optional[float] = None,
                       sample_tag: Optional[str] = None) -> None:
        if value is None:
            return
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(v):
            return
        slot = self._per_sample.get(name)
        if slot is None:
            slot = _PerSampleSlot()
            self._per_sample[name] = slot
        slot.add(v, t, sample_tag)

    def flush_scalars(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name, slot in self._scalars.items():
            mean = slot.mean()
            if mean is not None:
                out[name] = mean
        self._scalars.clear()
        return out

    def flush_per_sample(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for name, slot in self._per_sample.items():
            payload = slot.finalize(self._per_sample_cap)
            if payload["n"] > 0:
                out[name] = payload
        self._per_sample.clear()
        return out
