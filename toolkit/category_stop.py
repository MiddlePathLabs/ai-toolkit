"""``train.category_stop``: let photos, clips or voice finish training early.

In a mixed dataset one category can converge (or start to overbake) well before
the others, typically a small voice set next to a larger photo set. Once the
training step reaches ``<kind>_step`` that category retires:

- ``anchor`` (default): its windows keep training at ``ANCHOR_SCALE`` of the
  optimizer's applied update (rehearsal against drift on the shared adapter,
  and its loss stays visible). Uses the optimizer runtime's step scale, so it
  is an update scale, not a loss scale, and holds under Rose / Automagic too.
- ``stop``: its batches are skipped before the forward pass. Faster, blind.

Steps, not epochs (ai-toolkit counts steps), and ``step_num`` survives resume.
One item per optimizer window is required (batch_size 1, no gradient
accumulation) so every window has exactly one category. Regularisation items
(``is_reg``) never count toward a category: they are neither scaled nor skipped.

The window's categories are recorded by the base training loop, so every
trainer sees them. Anchor mode also needs the trainer to open the optimizer
runtime window (SDTrainer does); a trainer that doesn't is refused at bind
instead of silently anchoring at 1.0.

After Fizgig's per-category stop epoch (visual/audio_stop_epoch, ANCHOR_LR_SCALE
0.1).
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

from toolkit.h3_modality_routing import classify_file_item
from toolkit.print import print_acc

ANCHOR_SCALE = 0.1


def category_kinds(batch: Any) -> set[str]:
    """photo / clip / voice of a batch's non-regularisation items."""
    items = getattr(batch, "file_items", None) or ()
    return {
        classify_file_item(item)
        for item in items
        if not bool(getattr(item, "is_reg", False))
    }


def window_category_kinds(batches: Iterable[Any]) -> set[str]:
    kinds: set[str] = set()
    for batch in batches:
        if batch is not None:
            kinds |= category_kinds(batch)
    return kinds


class CategoryStop:
    def __init__(self, steps: dict[str, int], mode: str):
        self.steps = dict(steps)
        self.mode = mode
        self._announced: set[str] = set()
        self.skipped = 0

    def retired(self, kind: str, step: int) -> bool:
        stop_at = self.steps.get(kind)
        return stop_at is not None and step >= stop_at

    def _all_retired(self, kinds: Iterable[str], step: int) -> bool:
        kinds = set(kinds)
        return bool(kinds) and all(self.retired(kind, step) for kind in kinds)

    def _announce(self, kinds: Iterable[str], step: int) -> None:
        for kind in sorted(set(kinds) - self._announced):
            if not self.retired(kind, step):
                continue
            self._announced.add(kind)
            if self.mode == "anchor":
                how = f"keeps training at {ANCHOR_SCALE:g}x update scale (anchor)"
            else:
                how = "its batches are skipped from here on (stop)"
            print_acc(f"[category-stop] {kind} retired at step {step}: {how}")

    def step_scale(self, kinds: Iterable[str], step: int) -> float:
        """Update scale for an optimizer window holding ``kinds``."""
        if self.mode != "anchor" or not self._all_retired(kinds, step):
            return 1.0
        self._announce(kinds, step)
        return ANCHOR_SCALE

    def should_skip(self, batch: Any, step: int) -> bool:
        if self.mode != "stop" or batch is None:
            return False
        kinds = category_kinds(batch)
        if not self._all_retired(kinds, step):
            return False
        self._announce(kinds, step)
        return True


def bind_category_stop(
    train_config: Any,
    optimizer_runtime: Any,
    *,
    supports_anchor: bool = True,
    datasets: Optional[Iterable[Any]] = None,
) -> Optional[CategoryStop]:
    config = getattr(train_config, "category_stop", None)
    if config is None or not config.enabled:
        return None
    if config.mode == "anchor" and not supports_anchor:
        raise ValueError(
            "train.category_stop mode 'anchor' needs a trainer that opens the optimizer "
            "runtime window (the diffusion_trainer / SDTrainer does); this one does not, "
            "so the anchor scale would never apply. Use mode: stop."
        )
    problems = []
    if int(getattr(train_config, "batch_size", 1)) != 1:
        problems.append(f"batch_size={train_config.batch_size}")
    if int(getattr(train_config, "gradient_accumulation", 1) or 1) != 1:
        problems.append(f"gradient_accumulation={train_config.gradient_accumulation}")
    if getattr(train_config, "gradient_accumulation_steps", 1) != 1:
        problems.append(f"gradient_accumulation_steps={train_config.gradient_accumulation_steps}")
    for dataset in datasets or ():
        # with buckets, a dataset-level batch_size overrides train.batch_size, and
        # a bucket can hold more than one kind (stills and clips of the same crop)
        dataset_batch = getattr(dataset, "batch_size", None)
        if dataset_batch is not None and int(dataset_batch) != 1:
            name = getattr(dataset, "folder_path", None) or getattr(dataset, "dataset_path", None)
            problems.append(f"datasets[{name}].batch_size={dataset_batch}")
    if problems:
        raise ValueError(
            "train.category_stop needs one item per optimizer window so each window "
            "has one category; got " + ", ".join(problems)
        )
    if config.mode == "anchor":
        if optimizer_runtime is None or not optimizer_runtime.supports_step_scale:
            label = getattr(optimizer_runtime, "optimizer_label", "this optimizer")
            reason = getattr(optimizer_runtime, "unsupported_reason", None) or "no step scale"
            raise ValueError(
                f"train.category_stop mode 'anchor' scales the optimizer update, but "
                f"{label} cannot: {reason}. Use mode: stop or another optimizer."
            )
    plan = ", ".join(f"{kind} at step {step}" for kind, step in sorted(config.steps.items()))
    print_acc(f"[category-stop] {config.mode}: {plan}")
    return CategoryStop(config.steps, config.mode)
