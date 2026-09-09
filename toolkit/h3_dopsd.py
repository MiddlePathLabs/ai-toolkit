"""Extend existing D-OPSD without a second teacher model.

Self-reference (`model_kwargs.dopsd`) is unchanged: one ref2va DiT, two
forwards, the training item as its own reference. Other-photo pairing and
the identity-first schedule are default-off. This module does not wrap
extra LoRA modules — the saved adapter stays the ordinary H3 deployment
LoRA.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

from toolkit.h3_modality_routing import classify_file_item, is_h3_arch
from toolkit.print import print_acc

H3_ARCH_PREFIX = "minimax_h3"
REF_MODES = ("self", "other")
GROUP_BY_MODES = ("folder",)
IDENTITY_FIRST_AUTO_STEPS = 650
IDENTITY_FIRST_LR_SCALE = 1.0 / 3.0


@dataclass(frozen=True)
class DopsdSettings:
    enabled: bool = False
    ref_mode: str = "self"
    ref_count: int = 1
    group_by: str = "folder"
    pair_seed: int = 0
    identity_first: bool = False
    identity_first_steps: int = 0
    identity_first_lr_scale: float = IDENTITY_FIRST_LR_SCALE
    bleed_strength: float = 1.0

    @property
    def other_ref(self) -> bool:
        return self.enabled and self.ref_mode == "other"

    @property
    def self_ref(self) -> bool:
        return self.enabled and self.ref_mode == "self"


def _kwargs_of(model_config: Any) -> dict:
    if model_config is None:
        return {}
    raw = getattr(model_config, "model_kwargs", None)
    return dict(raw or {})


def _finite_float(name: str, raw: Any, default: float) -> float:
    if raw is None:
        value = default
    else:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a finite number, got {raw!r}") from None
    if value != value or abs(value) == float("inf"):
        raise ValueError(f"{name} must be a finite number, got {raw!r}")
    return value


def _int_field(name: str, raw: Any, default: int, *, minimum: Optional[int] = None) -> int:
    if raw is None:
        value = default
    else:
        if isinstance(raw, bool):
            raise ValueError(f"{name} must be an integer, got {raw!r}")
        try:
            value = int(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be an integer, got {raw!r}") from None
        if isinstance(raw, float) and float(value) != float(raw):
            raise ValueError(f"{name} must be an integer, got {raw!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def parse_dopsd_settings(model_config: Any) -> DopsdSettings:
    """Parse `model_kwargs` D-OPSD flags. Default-off."""
    kw = _kwargs_of(model_config)
    enabled = bool(kw.get("dopsd", False))
    mode = str(kw.get("dopsd_ref_mode", "self") or "self").strip().lower()
    if mode not in REF_MODES:
        raise ValueError(
            f"dopsd_ref_mode must be 'self' or 'other', got {kw.get('dopsd_ref_mode')!r}"
        )
    group_by = str(kw.get("dopsd_group_by", "folder") or "folder").strip().lower()
    if group_by not in GROUP_BY_MODES:
        raise ValueError(
            f"dopsd_group_by must be 'folder', got {kw.get('dopsd_group_by')!r}"
        )
    identity_first = bool(kw.get("dopsd_identity_first", False))
    raw_steps = kw.get("dopsd_identity_first_steps", -1)
    if identity_first:
        identity_steps = _int_field(
            "dopsd_identity_first_steps", raw_steps, -1, minimum=-1
        )
    else:
        identity_steps = 0
    if not enabled:
        if mode != "self":
            raise ValueError(
                "dopsd_ref_mode='other' requires model_kwargs.dopsd: true"
            )
        if identity_first:
            raise ValueError(
                "dopsd_identity_first requires model_kwargs.dopsd: true"
            )
        return DopsdSettings()
    lr_scale = _finite_float(
        "dopsd_identity_first_lr_scale",
        kw.get("dopsd_identity_first_lr_scale", IDENTITY_FIRST_LR_SCALE),
        IDENTITY_FIRST_LR_SCALE,
    )
    if lr_scale <= 0.0:
        raise ValueError(
            f"dopsd_identity_first_lr_scale must be > 0, got {lr_scale}"
        )
    return DopsdSettings(
        enabled=True,
        ref_mode=mode,
        ref_count=_int_field("dopsd_ref_count", kw.get("dopsd_ref_count", 1), 1, minimum=1),
        group_by=group_by,
        pair_seed=_int_field(
            "dopsd_pair_seed", kw.get("dopsd_pair_seed", 0), 0, minimum=0
        ),
        identity_first=identity_first,
        identity_first_steps=identity_steps,
        identity_first_lr_scale=lr_scale,
        bleed_strength=_finite_float(
            "dopsd_bleed_strength", kw.get("dopsd_bleed_strength", 1.0), 1.0
        ),
    )


def resolve_identity_first_steps(raw_steps: int, train_steps: int) -> int:
    """Optimizer updates, not epochs. -1 → ~650, clamped to the run length."""
    budget = max(0, int(train_steps))
    if raw_steps is None or int(raw_steps) < 0:
        resolved = min(IDENTITY_FIRST_AUTO_STEPS, budget)
    else:
        resolved = min(int(raw_steps), budget)
    return resolved


def identity_first_teacher_active(
    step_num: int, phase1_steps: int, *, identity_first: bool, dopsd_enabled: bool
) -> bool:
    if not dopsd_enabled:
        return False
    if not identity_first or phase1_steps <= 0:
        return True
    return int(step_num) < int(phase1_steps)


def identity_first_step_scale(
    teacher_active: bool, *, identity_first: bool, lr_scale: float = IDENTITY_FIRST_LR_SCALE
) -> float:
    if identity_first and teacher_active:
        return float(lr_scale)
    return 1.0


def is_dopsd_photo(item: Any) -> bool:
    return classify_file_item(item) == "photo"


def source_path(item: Any) -> str:
    return os.path.normcase(os.path.abspath(str(getattr(item, "path"))))


def pair_key(item: Any) -> str:
    flip_x = int(bool(getattr(item, "flip_x", False)))
    flip_y = int(bool(getattr(item, "flip_y", False)))
    return f"{source_path(item)}|{flip_x}|{flip_y}"



def group_key(item: Any, group_by: str = "folder") -> str:
    if group_by != "folder":
        raise ValueError(f"unsupported dopsd_group_by {group_by!r}")
    return os.path.dirname(source_path(item))



def rotation_partners(keys: Sequence[str], k: int) -> dict[str, list[str]]:
    """Item i references i+1..i+K (mod N). Self-excluding by construction."""
    n = len(keys)
    if n < 2:
        raise ValueError(
            "other-photo D-OPSD needs at least 2 photos in a subject group; "
            f"got {n}"
        )
    kk = min(int(k), n - 1)
    if kk < 1:
        raise ValueError("dopsd_ref_count must be >= 1")
    out: dict[str, list[str]] = {}
    for i, key in enumerate(keys):
        out[key] = [keys[(i + 1 + slot) % n] for slot in range(kk)]
        if key in out[key]:
            raise RuntimeError(f"other-photo pairing included self for {key}")
    return out


def pick_slot(item_path: str, epoch: int, seed: int, k: int) -> int:
    """Deterministic slot in 0..k-1 from item + epoch + seed."""
    if k <= 0:
        raise ValueError(f"k must be >= 1, got {k}")
    payload = f"{item_path}|{int(epoch)}|{int(seed)}".encode("utf-8")
    digest = hashlib.md5(payload).digest()
    return int.from_bytes(digest[:8], "little") % int(k)


def unweighted_errors(teacher_mean: float, photo_mean: Optional[float]) -> dict[str, float]:
    """Raw teacher / photo errors before bleed or distill weights."""
    photo = 0.0 if photo_mean is None else float(photo_mean)
    return {"teacher": float(teacher_mean), "photo": photo}


def format_dopsd_error_log(
    teacher_err: float, photo_err: float, teacher_weight: float
) -> str:
    w = float(teacher_weight)
    photo_w = 1.0 - w
    wt, wp = w * float(teacher_err), photo_w * float(photo_err)
    tot = wt + wp
    share = (100.0 * wp / tot) if tot else 0.0
    return (
        f"[dopsd] teacher err {teacher_err:.4f} x{w:.2f} = {wt:.4f} | "
        f"photo err {photo_err:.4f} x{photo_w:.2f} = {wp:.4f} | "
        f"real pixels are {share:.0f}% of this step's loss "
        f"(the weight alone says {100.0 * photo_w:.0f}%)"
    )


def assign_other_photo_pairs(items: Sequence[Any], settings: DopsdSettings) -> None:
    """Stamp each photo with rotation partners in the same folder group."""
    if not settings.other_ref:
        return
    photos = [item for item in items if is_dopsd_photo(item)]
    groups: dict[str, list[Any]] = {}
    for item in photos:
        groups.setdefault(group_key(item, settings.group_by), []).append(item)
    if not groups:
        raise ValueError(
            "dopsd_ref_mode='other' needs at least 2 photos in one folder; "
            "this dataset has none. Clips and voice sit out of pairing."
        )
    for folder, group in groups.items():
        unique_sources = {source_path(item) for item in group}
        if len(unique_sources) < 2:
            names = [os.path.basename(str(getattr(item, "path", "?"))) for item in group]
            raise ValueError(
                "dopsd_ref_mode='other' never pairs an item with itself or its "
                f"own flip. Folder {folder!r} has {len(unique_sources)} source "
                f"still(s) ({', '.join(names)}). Put at least two stills of the "
                "same subject in that folder."
            )
        by_key = {pair_key(item): item for item in group}
        for item in group:
            eligible = [
                p for p in group if source_path(p) != source_path(item)
            ]
            if not eligible:
                raise ValueError(
                    "dopsd_ref_mode='other' never pairs an item with itself or "
                    f"its own flip: {getattr(item, 'path', item)}"
                )
            eligible_keys = sorted(pair_key(p) for p in eligible)
            n = len(eligible_keys)
            kk = min(int(settings.ref_count), n)
            start = sorted(by_key).index(pair_key(item)) % n
            slots = []
            for slot in range(kk):
                partner_key = eligible_keys[(start + slot) % n]
                partner = by_key[partner_key]
                if source_path(partner) == source_path(item):
                    raise RuntimeError("other-photo pairing included a flip of self")
                if group_key(partner, settings.group_by) != folder:
                    raise RuntimeError("other-photo pairing crossed subjects")
                slots.append(
                    {
                        "path": str(partner.path),
                        "pair_key": partner_key,
                        "flip_x": bool(getattr(partner, "flip_x", False)),
                        "flip_y": bool(getattr(partner, "flip_y", False)),
                    }
                )
            item.dopsd_ref_slots = slots
    for item in items:
        if not hasattr(item, "dopsd_ref_slots"):
            item.dopsd_ref_slots = []


def choose_other_photo_slot(
    item: Any, *, epoch: int, seed: int
) -> Optional[dict[str, Any]]:
    slots = list(getattr(item, "dopsd_ref_slots", None) or [])
    if not slots:
        return None
    idx = pick_slot(pair_key(item), epoch, seed, len(slots))
    chosen = slots[idx]
    if chosen.get("pair_key") == pair_key(item):
        raise RuntimeError("other-photo mode selected the item as its own reference")
    chosen_path = os.path.normcase(os.path.abspath(str(chosen.get("path") or "")))
    if chosen_path == source_path(item):
        raise RuntimeError(
            "other-photo mode selected a flip of the item as its own reference"
        )
    return chosen



def batch_is_photo_only(batch: Any) -> bool:
    items = getattr(batch, "file_items", None) or ()
    if not items:
        return False
    return all(is_dopsd_photo(item) for item in items)


def dopsd_teacher_wanted(
    settings: Optional[DopsdSettings],
    *,
    step_num: int,
    batch: Any,
) -> bool:
    if settings is None or not settings.enabled:
        return False
    if not identity_first_teacher_active(
        step_num,
        settings.identity_first_steps,
        identity_first=settings.identity_first,
        dopsd_enabled=True,
    ):
        return False
    if settings.other_ref and not batch_is_photo_only(batch):
        return False
    return True


def validate_dopsd(
    settings: DopsdSettings,
    *,
    arch: Any,
    optimizer_runtime: Any = None,
    train_config: Any = None,
    model: Any = None,
) -> None:
    """Fail closed before the first batch. Pairing groups validate at dataset init."""
    if not settings.enabled:
        return
    if not is_h3_arch(arch):
        raise ValueError(
            "model_kwargs.dopsd is MiniMax-H3 only "
            f"(model.arch starts with {H3_ARCH_PREFIX}); got arch={arch!r}."
        )
    version = ""
    if model is not None and hasattr(model, "get_base_model_version"):
        version = str(model.get_base_model_version() or "")
    class_name = type(model).__name__ if model is not None else ""
    if "ref2va" not in version and class_name != "MinimaxH3Ref2VAModel":
        raise ValueError(
            "model_kwargs.dopsd requires MiniMax-H3 ref2va (one DiT, two forwards). "
            "Do not load a second teacher model. Set model.arch / partition to ref2va."
        )
    if settings.other_ref and train_config is not None:
        batch_size = int(getattr(train_config, "batch_size", 1) or 1)
        if batch_size != 1:
            raise ValueError(
                "dopsd_ref_mode='other' requires train.batch_size: 1 "
                f"(got {batch_size}); mixed-aspect other-photo refs cannot stack."
            )
    if settings.identity_first:
        if settings.identity_first_steps == 0:
            raise ValueError(
                "dopsd_identity_first is set but resolved to 0 optimizer updates. "
                "Set dopsd_identity_first_steps to a positive count or -1 for auto "
                f"({IDENTITY_FIRST_AUTO_STEPS} steps)."
            )
        if optimizer_runtime is not None and not getattr(
            optimizer_runtime, "supports_step_scale", False
        ):
            label = getattr(optimizer_runtime, "optimizer_label", type(optimizer_runtime).__name__)
            reason = getattr(optimizer_runtime, "unsupported_reason", None) or (
                "supports_step_scale is false"
            )
            raise ValueError(
                f"dopsd_identity_first requires supports_step_scale so phase 1 can "
                f"run at {settings.identity_first_lr_scale:g} of base LR. "
                f"{label} cannot ({reason}). Disable dopsd_identity_first or pick "
                "an optimizer whose step scale is classified as supported."
            )


def bind_dopsd(
    model: Any,
    optimizer_runtime: Any,
    train_config: Any,
) -> Optional[DopsdSettings]:
    """Stamp resolved settings on the model. No-op when D-OPSD is off."""
    model_config = getattr(model, "model_config", None)
    settings = parse_dopsd_settings(model_config)
    if train_config is not None and settings.identity_first:
        settings = DopsdSettings(
            enabled=settings.enabled,
            ref_mode=settings.ref_mode,
            ref_count=settings.ref_count,
            group_by=settings.group_by,
            pair_seed=settings.pair_seed,
            identity_first=settings.identity_first,
            identity_first_steps=resolve_identity_first_steps(
                settings.identity_first_steps,
                int(getattr(train_config, "steps", 0) or 0),
            ),
            identity_first_lr_scale=settings.identity_first_lr_scale,
            bleed_strength=settings.bleed_strength,
        )
    validate_dopsd(
        settings,
        arch=getattr(model, "arch", None) or getattr(model_config, "arch", None),
        optimizer_runtime=optimizer_runtime,
        train_config=train_config,
        model=model,
    )
    apply_settings_to_model(model, settings)
    if not settings.enabled:
        return None
    print_acc(
        f"[dopsd] enabled ref_mode={settings.ref_mode} "
        f"bleed={settings.bleed_strength:g}"
        + (
            f" identity_first={settings.identity_first_steps} steps "
            f"@ {settings.identity_first_lr_scale:g}x LR"
            if settings.identity_first
            else ""
        )
        + (
            f" other-photo k={settings.ref_count} group_by={settings.group_by}"
            if settings.other_ref
            else " self-reference"
        )
    )
    return settings


def apply_settings_to_model(model: Any, settings: DopsdSettings) -> None:
    model.dopsd = settings.enabled
    model.dopsd_enabled = settings.enabled
    model.dopsd_self_ref = settings.self_ref
    model.dopsd_other_ref = settings.other_ref
    model.dopsd_bleed_strength = settings.bleed_strength
    model.dopsd_settings = settings
    if settings.enabled:
        model.require_pixel_tensor_cache = True


def save_other_photo_teacher_cache(path: str, prompt_embeds: Any, ref_tensor) -> None:
    """Teacher embeds plus the other photo's pixels (0-1). Extra key is ignored by PromptEmbeds.load."""
    from safetensors.torch import load_file, save_file

    prompt_embeds.save(path)
    state = dict(load_file(path))
    state["dopsd_ref_tensor"] = ref_tensor.detach().float().cpu().contiguous()
    save_file(state, path)


def _prompt_embeds_from_state(state: dict) -> Any:
    from toolkit.prompt_utils import PromptEmbeds

    text_embeds = []
    pooled_embeds = None
    attention_mask = []
    is_list = False
    for key in sorted(state.keys()):
        if key.startswith("text_embed_"):
            is_list = True
            text_embeds.append(state[key])
        elif key == "text_embed":
            text_embeds.append(state[key])
        elif key == "pooled_embed":
            pooled_embeds = state[key]
        elif key.startswith("attention_mask_"):
            attention_mask.append(state[key])
        elif key == "attention_mask":
            attention_mask.append(state[key])
    pe = PromptEmbeds(None)
    pe.text_embeds = text_embeds
    if len(text_embeds) == 1 and not is_list:
        pe.text_embeds = text_embeds[0]
    if pooled_embeds is not None:
        pe.pooled_embeds = pooled_embeds
    if len(attention_mask) == 1:
        pe.attention_mask = attention_mask[0]
    elif attention_mask:
        pe.attention_mask = attention_mask
    return pe


def load_other_photo_teacher_cache(path: str) -> Tuple[Any, Any]:
    from safetensors.torch import load_file

    if not os.path.exists(path):
        raise ValueError(
            "D-OPSD other-photo teacher cache is missing; enable "
            f"train.cache_text_embeddings and recache. Missing: {path}"
        )
    state = load_file(path, device="cpu")
    if "dopsd_ref_tensor" not in state:
        raise ValueError(
            "D-OPSD other-photo cache has no dopsd_ref_tensor; recache text "
            f"embeddings with dopsd_ref_mode: other. File: {path}"
        )
    return _prompt_embeds_from_state(state), state["dopsd_ref_tensor"]


def load_chosen_other_photo(item: Any, *, epoch: int, seed: int) -> Tuple[Any, Any, dict]:
    slot = choose_other_photo_slot(item, epoch=epoch, seed=seed)
    if slot is None:
        raise ValueError(
            f"D-OPSD other-photo item has no partners: {getattr(item, 'path', item)}"
        )
    path = slot.get("embed_path") or item.get_dopsd_other_text_embedding_path(
        slot["pair_key"]
    )
    embeds, ref_tensor = load_other_photo_teacher_cache(path)
    return embeds, ref_tensor, slot

