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
from dataclasses import dataclass, replace
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
    # teacher-loss shape (musubi-tuner): weight of the magnitude term of the
    # MSE split, and of the video residual's per-channel DC (colour/tone
    # cast). 1.0 / 1.0 is plain MSE.
    loss_mag_weight: float = 1.0
    loss_dc_weight: float = 1.0
    # self-ref teacher recipe (musubi-tuner's `ref` teacher). Base sigma
    # (pre-shift, 1 = pure noise) band where the teacher sees the reference;
    # outside it the step is a base-preservation anchor (teacher runs on the
    # student's own text, no reference). Defaults = always conditioned.
    teacher_sigma_max: float = 1.0
    teacher_sigma_min: float = 0.0
    preservation_weight: float = 1.0
    # wrap self-ref video teacher captions in the official copy declaration
    # (`<Video 1>` fully_preserved, `<Audio 1>` fully_copy)
    copy_declaration: bool = False
    # wrap other-photo teacher captions in the official subject-reference
    # declaration; the reference token becomes <Subject 1>
    subject_declaration: bool = False
    musubi_recipe: bool = False

    @property
    def teacher_gated(self) -> bool:
        return self.teacher_sigma_max < 1.0 or self.teacher_sigma_min > 0.0

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


# musubi-tuner's validated starting recipes, filled in by
# `dopsd_musubi_recipe: true` for any key the config leaves unset.
# self  = their `ref` teacher (the training item is its own reference);
# other = their `subject_ref` teacher (other pictures of the subject).
# Train-config parts of the recipes are only logged (see bind_dopsd).
MUSUBI_RECIPES = {
    "self": {
        "dopsd_teacher_sigma_max": 0.75,
        "dopsd_loss_dc_weight": 0.3,
        "dopsd_copy_declaration": True,
    },
    "other": {
        # identity is decided at base 0.92-1.0: never anchor the top
        "dopsd_teacher_sigma_max": 1.0,
        "dopsd_teacher_sigma_min": 0.15,
        "dopsd_loss_mag_weight": 0.5,
        "dopsd_loss_dc_weight": 0.3,
        "dopsd_subject_declaration": True,
    },
}
MUSUBI_RECIPE_TRAIN_HINTS = {
    "self": "timestep_focus_prob 0.5; watch for a plateau around ~300 steps "
            "and keep the checkpoints at or just after it",
    "other": "lr 3e-4 with 50 warmup steps, ~500 steps, no timestep focus "
             "(1e-3 random-walks away from the teacher, 1e-4 stays weak)",
}


def parse_dopsd_settings(model_config: Any) -> DopsdSettings:
    """Parse `model_kwargs` D-OPSD flags. Default-off."""
    kw = _kwargs_of(model_config)
    enabled = bool(kw.get("dopsd", False))
    mode = str(kw.get("dopsd_ref_mode", "self") or "self").strip().lower()
    if mode not in REF_MODES:
        raise ValueError(
            f"dopsd_ref_mode must be 'self' or 'other', got {kw.get('dopsd_ref_mode')!r}"
        )
    if enabled and bool(kw.get("dopsd_musubi_recipe", False)):
        kw = {**MUSUBI_RECIPES[mode], **kw}
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
        loss_mag_weight=_nonneg_float(
            "dopsd_loss_mag_weight", kw.get("dopsd_loss_mag_weight"), 1.0
        ),
        loss_dc_weight=_nonneg_float(
            "dopsd_loss_dc_weight", kw.get("dopsd_loss_dc_weight"), 1.0
        ),
        teacher_sigma_max=_unit_float(
            "dopsd_teacher_sigma_max", kw.get("dopsd_teacher_sigma_max"), 1.0
        ),
        teacher_sigma_min=_unit_float(
            "dopsd_teacher_sigma_min", kw.get("dopsd_teacher_sigma_min"), 0.0
        ),
        preservation_weight=_nonneg_float(
            "dopsd_preservation_weight", kw.get("dopsd_preservation_weight"), 1.0
        ),
        copy_declaration=bool(kw.get("dopsd_copy_declaration", False)),
        subject_declaration=bool(kw.get("dopsd_subject_declaration", False)),
        musubi_recipe=bool(kw.get("dopsd_musubi_recipe", False)),
    )


def _unit_float(name: str, raw: Any, default: float) -> float:
    value = _finite_float(name, raw, default)
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


# musubi-tuner's ref-teacher caption header (src/musubi_tuner/minimax_h3/
# text_encoder.py, Apache-2.0): the official editing-prompt declaration that
# makes the base treat the reference as a 1:1 copy source. musubi measured
# that video copying saturates without it, but the <Audio 1> fully_copy line
# is what opens audio teaching across the whole band.
_REF_TEACHER_HEADER_VIDEO = (
    "subject_definitions:\n"
    "<Video 1> is the source video for the target video edit.\n"
    "{audio_definition}"
    "\n"
    "summary:\n"
    "[video editing{audio_tag}] The target video is an edited version of <Video 1> "
    "with no changes; all shots, subjects, camera movement, and sound are preserved "
    "as they are.\n"
    "\n"
    "retention_analysis:\n"
    "<Video 1> (all shots): fully_preserved - every shot, subject, action, and camera "
    "movement of the source video is retained without modification.\n"
    "{audio_retention}"
    "\n"
    "detailed_description:\n"
)


def wrap_ref_teacher_caption(caption: str, with_audio: bool) -> str:
    """Self-ref video teacher caption: copy declaration + the caption."""
    header = _REF_TEACHER_HEADER_VIDEO.format(
        audio_definition=(
            "<Audio 1> is the synchronized audio track of <Video 1> and is reused "
            "in the target video.\n" if with_audio else ""
        ),
        audio_tag=" + audio reuse" if with_audio else "",
        audio_retention=(
            "<Audio 1>: fully_copy - <Audio 1> is reused 1:1 as the target video's "
            "complete final audio track.\n" if with_audio else ""
        ),
    )
    return header + caption


# musubi-tuner's subject-reference teacher wrap (same file, Apache-2.0): the
# official full-reference declaration that makes the base read the picture as
# a *subject* reference (identity/appearance) rather than a frame of the
# target. `attribute_transfer` is their validated marker, and the "pose,
# framing, outfit and setting follow the description" clause keeps the
# picture from acting as a copy source. One picture: D-OPSD's other-photo
# teacher shows one partner photo per step.
SUBJECT_REF_TOKEN = "<Subject 1>"


def wrap_subject_reference_caption(caption: str, *, still_image: bool) -> str:
    summary = (
        f"The target is a single still image with no motion, a static shot of "
        f"{SUBJECT_REF_TOKEN} as described below."
        if still_image
        else f"The target video shows {SUBJECT_REF_TOKEN} as described below."
    )
    return (
        "subject_definitions:\n"
        f"{SUBJECT_REF_TOKEN} is the subject whose appearance comes from <Picture 1> "
        "(face and hair style).\n\n"
        f"summary:\n[reference generation] {summary}\n\n"
        "retention_analysis:\n"
        f"{SUBJECT_REF_TOKEN} (appears in [Shot 1]): attribute_transfer - the appearance "
        f"of {SUBJECT_REF_TOKEN} in <Picture 1> is referenced; pose, framing, outfit and "
        "setting follow the description.\n\n"
        f"detailed_description:\n{caption}"
    )


def recipe_warnings(settings: "DopsdSettings") -> list:
    """musubi warns when a teacher band contradicts the mode's validated recipe."""
    warnings = []
    if not settings.enabled:
        return warnings
    if settings.other_ref and settings.teacher_sigma_max < 1.0:
        warnings.append(
            f"dopsd_teacher_sigma_max={settings.teacher_sigma_max:g} with ref_mode "
            "'other': identity is decided at base sigma 0.92-1.0, so the anchor band "
            "pulls exactly those decisions back to the base (musubi: the student "
            "learns composition but never identity). The recipe keeps 1.0."
        )
    if settings.self_ref and settings.teacher_sigma_max > 0.85:
        warnings.append(
            f"dopsd_teacher_sigma_max={settings.teacher_sigma_max:g} with ref_mode "
            "'self': above base sigma ~0.85 the weights cannot align a self-reference "
            "against a noise-dominated input, and unrestricted teaching there "
            "overwrites the base's composition prior (musubi recipe: 0.75)."
        )
    return warnings


def teacher_step_conditioned(settings: "DopsdSettings", base_sigma: float) -> bool:
    """True when the teacher sees the reference this step; False = anchor."""
    return settings.teacher_sigma_min <= base_sigma <= settings.teacher_sigma_max


def _nonneg_float(name: str, raw: Any, default: float) -> float:
    value = _finite_float(name, raw, default)
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def decomposed_teacher_loss(pred, target, mag_weight=1.0, dc_weight=1.0):
    """Per-sample teacher-matching loss, mean-per-element like MSE -> (B,).

    musubi-tuner's exact split of the MSE, ||p - t||^2 =
    (||p|| - ||t||)^2 + 2 ||p|| ||t|| (1 - cos), with the ||p|| factor of the
    direction term detached. Plain MSE couples the two: hedging an
    unpredictable direction pays off by shrinking the norm, so the student
    converges to the conditional mean's reduced magnitude (delayed
    commitment, washed-out contrast at inference). Decoupled, the direction
    gradient is purely rotational and the magnitude optimum is the per-sample
    teacher norm. At mag_weight = dc_weight = 1 the value equals plain MSE.

    mag_weight scales the magnitude term (0 = pure direction). dc_weight
    scales the per-channel DC of the residual (mean over every dim after the
    channel dim: a global colour/tone cast), applied by shrinking the DC of
    both sides so ||p~ - t~||^2 = ||r_ac||^2 + w ||r_dc||^2. Either may be a
    float or a (B,) tensor (per-sample weights, e.g. full weights on anchor
    steps).
    """
    import torch

    p = pred.float()
    t = target.float().detach()
    b = p.shape[0]
    bshape = (b,) + (1,) * (p.ndim - 1)
    dc = torch.as_tensor(dc_weight, dtype=p.dtype, device=p.device)
    if bool((dc != 1.0).any()):
        s = dc.sqrt().expand(b).reshape(bshape) if dc.ndim else dc.sqrt()
        dims = tuple(range(2, p.ndim))
        p_dc = p.mean(dim=dims, keepdim=True)
        t_dc = t.mean(dim=dims, keepdim=True)
        p = p + (s - 1.0) * p_dc
        t = t + (s - 1.0) * t_dc
    pf = p.reshape(b, -1)
    tf = t.reshape(b, -1)
    p_norm = pf.norm(dim=1)
    t_norm = tf.norm(dim=1)
    cos = (pf * tf).sum(dim=1) / (p_norm * t_norm).clamp(min=1e-12)
    magnitude = (p_norm - t_norm).square()
    direction = 2.0 * p_norm.detach() * t_norm * (1.0 - cos)
    mag = torch.as_tensor(mag_weight, dtype=p.dtype, device=p.device)
    return (mag * magnitude + direction) / pf.shape[1]


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



def assign_other_photo_pairs(items: Sequence[Any], settings: DopsdSettings) -> None:
    """Stamp each photo with rotation partners in the same folder group."""
    if not settings.other_ref:
        return
    photos = [item for item in items if is_dopsd_photo(item)]
    groups: dict[str, list[Any]] = {}
    for item in photos:
        groups.setdefault(group_key(item, settings.group_by), []).append(item)
    if not groups:
        # a dataset with NO photos (voice-only / clips-only) sits out of pairing
        # by design — skip it instead of failing mixed stills+voice jobs. The
        # job-wide no-photos-anywhere case still fails closed later, at
        # load_chosen_other_photo ("item has no partners") on the first photo.
        print_acc(
            "[dopsd] other-photo: no photos in this dataset — pairing skipped "
            "(voice/clips sit out)"
        )
        return
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
    completed_update_id: int,
    batch: Any,
) -> bool:
    """Whether the current objective uses the teacher at an update boundary.

    ``completed_update_id`` is intentionally not the processed-input clock:
    accumulation, empty windows, and skipped AMP steps must not advance the
    identity-first phase.
    """
    if settings is None or not settings.enabled:
        return False
    if not identity_first_teacher_active(
        completed_update_id,
        settings.identity_first_steps,
        identity_first=settings.identity_first,
        dopsd_enabled=True,
    ):
        return False
    if settings.other_ref and not batch_is_photo_only(batch):
        return False
    return True


def validate_dopsd_static(
    settings: DopsdSettings,
    *,
    arch: Any,
    train_config: Any = None,
    optimizer_runtime: Any = None,
) -> None:
    """Validate config-only D-OPSD constraints before model/cache setup.

    The loaded model/version and concrete optimizer remain checked by
    :func:`validate_dopsd`; this helper intentionally does not inspect or
    mutate a model.
    """
    if not settings.enabled:
        return
    if not is_h3_arch(arch):
        raise ValueError(
            "model_kwargs.dopsd is MiniMax-H3 only "
            f"(model.arch starts with {H3_ARCH_PREFIX}); got arch={arch!r}."
        )
    if (
        (settings.loss_mag_weight != 1.0 or settings.loss_dc_weight != 1.0)
        and train_config is not None
        and str(getattr(train_config, "loss_type", "mse")) != "mse"
    ):
        raise ValueError(
            "dopsd_loss_mag_weight / dopsd_loss_dc_weight split the MSE; "
            f"they need train.loss_type: mse (got {train_config.loss_type!r})."
        )
    if settings.teacher_sigma_min >= settings.teacher_sigma_max:
        raise ValueError(
            "dopsd_teacher_sigma_min must be below dopsd_teacher_sigma_max, got "
            f"{settings.teacher_sigma_min} / {settings.teacher_sigma_max}"
        )
    if settings.teacher_gated and train_config is not None:
        batch_size = int(getattr(train_config, "batch_size", 1) or 1)
        if batch_size != 1:
            raise ValueError(
                "D-OPSD teacher sigma gates require train.batch_size: 1 "
                f"(got {batch_size})."
            )
    if settings.other_ref and train_config is not None:
        batch_size = int(getattr(train_config, "batch_size", 1) or 1)
        if batch_size != 1:
            raise ValueError(
                "dopsd_ref_mode='other' requires train.batch_size: 1 "
                f"(got {batch_size})."
            )
    if settings.identity_first and settings.identity_first_steps == 0:
        raise ValueError(
            "dopsd_identity_first is set but resolved to 0 optimizer updates."
        )
    if (
        settings.identity_first
        and optimizer_runtime is not None
        and not getattr(optimizer_runtime, "supports_step_scale", False)
    ):
        label = getattr(optimizer_runtime, "optimizer_label", type(optimizer_runtime).__name__)
        raise ValueError(
            f"dopsd_identity_first requires optimizer step scaling; {label} "
            "does not expose supports_step_scale."
        )


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
    validate_dopsd_static(
        settings,
        arch=arch,
        train_config=train_config,
        optimizer_runtime=optimizer_runtime,
    )
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
    if (
        (settings.loss_mag_weight != 1.0 or settings.loss_dc_weight != 1.0)
        and train_config is not None
        and str(getattr(train_config, "loss_type", "mse")) != "mse"
    ):
        raise ValueError(
            "dopsd_loss_mag_weight / dopsd_loss_dc_weight split the MSE; they need "
            f"train.loss_type: mse (got {train_config.loss_type!r})."
        )
    if settings.teacher_sigma_min >= settings.teacher_sigma_max:
        raise ValueError(
            "dopsd_teacher_sigma_min must be below dopsd_teacher_sigma_max, got "
            f"{settings.teacher_sigma_min} / {settings.teacher_sigma_max}"
        )
    if settings.teacher_gated and train_config is not None:
        batch_size = int(getattr(train_config, "batch_size", 1) or 1)
        if batch_size != 1:
            raise ValueError(
                "dopsd_teacher_sigma_min/max gate the teacher per step and need "
                f"train.batch_size: 1 (got {batch_size})."
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
        settings = replace(
            settings,
            identity_first_steps=resolve_identity_first_steps(
                settings.identity_first_steps,
                int(getattr(train_config, "steps", 0) or 0),
            ),
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
        f"bleed={settings.bleed_strength:g} "
        f"loss mag={settings.loss_mag_weight:g} dc={settings.loss_dc_weight:g}"
        + (
            f" teacher base-sigma band [{settings.teacher_sigma_min:g}, "
            f"{settings.teacher_sigma_max:g}] anchor weight {settings.preservation_weight:g}"
            if settings.teacher_gated
            else ""
        )
        + (" copy_declaration" if settings.self_ref and settings.copy_declaration else "")
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
        + (" subject_declaration" if settings.other_ref and settings.subject_declaration else "")
    )
    for warning in recipe_warnings(settings):
        print_acc(f"[dopsd] WARNING: {warning}")
    if settings.musubi_recipe:
        print_acc(
            f"[dopsd] musubi recipe ({settings.ref_mode}); train-side settings it "
            f"was validated with: {MUSUBI_RECIPE_TRAIN_HINTS[settings.ref_mode]}"
        )
    return settings


def apply_settings_to_model(model: Any, settings: DopsdSettings) -> None:
    model.dopsd = settings.enabled
    model.dopsd_enabled = settings.enabled
    model.dopsd_self_ref = settings.self_ref
    model.dopsd_other_ref = settings.other_ref
    model.dopsd_bleed_strength = settings.bleed_strength
    model.dopsd_copy_declaration = settings.self_ref and settings.copy_declaration
    model.dopsd_subject_declaration = settings.other_ref and settings.subject_declaration
    model.dopsd_settings = settings
    if settings.enabled:
        model.require_pixel_tensor_cache = True


def save_other_photo_teacher_cache(path: str, prompt_embeds: Any, ref_tensor) -> None:
    """Teacher embeds plus the other photo's pixels (0-1). Extra key is ignored by PromptEmbeds.load."""
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    prompt_embeds.save(path)
    # preserve class metadata so PromptEmbeds.load keeps dispatching to the
    # right subclass (the raw re-save below would otherwise strip it)
    with safe_open(path, framework="pt") as f:
        meta = f.metadata()
    state = dict(load_file(path))
    state["dopsd_ref_tensor"] = ref_tensor.detach().float().cpu().contiguous()
    save_file(state, path, metadata=meta)


def _prompt_embeds_from_state(state: dict) -> Any:
    if "text_embeds" in state:
        # H3 caches are AdvancedPromptEmbeds: tensors live under 'text_embeds'
        # (plural) with parallel 'text_token_tags' — the generic PromptEmbeds
        # roundtrip keys (text_embed / text_embed_{i}) never appear here, and
        # token tags must survive (the packed layout consumes them).
        from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds

        pe = AdvancedPromptEmbeds(
            text_embeds=state["text_embeds"],
            text_token_tags=state["text_token_tags"],
        )
        pe.frozen_dtype_keys = ["text_token_tags"]
        return pe

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

