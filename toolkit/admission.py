"""Side-effect-free Krea/H3 admission diagnostics.

This module deliberately depends only on the Python standard library.  The
public ``collect_admission_diagnostics`` function accepts the same serialized
job shape used by the UI and direct CLI.  It resolves defaults and process /
dataset inheritance without constructing a model, optimizer, accelerator,
data loader, or cache.

The returned contract is stable and JSON serializable::

    {"valid": bool, "diagnostics": [{
        "rule_id": str, "severity": "error" | "warning",
        "fields": [str, ...], "reason": str, "remedy": str,
        "phase": "static" | "runtime"
    }], "deferred": [{...}], "resolved": [{...}]}

Runtime-only checks are returned in ``deferred`` and remain authoritative in
model/optimizer/dataset final guards.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional


# Stable IDs consumed by the future UI/API adapter.
RULE_LEGACY_ACCUMULATION = "admission.legacy_accumulation"
RULE_FUSED_BACKWARD = "admission.fused_backward_multi_backward"
RULE_MEAN_FLOW = "admission.mean_flow"
RULE_TARGET_COLLISION = "admission.target_collision"
RULE_FLOW_UNAUGMENTED = "admission.flow_unaugmented_target"
RULE_H3_IMAGE_AUXILIARY = "admission.h3_image_auxiliary"
RULE_H3_AUDIO = "admission.h3_standalone_audio"
RULE_H3_AUDIO_DURATION = "admission.h3_audio_duration_runtime"
RULE_H3_DOPSD_AUDIO = "admission.h3_dopsd_standalone_audio"
RULE_H3_FPS = "admission.h3_target_fps"
RULE_H3_FRAME_GEOMETRY = "admission.h3_frame_geometry"
RULE_H3_FAST_CONDITIONING = "admission.h3_fast_conditioning"
RULE_TREAD_VARIANT = "admission.tread_variant"
RULE_TREAD_SPAN = "admission.tread_span_runtime"
RULE_DOPSD_VARIANT = "admission.dopsd_variant"
RULE_DOPSD_LOSS = "admission.dopsd_loss"
RULE_DOPSD_BATCH = "admission.dopsd_batch"
RULE_DOPSD_OPTIMIZER = "admission.dopsd_optimizer"
RULE_KREA_CONFLICT = "admission.krea_conflict"
RULE_KREA_LOW_VRAM_AUX = "admission.krea_low_vram_auxiliary"
RULE_TEXT_CACHE_MIX = "admission.mixed_text_cache"
RULE_LATENT_CACHE_AUGMENT = "admission.latent_cache_augmentation"
RULE_VAE_ANCHOR = "admission.vae_anchor_backend"
RULE_PREVIEW_LORA = "admission.preview_lora"
RULE_TEMPORARY_SAFETY = "admission.temporary_safety_block"
RULE_TURBO_WARNING = "admission.krea_turbo_warning"
RULE_UNCERTIFIED_RUNTIME = "admission.runtime_deferred"
RULE_RESUME_DETERMINISM = "admission.resume_determinism"


_AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".aiff", ".alac"
}
_H3_FRAME_CHUNK = 17
_H3_FRAME_REMAINDER = 5
_H3_FPS = 24
_AUDIO_SCAN_MAX_ENTRIES = 256
_AUDIO_SCAN_MAX_DEPTH = 2


class AdmissionError(ValueError):
    """Raised by the training preflight when static admission has errors."""

    def __init__(self, result: Mapping[str, Any]):
        self.result = dict(result)
        errors = [d for d in result.get("diagnostics", []) if d.get("severity") == "error"]
        message = "\n".join(
            f"[{d['rule_id']}] {d['reason']} Remedy: {d['remedy']}"
            for d in errors
        )
        super().__init__(message or "configuration failed admission")


class _Collector:
    def __init__(self, *, source: str):
        self.source = source
        self.diagnostics: list[dict[str, Any]] = []
        self.deferred: list[dict[str, Any]] = []

    def add(
        self,
        rule_id: str,
        fields: Sequence[str],
        reason: str,
        remedy: str,
        *,
        severity: str = "error",
        phase: str = "static",
    ) -> None:
        self.diagnostics.append({
            "rule_id": rule_id,
            "severity": severity,
            "fields": list(fields),
            "reason": reason,
            "remedy": remedy,
            "phase": phase,
        })

    def defer(self, rule_id: str, fields: Sequence[str], reason: str, remedy: str) -> None:
        self.deferred.append({
            "rule_id": rule_id,
            "severity": "warning",
            "fields": list(fields),
            "reason": reason,
            "remedy": remedy,
            "phase": "runtime",
        })


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"false", "no", "off", "0", "none", "null", ""}:
            return False
        if normalized in {"true", "yes", "on", "1"}:
            return True
    return bool(value)

def _number(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _integer(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    if isinstance(value, float) and result != value:
        return default
    return result


def _positive(value: Any) -> bool:
    return _number(value, 0.0) > 0.0


def _nonempty(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def _nonempty_text(value: Any) -> bool:
    """Whether a serialized path/text field contains a non-empty string."""
    return isinstance(value, str) and value.strip() != ""


def _preview_guidance_scales(
    sample: Mapping[str, Any], process_path: str
) -> list[tuple[float, str]]:
    """Resolve the guidance scale each preview item receives at runtime.

    ``SampleConfig`` defaults the root scale to 7 and turns legacy ``prompts``
    into ``SampleItem`` objects.  Admission must inspect those effective items,
    rather than only a possibly-absent root field.
    """
    root_field = f"{process_path}.sample.guidance_scale"
    root_scale = _number(sample.get("guidance_scale", 7.0), 7.0)
    if "samples" in sample:
        raw_items = _list(sample.get("samples"))
        item_entries = [
            (_mapping(item), f"{process_path}.sample.samples[{index}].guidance_scale")
            for index, item in enumerate(raw_items)
        ]
    else:
        prompts = _list(sample.get("prompts"))
        item_entries = [({}, root_field) for _ in prompts]

    resolved: list[tuple[float, str]] = []
    for item, item_field in item_entries:
        if "guidance_scale" in item:
            resolved.append((_number(item.get("guidance_scale"), root_scale), item_field))
        else:
            resolved.append((root_scale, root_field))
    return resolved

def _arch_info(model: Mapping[str, Any]) -> tuple[str, str, str, Mapping[str, Any]]:
    raw_arch = str(model.get("arch") or "")
    base, _, suffix = raw_arch.partition(":")
    model_kwargs = _mapping(model.get("model_kwargs"))
    partition = str(model_kwargs.get("partition") or model.get("partition") or "").lower()
    if not base:
        if _bool(model.get("is_flux")):
            base = "flux"
        elif _bool(model.get("is_xl")):
            base = "sdxl"
    variant = suffix or base
    if base == "minimax_h3" and partition.startswith("ref2va"):
        variant = "minimax_h3_ref2va"
    elif base == "minimax_h3" and suffix.lower() in {"fast", "vsa", "fasth3"}:
        variant = "minimax_h3_vsa"
    return raw_arch or base, base.lower(), variant.lower(), model_kwargs


def _is_h3(base: str) -> bool:
    return base.startswith("minimax_h3")


def _is_krea(base: str) -> bool:
    return base.startswith("krea2") or base.startswith("krea_2")


def _is_flow(base: str, train: Mapping[str, Any]) -> bool:
    scheduler = str(train.get("noise_scheduler") or "").lower()
    return _is_h3(base) or _is_krea(base) or scheduler in {"flowmatch", "flow_matching", "flow"}


def _known_fused_backward(train: Mapping[str, Any]) -> bool:
    optimizer = str(train.get("optimizer") or "").lower().replace("-", "")
    params = _mapping(train.get("optimizer_params"))
    if optimizer in {"automagic2", "automagicv2"}:
        return True
    if optimizer in {"automagic3", "automagicexperiment", "automagicexperimental"}:
        return bool(params.get("fused", True))
    if optimizer in {"adamconvrot", "adam_convrot"}:
        return bool(params.get("fused", False))
    return False


def _known_step_scale_support(train: Mapping[str, Any]) -> Optional[bool]:
    optimizer = str(train.get("optimizer") or "").lower().replace("-", "")
    params = _mapping(train.get("optimizer_params"))
    if optimizer in {
        "adam", "adamw", "adagrad", "adam8", "adam8bit", "adamw8", "adamw8bit",
        "rose", "automagic", "automagic2", "automagic3", "automagicexperiment",
        "adamconvrot", "adam_convrot", "prodigy8bit",
    }:
        return True
    if optimizer == "adafactor":
        return not bool(params.get("relative_step", False)) and not bool(params.get("scale_parameter", False))
    if optimizer.startswith("dadaptation") or optimizer in {"prodigy", "lion"}:
        return False
    return None


def _has_multi_backward(train: Mapping[str, Any]) -> bool:
    modern = _integer(train.get("gradient_accumulation", 1), 1)
    legacy = _integer(train.get("gradient_accumulation_steps", 1), 1)
    return modern > 1 or legacy != 1 or _bool(train.get("single_item_batching")) or str(train.get("loss_type", "mse")) == "mean_flow"


def _dataset_has_audio(dataset: Mapping[str, Any]) -> Optional[bool]:
    """Find one audio file with a bounded, non-materializing directory scan.

    Admission only needs to know whether a dataset can contain standalone
    audio. Decoding, duration checks, and complete dataset enumeration remain
    deferred to the runtime loader.
    """
    raw = dataset.get("dataset_path", dataset.get("folder_path"))
    if not _nonempty(raw):
        return None
    path = Path(str(raw)).expanduser()
    try:
        if path.is_file():
            return path.suffix.lower() in _AUDIO_EXTENSIONS
        if not path.is_dir():
            return None
        pending: list[tuple[Path, int]] = [(path, 0)]
        scanned = 0
        while pending:
            current, depth = pending.pop()
            with os.scandir(current) as entries:
                for entry in entries:
                    scanned += 1
                    if scanned > _AUDIO_SCAN_MAX_ENTRIES:
                        return None
                    if entry.is_file(follow_symlinks=False):
                        if Path(entry.name).suffix.lower() in _AUDIO_EXTENSIONS:
                            return True
                    elif depth < _AUDIO_SCAN_MAX_DEPTH and entry.is_dir(follow_symlinks=False):
                        pending.append((Path(entry.path), depth + 1))
        return False
    except OSError:
        return None




def _auxiliary_fields(
    process: Mapping[str, Any], datasets: list[Mapping[str, Any]], process_path: str
) -> tuple[list[str], list[tuple[int, list[str]]]]:
    process_fields: list[str] = []
    dataset_fields: list[tuple[int, list[str]]] = []
    entries = (
        ("depth_consistency", "depth_loss_weight"),
        ("normal_id", "normal_loss_weight"),
        ("body_proportion", "body_proportion_loss_weight"),
        ("face_id", "identity_loss_weight"),
        ("body_shape", "body_shape_loss_weight"),
        ("vae_anchor", "vae_anchor_loss_weight"),
    )
    for name, override in entries:
        global_raw = _mapping(process.get(name))
        if name == "face_id":
            global_weight = global_raw.get("identity_loss_weight", 0.0)
        else:
            global_weight = global_raw.get("loss_weight", 0.0)
        if _positive(global_weight) or (
            name in {"depth_consistency", "normal_id"} and _bool(global_raw.get("preview_only"))
        ):
            process_fields.append(f"{process_path}.{name}")
        for index, dataset in enumerate(datasets):
            dataset = _mapping(dataset)
            override_value = dataset.get(override, None)
            effective = global_weight if override_value is None else override_value
            preview = name in {"depth_consistency", "normal_id"} and _bool(global_raw.get("preview_only"))
            if _positive(effective) or preview:
                fields = [
                    f"{process_path}.datasets[{index}].{override}"
                    if override_value is not None
                    else f"{process_path}.{name}"
                ]
                dataset_fields.append((index, fields))
    return process_fields, dataset_fields


def _validate_process(process: Mapping[str, Any], process_index: int, collector: _Collector) -> dict[str, Any]:
    p = f"config.process[{process_index}]"
    train = _mapping(process.get("train"))
    model = _mapping(process.get("model"))
    network_raw = process.get("network")
    network = _mapping(network_raw)
    adapter_raw = process.get("adapter")
    embedding_raw = process.get("embedding")
    decorator_raw = process.get("decorator")
    sample = _mapping(process.get("sample"))
    datasets = [_mapping(item) for item in _list(process.get("datasets"))]
    raw_arch, base, variant, model_kwargs = _arch_info(model)
    h3 = _is_h3(base)
    krea = _is_krea(base)
    flow = _is_flow(base, train)
    resume_mode = train.get("resume_mode", "auto")
    if resume_mode not in {"auto", "exact", "weights_only"}:
        collector.add(
            RULE_RESUME_DETERMINISM,
            [f"{p}.train.resume_mode"],
            f"Unknown resume mode {resume_mode!r}.",
            "Use train.resume_mode: auto, exact, or weights_only.",
        )
    elif resume_mode == "exact":
        if str(process.get("device", "cuda")).split(":")[0] != "cpu":
            collector.add(
                RULE_RESUME_DETERMINISM,
                [f"{p}.device", f"{p}.train.resume_mode"],
                "Exact replay is currently admitted only for the exercised CPU state-machine contract.",
                "Use device: cpu for exact replay or choose weights_only for an explicit GPU warm start.",
            )
        worker_fields = []
        has_buckets = False
        for index, dataset in enumerate(datasets):
            workers = _integer(dataset.get("num_workers", 2), 2)
            if workers != 0:
                worker_fields.append(f"{p}.datasets[{index}].num_workers")
            has_buckets = has_buckets or _bool(dataset.get("buckets"))
        if worker_fields:
            collector.add(
                RULE_RESUME_DETERMINISM,
                worker_fields,
                "Exact resume replay is only proven with a main-process, zero-worker loader; worker prefetch and worker RNG state are not captured.",
                "Set num_workers: 0 on every dataset or use resume_mode: weights_only.",
            )
        if has_buckets:
            collector.add(
                RULE_RESUME_DETERMINISM,
                [f"{p}.datasets[{index}].buckets" for index, dataset in enumerate(datasets) if _bool(dataset.get("buckets"))],
                "Exact replay does not yet prove bucket reshuffle and bucket-local augmentation state.",
                "Disable buckets for exact resume or use resume_mode: weights_only.",
            )
        if _integer(train.get("gradient_accumulation", 1), 1) != 1:
            collector.add(
                RULE_RESUME_DETERMINISM,
                [f"{p}.train.gradient_accumulation"],
                "Exact completed-update snapshots do not capture an in-flight accumulation window.",
                "Use gradient_accumulation: 1 for exact resume or use weights_only.",
            )
        if _integer(train.get("gradient_accumulation_steps", 1), 1) != 1:
            collector.add(
                RULE_RESUME_DETERMINISM,
                [f"{p}.train.gradient_accumulation_steps"],
                "Exact completed-update snapshots do not capture legacy accumulation windows.",
                "Use gradient_accumulation_steps: 1 for exact resume or use weights_only.",
            )
        if _bool(model.get("compile")) or _bool(model.get("block_compile")):
            collector.add(
                RULE_RESUME_DETERMINISM,
                [f"{p}.model.compile", f"{p}.model.block_compile"],
                "Exact resume has only been proved with compilation disabled.",
                "Set compile and block_compile false or use resume_mode: weights_only.",
            )
        if any(_bool(train.get(name)) for name in ("distributed", "multi_gpu")) or _integer(train.get("num_processes", 1), 1) != 1:
            collector.add(
                RULE_RESUME_DETERMINISM,
                [f"{p}.train.distributed", f"{p}.train.multi_gpu", f"{p}.train.num_processes"],
                "Exact resume has only been proved for one process; rank-local sampler/RNG state is not captured.",
                "Use one process for exact resume or use resume_mode: weights_only.",
            )

    # Match BaseSDTrainProcess.is_fine_tuning: a configured network, embedding,
    # decorator, or trained adapter makes this an adapter-role process even when
    # train_unet remains true for the network's forward path.
    trainability_roles: list[str] = []
    if network_raw is not None:
        trainability_roles.append("network")
    if _mapping(adapter_raw) and _bool(_mapping(adapter_raw).get("train")):
        trainability_roles.append("adapter")
    if embedding_raw is not None:
        trainability_roles.append("embedding")
    if decorator_raw is not None:
        trainability_roles.append("decorator")
    is_fine_tuning = not trainability_roles

    network_kwargs = _mapping(network.get("network_kwargs"))
    module_dropout = network.get("module_dropout", network_kwargs.get("module_dropout"))
    if (h3 or krea) and _positive(module_dropout):
        collector.add(
            RULE_TEMPORARY_SAFETY,
            [f"{p}.network.module_dropout", f"{p}.network.network_kwargs.module_dropout"],
            "Module dropout remains temporarily blocked on the Krea/H3 admission surface until its production checkpoint/RNG acceptance path is complete.",
            "Set module_dropout to 0 or remove it explicitly; admission never silently disables dropout.",
        )
    preview_scales = _preview_guidance_scales(sample, p)
    sample_has_preview = bool(preview_scales)
    if krea and is_fine_tuning and _bool(train.get("train_unet", True)) and sample_has_preview:
        collector.add(
            RULE_TEMPORARY_SAFETY,
            [f"{p}.train.train_unet", f"{p}.sample.prompts", f"{p}.sample.samples"],
            "Krea full-tune preview cycles remain temporarily blocked until trainability/state restoration is production-verified.",
            "Disable preview samples for full-tune admission or use an adapter-only recipe with the verified preview path.",
        )
    edit_preview = variant in {"o_edit", "edit", "krea2:o_edit"} or _bool(model_kwargs.get("edit"))
    if krea and edit_preview:
        cfg_fields = [field for scale, field in preview_scales if scale > 1.0]
        if cfg_fields:
            collector.add(
                RULE_TEMPORARY_SAFETY,
                [f"{p}.model.model_kwargs.edit", *cfg_fields],
                "Krea edit CFG remains temporarily blocked until reference CFG expansion is production-verified.",
                "Use guidance_scale 1 for every edit preview item or disable edit conditioning explicitly.",
            )
    if (h3 or krea) and str(train.get("loss_target", "noise") or "noise").lower() == "source":
        collector.add(
            RULE_TEMPORARY_SAFETY,
            [f"{p}.train.loss_target", f"{p}.model.arch"],
            "The source-target path is temporarily blocked on Krea/H3 pending its production sigma/target acceptance evidence.",
            "Use loss_target: noise until the model-specific source path is verified.",
        )
    if (h3 or krea) and _bool(train.get("do_guidance_loss_cfg_zero")):
        collector.add(
            RULE_TEMPORARY_SAFETY,
            [f"{p}.train.do_guidance_loss_cfg_zero", f"{p}.model.arch"],
            "CFG-Zero guidance is temporarily blocked on Krea/H3 pending its production precision acceptance evidence.",
            "Disable do_guidance_loss_cfg_zero explicitly.",
        )
    if krea and ("turbo" in variant or _bool(train.get("train_turbo"))):
        collector.add(
            RULE_TURBO_WARNING,
            [f"{p}.model.arch", f"{p}.train.train_turbo"],
            "Turbo training is a fork-specific recipe; RAW training with Turbo inference remains the recommended baseline.",
            "Prefer RAW training; retain Turbo only when its recipe-specific evidence is understood.",
            severity="warning",
        )

    if not _nonempty(model.get("name_or_path")):
        collector.add(
            "admission.model_required", [f"{p}.model.name_or_path"],
            "A model checkpoint or repository is required before a process can run.",
            "Set model.name_or_path to a local checkpoint or supported repository.",
        )

    legacy_present = "gradient_accumulation_steps" in train
    legacy_value = train.get("gradient_accumulation_steps", 1)
    if legacy_present and _integer(legacy_value, 1) != 1:
        collector.add(
            RULE_LEGACY_ACCUMULATION,
            [f"{p}.train.gradient_accumulation_steps"],
            f"Legacy gradient_accumulation_steps={legacy_value!r} is unsupported for this admission surface.",
            "Set gradient_accumulation_steps: 1 and use train.gradient_accumulation for modern windows; whole-epoch -1 is not translated.",
        )

    if _known_fused_backward(train) and _has_multi_backward(train):
        fields = [f"{p}.train.optimizer"]
        for key in ("gradient_accumulation", "gradient_accumulation_steps", "single_item_batching", "loss_type"):
            if key in train:
                fields.append(f"{p}.train.{key}")
        collector.add(
            RULE_FUSED_BACKWARD,
            fields,
            "The selected fused-backward optimizer updates parameters inside backward and cannot preserve a multi-backward update window.",
            "Use one backward per update, disable single_item_batching/extra accumulation, or select a step-time optimizer. This gate is independent of adaptive-LR mode.",
        )

    loss_type = str(train.get("loss_type", "mse") or "mse").lower()
    if (h3 or krea) and loss_type == "mean_flow":
        collector.add(
            RULE_MEAN_FLOW,
            [f"{p}.train.loss_type", f"{p}.model.arch"],
            "Krea/H3 wrappers do not implement the compatible two-time mean-flow formulation.",
            "Use the model's supported loss_type (normally mse) instead of mean_flow.",
        )

    loss_target = str(train.get("loss_target", "noise") or "noise").lower()
    target_fields: list[str] = []
    if _bool(model_kwargs.get("dopsd")):
        target_fields.append(f"{p}.model.model_kwargs.dopsd")
    if _bool(train.get("do_guidance_loss")):
        target_fields.append(f"{p}.train.do_guidance_loss")
    if _bool(train.get("do_guidance_loss_cfg_zero")):
        target_fields.append(f"{p}.train.do_guidance_loss_cfg_zero")
    if _bool(train.get("inverted_mask_prior")):
        target_fields.append(f"{p}.train.inverted_mask_prior")
    replacement_fields = []
    if loss_target in {"source", "unaugmented"}:
        replacement_fields.append(f"{p}.train.loss_target")
    if _bool(train.get("t0_loss_target")):
        replacement_fields.append(f"{p}.train.t0_loss_target")
    if target_fields and replacement_fields:
        collector.add(
            RULE_TARGET_COLLISION,
            target_fields + replacement_fields,
            "Two enabled branches compete to replace the effective diffusion target: "
            + ", ".join(target_fields + replacement_fields) + ".",
            "Choose one target builder: disable the guidance/teacher/prior replacement or use loss_target: noise and t0_loss_target: false.",
        )
    if loss_target in {"source", "unaugmented"} and _bool(train.get("t0_loss_target")):
        collector.add(
            RULE_TARGET_COLLISION,
            [f"{p}.train.loss_target", f"{p}.train.t0_loss_target"],
            "source/unaugmented target replacement and t0_loss_target overwrite one another.",
            "Choose one target replacement path, not both.",
        )
    noisy_latent_multiplier = _number(train.get("noisy_latent_multiplier", 1.0), 1.0)
    if (h3 or krea) and not math.isclose(noisy_latent_multiplier, 1.0):
        reconstruction_fields: list[str] = []
        if loss_target in {"source", "unaugmented"}:
            reconstruction_fields.append(f"{p}.train.loss_target")
        if _bool(train.get("t0_loss_target")):
            reconstruction_fields.append(f"{p}.train.t0_loss_target")
        if _bool(train.get("do_fft_loss")):
            reconstruction_fields.append(f"{p}.train.do_fft_loss")
        if reconstruction_fields:
            collector.add(
                RULE_TARGET_COLLISION,
                [f"{p}.train.noisy_latent_multiplier", *reconstruction_fields],
                "A non-unit noisy_latent_multiplier changes the input used by source/x0/FFT reconstruction, and no verified inverse preserves those targets.",
                "Set noisy_latent_multiplier to 1 for source, t0, or FFT reconstruction, or use a model-specific verified inverse.",
            )
    if flow and loss_target == "unaugmented":
        collector.add(
            RULE_FLOW_UNAUGMENTED,
            [f"{p}.train.loss_target", f"{p}.model.arch"],
            "The unaugmented branch uses DDPM velocity conversion, which is not a verified target parameterization for this flow wrapper.",
            "Use loss_target: noise or a model-specific verified flow target.",
        )

    process_aux, dataset_aux = _auxiliary_fields(process, datasets, p)
    if h3 and (process_aux or dataset_aux):
        fields = process_aux + [field for _, values in dataset_aux for field in values]
        collector.add(
            RULE_H3_IMAGE_AUXILIARY,
            fields,
            "H3 does not implement image-only perceptual auxiliary losses; this includes still-image/T=1 datasets and preview-only matching work.",
            "Remove the process auxiliary block and every positive dataset override, or use a supported Krea image job.",
        )
    if krea and _bool(model.get("low_vram")) and (process_aux or dataset_aux):
        collector.add(
            RULE_KREA_LOW_VRAM_AUX,
            [f"{p}.model.low_vram"] + process_aux + [field for _, values in dataset_aux for field in values],
            "Differentiable Krea image auxiliaries are incompatible with low_vram tiled decode.",
            "Disable model.low_vram for the auxiliary path, or disable the auxiliary explicitly.",
        )

    # VAE anchor is intentionally fail-closed: this is a local, read-only
    # implementation/checkpoint availability check.
    vae_raw = _mapping(process.get("vae_anchor"))
    if krea and (process.get("vae_anchor") is not None or any("vae_anchor_loss_weight" in ds for ds in datasets)):
        effective_vae = _positive(vae_raw.get("loss_weight", 0.0)) or any(
            _positive(ds.get("vae_anchor_loss_weight", vae_raw.get("loss_weight", 0.0))) for ds in datasets
        )
        if effective_vae:
            vae_path = vae_raw.get("vae_model_path")
            # The canonical loader verifies checkpoint keys and shapes before
            # caching. Static admission checks local implementation availability.
            implementation = Path(__file__).resolve().parent / "models" / "v2" / "vae" / "flux2_kl.py"
            path_exists = (
                _nonempty(vae_path)
                and str(vae_path).lower().endswith(".safetensors")
                and os.path.isfile(os.path.expanduser(str(vae_path)))
            )
            if not implementation.is_file() or not path_exists:
                collector.add(
                    RULE_VAE_ANCHOR,
                    [f"{p}.vae_anchor.loss_weight", f"{p}.vae_anchor.vae_model_path"],
                    "The enabled cross-VAE anchor has no available matching licensed Flux2 encoder implementation and local checkpoint.",
                    "Provide the expected Flux2 encoder source plus an existing licensed ae.safetensors path, or set the effective VAE-anchor weight to 0; no fallback VAE or download is used.",
                )
            else:
                collector.defer(
                    RULE_VAE_ANCHOR,
                    [f"{p}.vae_anchor.vae_model_path"],
                    "Checkpoint keys, channel layout, feature hooks and frozen encoder recipe require loading the matching implementation.",
                    "The VAE-anchor final guard must verify all required keys and the expected feature recipe before cache setup.",
                )
    if _nonempty(model.get("accuracy_recovery_adapter")) and _nonempty(model.get("assistant_lora_path")):
        collector.add(
            RULE_KREA_CONFLICT,
            [f"{p}.model.accuracy_recovery_adapter", f"{p}.model.assistant_lora_path"],
            "Accuracy recovery and assistant LoRA adapters cannot be active together.",
            "Keep one adapter role and explicitly remove the other field.",
        )
    if _bool(train.get("bypass_guidance_embedding")) and _bool(train.get("do_guidance_loss")):
        collector.add(
            RULE_KREA_CONFLICT,
            [f"{p}.train.bypass_guidance_embedding", f"{p}.train.do_guidance_loss"],
            "Guidance-loss training needs the guidance embedding and cannot bypass it.",
            "Disable bypass_guidance_embedding or disable do_guidance_loss explicitly.",
        )
    if base in {"qwen_image_edit", "boogu_image_edit"} and _bool(train.get("unload_text_encoder")):
        collector.add(
            RULE_KREA_CONFLICT,
            [f"{p}.model.arch", f"{p}.train.unload_text_encoder"],
            "This edit model encodes control images through the text encoder and cannot unload it for training.",
            "Keep the text encoder loaded, or disable text-embedding caching only where supported.",
        )
    # Existing generic restrictions are surfaced before model/cache work.
    if _nonempty(model.get("assistant_lora_path")) and _nonempty(model.get("inference_lora_path")):
        collector.add(
            RULE_KREA_CONFLICT,
            [f"{p}.model.assistant_lora_path", f"{p}.model.inference_lora_path"],
            "Training assistant and inference adapter paths cannot be active together.",
            "Keep only the adapter role supported by this process (assistant_lora_path or inference_lora_path).",
        )
    if _bool(train.get("diff_output_preservation")) and _bool(train.get("blank_prompt_preservation")):
        collector.add(
            RULE_KREA_CONFLICT,
            [f"{p}.train.diff_output_preservation", f"{p}.train.blank_prompt_preservation"],
            "Differential-output and blank-prompt preservation are competing preservation modes.",
            "Disable one preservation mode explicitly; the validator does not silently pick one.",
        )
    # BaseSDTrainProcess applies the process-level default to every dataset
    # before constructing DatasetConfig.  Resolve that effective state first;
    # an explicit dataset false cannot undo an enabled process default.
    process_cache_text = _bool(train.get("cache_text_embeddings"))
    effective_cache_text = [
        process_cache_text or _bool(ds.get("cache_text_embeddings"))
        for ds in datasets
    ]
    cached_text = [i for i, enabled in enumerate(effective_cache_text) if enabled]
    if cached_text and len(cached_text) != len(datasets):
        fields = [f"{p}.datasets[{i}].cache_text_embeddings" for i in range(len(datasets))]
        collector.add(
            RULE_TEXT_CACHE_MIX,
            fields,
            "Text embeddings cannot be cached for only part of a process dataset set.",
            "Set cache_text_embeddings consistently for every dataset, or disable it for all datasets.",
        )
    for index, ds in enumerate(datasets):
        if (_bool(ds.get("cache_latents")) or _bool(ds.get("cache_latents_to_disk"))) and (
            _list(ds.get("augments")) or _list(ds.get("augmentations"))
        ):
            collector.add(
                RULE_LATENT_CACHE_AUGMENT,
                [f"{p}.datasets[{index}].cache_latents", f"{p}.datasets[{index}].cache_latents_to_disk", f"{p}.datasets[{index}].augments", f"{p}.datasets[{index}].augmentations"],
                "Requested latent caching cannot preserve stochastic dataset augmentations.",
                "Disable latent caching or remove the stochastic augmentation explicitly; admission never silently turns caching off.",
            )

    # H3 target clocks, standalone audio and Fast/VSA conditioning.
    actual_audio_datasets: list[int] = []
    if h3:
        for index, ds in enumerate(datasets):
            dpath = f"{p}.datasets[{index}]"
            num_frames = _integer(ds.get("num_frames", 1), 1)
            auto_frame_count = _bool(ds.get("auto_frame_count"))
            if (num_frames > 1 or auto_frame_count) and _integer(ds.get("fps", _H3_FPS), _H3_FPS) != _H3_FPS:
                collector.add(
                    RULE_H3_FPS,
                    [f"{dpath}.fps"],
                    "H3's consumed training clock is fixed at 24 fps; a non-24 target fps is not implemented.",
                    "Set the dataset target fps to 24. Source media fps may differ and is resampled by the loader.",
                )
            if num_frames > 1 and not auto_frame_count and (num_frames - _H3_FRAME_REMAINDER) % _H3_FRAME_CHUNK != 0:
                collector.add(
                    RULE_H3_FRAME_GEOMETRY,
                    [f"{dpath}.num_frames"],
                    "Explicit H3 video frame counts must fit the packed 17n+5 frame grid.",
                    "Use 17n+5 frames (5, 22, 39, ...) or enable auto_frame_count so the model snapper trims to the grid.",
                )
            # Probe standalone audio only when this dataset's effective
            # do_audio policy is enabled. Dormant media in an image/video folder
            # must not activate standalone-voice restrictions.
            known_audio = _dataset_has_audio(ds) if _bool(ds.get("do_audio")) else None
            if known_audio is True:
                actual_audio_datasets.append(index)
                if not _bool(ds.get("buckets", True)):
                    collector.add(
                        RULE_H3_AUDIO,
                        [f"{dpath}.buckets", f"{dpath}.dataset_path"],
                        "Standalone H3 voice items require bucketed loading to avoid collating them with visual rows.",
                        "Set buckets: true or separate the standalone voice dataset from visual datasets.",
                    )
                if _number(ds.get("caption_dropout_rate", 0.0), 0.0) > 0:
                    collector.add(
                        RULE_H3_AUDIO,
                        [f"{dpath}.caption_dropout_rate"],
                        "Caption dropout on standalone voice trains empty prompt to that voice.",
                        "Set caption_dropout_rate: 0 for the standalone voice dataset.",
                    )
                collector.defer(
                    RULE_H3_AUDIO_DURATION,
                    [f"{dpath}.dataset_path"],
                    "Audio duration, sample rate, silence and hop alignment require decoding each known voice file.",
                    "Run the H3 audio final guard before latent/cache setup; only 17n+5 / hop-snapped non-silent items are admitted.",
                )
            elif _bool(ds.get("do_audio")):
                collector.defer(
                    RULE_H3_AUDIO_DURATION,
                    [f"{dpath}.do_audio", f"{dpath}.dataset_path"],
                    "The configured dataset may contain standalone voice files, but file content is not knowable from config alone.",
                    "Run the H3 audio final guard before cache setup; it must reject non-grid duration, silence and missing audio content.",
                )

        if variant in {"minimax_h3_vsa", "minimax_h3_fast"} or base in {"minimax_h3_vsa", "minimax_h3_fast"}:
            for index, ds in enumerate(datasets):
                control_fields = []
                for key in ("do_i2v", "control_path", "control_path_1", "control_path_2", "control_path_3", "clip_image_path", "inpaint_path"):
                    raw_value = ds.get(key)
                    active = _bool(raw_value) if key == "do_i2v" else _nonempty_text(raw_value)
                    if active:
                        control_fields.append(f"{p}.datasets[{index}].{key}")
                if control_fields:
                    collector.add(
                        RULE_H3_FAST_CONDITIONING,
                        control_fields + [f"{p}.model.arch"],
                        "Fast/VSA H3 is text-to-video only and does not consume image/reference conditioning controls.",
                        "Remove the i2v/reference controls or select dense H3/Ref2VA for a conditioning path.",
                    )

    # D-OPSD static subset; loaded model/version and concrete optimizer checks stay final.
    dopsd = _bool(model_kwargs.get("dopsd"))
    if dopsd:
        ref_role = "ref2va" in base or "ref2va" in variant or str(model_kwargs.get("partition", "")).lower().startswith("ref2va")
        if not h3 or not ref_role:
            collector.add(
                RULE_DOPSD_VARIANT,
                [f"{p}.model.model_kwargs.dopsd", f"{p}.model.arch", f"{p}.model.model_kwargs.partition"],
                "D-OPSD requires a MiniMax-H3 Ref2VA/Ref2VA-partition model with one loaded DiT.",
                "Use model.arch minimax_h3_ref2va (or partition ref2va/ref2va_pruned), or disable model_kwargs.dopsd.",
            )
        if (not math.isclose(_number(model_kwargs.get("dopsd_loss_mag_weight", 1.0), 1.0), 1.0) or not math.isclose(_number(model_kwargs.get("dopsd_loss_dc_weight", 1.0), 1.0), 1.0)) and loss_type != "mse":
            collector.add(
                RULE_DOPSD_LOSS,
                [f"{p}.model.model_kwargs.dopsd_loss_mag_weight", f"{p}.model.model_kwargs.dopsd_loss_dc_weight", f"{p}.train.loss_type"],
                "D-OPSD decomposed magnitude/direction weighting is defined only for MSE.",
                "Set train.loss_type: mse or restore both D-OPSD split weights to 1.0.",
            )
        teacher_min = _number(model_kwargs.get("dopsd_teacher_sigma_min", 0.0), 0.0)
        teacher_max = _number(model_kwargs.get("dopsd_teacher_sigma_max", 1.0), 1.0)
        ref_mode = str(model_kwargs.get("dopsd_ref_mode", "self") or "self").lower()
        if ref_mode not in {"self", "other"}:
            collector.add(
                RULE_DOPSD_VARIANT,
                [f"{p}.model.model_kwargs.dopsd_ref_mode"],
                f"Unsupported D-OPSD reference mode {ref_mode!r}.",
                "Set dopsd_ref_mode to self or other.",
            )
        group_by = str(model_kwargs.get("dopsd_group_by", "folder") or "folder").lower()
        if group_by != "folder":
            collector.add(
                RULE_DOPSD_VARIANT,
                [f"{p}.model.model_kwargs.dopsd_group_by"],
                f"Unsupported D-OPSD grouping mode {group_by!r}.",
                "Set dopsd_group_by: folder.",
            )
        if _integer(model_kwargs.get("dopsd_ref_count", 1), 1) < 1:
            collector.add(
                RULE_DOPSD_BATCH,
                [f"{p}.model.model_kwargs.dopsd_ref_count"],
                "D-OPSD reference count must be at least one.",
                "Set dopsd_ref_count to a positive integer.",
            )
        if teacher_min >= teacher_max:
            collector.add(
                RULE_DOPSD_BATCH,
                [f"{p}.model.model_kwargs.dopsd_teacher_sigma_min", f"{p}.model.model_kwargs.dopsd_teacher_sigma_max"],
                "D-OPSD teacher sigma minimum must be below its maximum.",
                "Set a valid base-sigma interval, for example [0, 1] or the documented recipe band.",
            )
        if (teacher_min > 0.0 or teacher_max < 1.0 or ref_mode == "other") and _integer(train.get("batch_size", 1), 1) != 1:
            collector.add(
                RULE_DOPSD_BATCH,
                [f"{p}.train.batch_size", f"{p}.model.model_kwargs.dopsd_ref_mode"],
                "Gated or other-photo D-OPSD requires batch_size 1 for per-step reference semantics.",
                "Set train.batch_size: 1 or disable the gated/other-photo D-OPSD mode.",
            )
        if _bool(model_kwargs.get("dopsd_identity_first")):
            if _integer(model_kwargs.get("dopsd_identity_first_steps", -1), -1) == 0:
                collector.add(
                    RULE_DOPSD_OPTIMIZER,
                    [f"{p}.model.model_kwargs.dopsd_identity_first_steps"],
                    "D-OPSD identity-first resolves to zero teacher updates.",
                    "Set a positive step count or -1 for the documented automatic schedule.",
                )
            supported = _known_step_scale_support(train)
            if supported is False:
                collector.add(
                    RULE_DOPSD_OPTIMIZER,
                    [f"{p}.model.model_kwargs.dopsd_identity_first", f"{p}.train.optimizer"],
                    "D-OPSD identity-first needs a faithful optimizer step-scale capability.",
                    "Choose an optimizer classified for step scaling or disable dopsd_identity_first.",
                )
            elif supported is None:
                collector.defer(
                    RULE_DOPSD_OPTIMIZER,
                    [f"{p}.model.model_kwargs.dopsd_identity_first", f"{p}.train.optimizer"],
                    "The selected optimizer's concrete step-scale capability is unknown until construction.",
                    "Keep the final OptimizerRuntimeAdapter identity-first guard enabled; unknown optimizers must fail closed.",
                )
        if ref_mode == "self" and actual_audio_datasets:
            collector.add(
                RULE_H3_DOPSD_AUDIO,
                [f"{p}.model.model_kwargs.dopsd_ref_mode"] + [f"{p}.datasets[{i}].dataset_path" for i in actual_audio_datasets],
                "Self-D-OPSD uses visual target pixels and cannot teach a standalone-voice target dataset.",
                "Use dopsd_ref_mode: other with an eligible still-photo dataset, or separate voice training from self-D-OPSD.",
            )

    # TREAD static gates; block known impossible variants and defer block-count checks.
    tread_ratio = _number(train.get("tread_ratio", 0.0), 0.0)
    if tread_ratio > 0.0:
        if not h3:
            collector.add(
                RULE_TREAD_VARIANT,
                [f"{p}.train.tread_ratio", f"{p}.model.arch"],
                "TREAD token routing is MiniMax-H3 only.",
                "Set tread_ratio: 0 or select a supported dense H3 architecture.",
            )
        elif variant in {"minimax_h3_vsa", "minimax_h3_fast"}:
            collector.add(
                RULE_TREAD_VARIANT,
                [f"{p}.train.tread_ratio", f"{p}.model.arch"],
                "TREAD has no routed VSA/Fast context and cannot be combined with the sparse Fast wrapper.",
                "Set tread_ratio: 0 or use a dense H3 checkpoint.",
            )
        else:
            start = _integer(train.get("tread_start", 2), 2)
            end = _integer(train.get("tread_end", 47), 47)
            if start < 0 or start >= end:
                collector.add(
                    RULE_TREAD_SPAN,
                    [f"{p}.train.tread_start", f"{p}.train.tread_end"],
                    "TREAD start must be non-negative and below tread_end.",
                    "Set a valid half-open block span; the loaded transformer final guard also checks its block count.",
                )
            collector.defer(
                RULE_TREAD_SPAN,
                [f"{p}.train.tread_start", f"{p}.train.tread_end", f"{p}.model.arch"],
                "TREAD's post-rejoin minimum and .blocks length are runtime transformer facts.",
                "Retain bind_tread/validate_tread_span and fail closed before the first batch.",
            )

    if h3 and _nonempty(model.get("preview_lora_path")):
        collector.defer(
            RULE_PREVIEW_LORA,
            [f"{p}.model.preview_lora_path"],
            "H3 preview LoRA checkpoint state and compatibility cannot be established from fields alone.",
            "The final model loader must validate the local/managed checkpoint without silently downloading or merging it.",
        )
    if _nonempty(model.get("preview_lora_path")) and _nonempty(model.get("inference_lora_path")):
        collector.add(
            RULE_PREVIEW_LORA,
            [f"{p}.model.preview_lora_path", f"{p}.model.inference_lora_path"],
            "H3 preview_lora_path and inference_lora_path are mutually exclusive roles.",
            "Keep preview_lora_path for H3 sampling or use inference_lora_path for the other supported model path, not both.",
        )

    # Runtime facts that must never be represented as a static pass.
    collector.defer(
        RULE_UNCERTIFIED_RUNTIME,
        [f"{p}.model.name_or_path"],
        "Model checkpoint role, transformer geometry, optimizer capabilities, auxiliary dependency state, and actual media content are runtime facts.",
        "Run the model/optimizer/dataset final guards after static admission and before the first batch; deferred checks are not a pass.",
    )

    return {
        "process_index": process_index,
        "arch": raw_arch,
        "base_arch": base,
        "variant": variant,
        "dataset_count": len(datasets),
        "dataset_audio_indices": actual_audio_datasets,
        "effective": {
            "loss_type": loss_type,
            "loss_target": loss_target,
            "legacy_gradient_accumulation_steps": legacy_value,
            "gradient_accumulation": _integer(train.get("gradient_accumulation", 1), 1),
            "optimizer": str(train.get("optimizer", "adamw")),
            "cache_text_embeddings": effective_cache_text,
            "trainability_roles": trainability_roles,
            "is_fine_tuning": is_fine_tuning,
            "h3": h3,
            "krea": krea,
        },
    }


def _unwrap_root(raw_config: Any) -> Mapping[str, Any]:
    raw = _mapping(raw_config)
    nested = raw.get("config")
    return _mapping(nested) if isinstance(nested, Mapping) else raw


def collect_admission_diagnostics(raw_config: Mapping[str, Any], *, source: str = "api") -> dict[str, Any]:
    """Resolve every process/dataset and return JSON-safe diagnostics.

    Input is copied before traversal to guarantee that callers can safely reuse
    their in-memory config.  No model, optimizer, GPU, loader, cache, or job
    execution is touched.
    """
    copied = copy.deepcopy(raw_config)
    root = _unwrap_root(copied)
    collector = _Collector(source=source)
    processes = _list(root.get("process"))
    if not processes:
        collector.add(
            "admission.process_required",
            ["config.process"],
            "A training job must contain at least one process.",
            "Add a process object with model, train and datasets sections.",
        )
    resolved = []
    for index, process in enumerate(processes):
        if not isinstance(process, Mapping):
            collector.add(
                "admission.process_shape",
                [f"config.process[{index}]"],
                "Each process must be a mapping.",
                "Replace the process entry with a YAML/JSON object.",
            )
            continue
        resolved.append(_validate_process(process, index, collector))
    diagnostics = collector.diagnostics
    return {
        "valid": not any(d.get("severity") == "error" for d in diagnostics),
        "diagnostics": diagnostics,
        "deferred": collector.deferred,
        "resolved": resolved,
        "source": source,
    }


def collect_process_admission_diagnostics(process: Mapping[str, Any], *, process_index: int = 0, source: str = "runtime") -> dict[str, Any]:
    """Convenience wrapper used by ``BaseSDTrainProcess`` before model load."""
    copied = copy.deepcopy(process)
    collector = _Collector(source=source)
    resolved = [_validate_process(copied, process_index, collector)]
    diagnostics = collector.diagnostics
    return {
        "valid": not any(d.get("severity") == "error" for d in diagnostics),
        "diagnostics": diagnostics,
        "deferred": collector.deferred,
        "resolved": resolved,
        "source": source,
    }


def raise_for_process_admission(process: Mapping[str, Any], *, process_index: int = 0, source: str = "runtime") -> dict[str, Any]:
    result = collect_process_admission_diagnostics(process, process_index=process_index, source=source)
    if not result["valid"]:
        raise AdmissionError(result)
    return result


def _load_input(path: str, *, use_stdin: bool = False) -> Any:
    text = sys.stdin.read() if use_stdin or path == "-" else Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise SystemExit("YAML input requires PyYAML; provide JSON or install the managed YAML dependency") from exc
        return yaml.safe_load(text)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Side-effect-free Krea/H3 config admission diagnostics (no model/GPU/cache/job execution)."
    )
    parser.add_argument("config", nargs="?", default="-", help="YAML or JSON job config path, or - for stdin")
    parser.add_argument("--stdin", action="store_true", help="Read YAML/JSON job config from stdin")
    parser.add_argument("--source", default="cli", help="diagnostic source label (default: cli)")
    args = parser.parse_args(argv)
    result = collect_admission_diagnostics(_load_input(args.config, use_stdin=args.stdin), source=args.source)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 2




if __name__ == "__main__":
    sys.exit(main())
