"""Training-only guards and loss seams for opt-in identity isolation."""

from __future__ import annotations

import torch

from toolkit.optimizer_runtime import unwrap_optimizer


def dop_selected(config, completed_update_id: int, legacy_enabled: bool) -> bool:
    if config is None or not config.dop_enabled:
        return bool(legacy_enabled)
    return bool(legacy_enabled and completed_update_id % config.dop_interval == 0)


def validate_isolation_process(process, config) -> None:
    """Validate declared compatibility without loading models or preparing assets."""
    if config is None or not config.active:
        return
    import copy
    from types import SimpleNamespace

    from toolkit.config_modules import ModelConfig, NetworkConfig, TrainConfig

    train_raw = copy.deepcopy(process.get("train", {}))
    if config.dop_enabled:
        train_raw.setdefault("diff_output_preservation", True)
    network_raw = process.get("network")
    trainer = SimpleNamespace(
        isolation_config=config,
        train_config=TrainConfig(**train_raw),
        model_config=ModelConfig(**copy.deepcopy(process.get("model", {}))),
        network_config=NetworkConfig(**network_raw)
        if network_raw is not None
        else None,
        adapter_config=process.get("adapter"),
        embed_config=process.get("embedding"),
        decorator_config=process.get("decorator"),
        config=process,
    )
    validate_isolation_training(trainer)


def validate_isolation_training(trainer) -> None:
    config = getattr(trainer, "isolation_config", None)
    if config is None:
        return
    train = trainer.train_config
    model = trainer.model_config
    problems = []
    raw_model = getattr(trainer, "config", {}).get("model", {})
    variant = str(raw_model.get("arch", model.arch)).partition(":")[2].lower()
    source = (
        str(raw_model.get("name_or_path", getattr(model, "name_or_path", "")))
        .rstrip("/")
        .lower()
    )
    if variant not in ("", "raw") or source == "krea/krea-2-turbo":
        problems.append("the declared model role must be RAW, not turbo/edit")
    if any(
        getattr(model, name, None)
        for name in ("assistant_lora_path", "accuracy_recovery_adapter")
    ):
        problems.append(
            "training assistant/recovery adapters are not validated for isolation"
        )
    if model.arch != "krea2" or getattr(model, "model_kwargs", {}).get("edit", False):
        problems.append("only Krea 2 RAW text-to-image is supported")
    if trainer.network_config is None or trainer.network_config.type != "lora":
        problems.append("network.type must be lora (not full fine-tuning)")
    if any(
        getattr(trainer, name, None) is not None
        for name in ("adapter_config", "embed_config", "decorator_config")
    ):
        problems.append("adapters, embeddings and decorators are unsupported")
    if train.train_text_encoder:
        problems.append("train_text_encoder must be false")
    if train.loss_type == "mean_flow":
        problems.append("mean_flow uses unmatched preservation inputs")
    if any(
        getattr(train, name, False)
        for name in ("do_guidance_loss", "do_differential_guidance", "train_turbo")
    ):
        problems.append("guided and turbo objectives are unsupported")
    if getattr(train, "adapter_assist_name_or_path", None) is not None:
        problems.append("assistant adapters are unsupported")
    if getattr(train, "blank_prompt_preservation", False):
        problems.append("blank_prompt_preservation conflicts with isolation")
    raw = getattr(trainer, "config", {})
    if str(train.optimizer).lower() not in ("adamw", "rose"):
        problems.append(
            "the configured optimizer must be adamw or rose; actual optimizer capability is checked at runtime"
        )
    if any(
        dataset.get("is_reg") or dataset.get("prior_reg")
        for dataset in raw.get("datasets", [])
    ):
        problems.append(
            "is_reg/prior_reg datasets use an unvalidated separate preservation/reference path"
        )
    if (
        raw.get("train", {}).get("dopsd")
        or raw.get("dopsd")
        or getattr(model, "model_kwargs", {}).get("dopsd")
    ):
        problems.append("D-OPSD cannot share isolation's reference prediction")
    if train.diff_output_preservation and any(
        getattr(train, name, False)
        for name in ("inverted_mask_prior", "correct_pred_norm", "do_prior_divergence")
    ):
        problems.append(
            "inverted_mask_prior/correct_pred_norm/do_prior_divergence require a different reference caption"
        )
    if config.composites and config.composite_lr_multiplier != 1.0:
        if any(
            int(getattr(train, name, 1)) != 1
            for name in ("gradient_accumulation", "gradient_accumulation_steps")
        ):
            problems.append(
                "composite LR scaling requires gradient_accumulation=1 and gradient_accumulation_steps=1"
            )
    if problems:
        raise ValueError("identity_isolation: " + "; ".join(problems))


def validate_isolation_optimizer(config, optimizer, runtime) -> None:
    if config is None:
        return
    from toolkit.optimizers.rose import Rose

    actual = unwrap_optimizer(optimizer)
    if (
        type(actual) not in (torch.optim.AdamW, Rose)
        or runtime.update_phase != "step"
        or not runtime.supports_step_scale
    ):
        raise ValueError(
            "identity_isolation requires actual torch AdamW or toolkit Rose with step-time optimizer runtime; fused-backward is unsupported"
        )


def isolation_batch_category(config, batches, previous=None):
    if config is None:
        return None
    categories = set()
    for batch in batches:
        for item in batch.file_items:
            category = getattr(item.dataset_config, "isolation_category", "normal")
            if category not in ("normal", "composite"):
                raise ValueError(
                    f"identity_isolation: unknown dataset category {category!r}"
                )
            categories.add(category)
    if previous is not None:
        categories.add(previous)
    if len(categories) > 1:
        if config.composite_lr_multiplier != 1.0:
            raise ValueError(
                "identity_isolation composite LR requires a homogeneous normal/composite batch and optimizer window before backward"
            )
        return "mixed"
    return next(iter(categories), previous)


def isolation_lr_multiplier(trainer) -> float:
    config = getattr(trainer, "isolation_config", None)
    if (
        config is None
        or getattr(trainer, "_isolation_window_category", None) != "composite"
    ):
        return 1.0
    return float(config.composite_lr_multiplier)


def preservation_mse(student, reference, multiplier, *, isolation=False):
    if isolation:
        raw = torch.nn.functional.mse_loss(student.float(), reference.detach().float())
    else:
        raw = torch.nn.functional.mse_loss(student, reference)
    return raw, raw * multiplier
