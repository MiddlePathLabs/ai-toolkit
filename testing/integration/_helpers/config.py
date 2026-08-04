"""Krea 2 config builders for Phase 1 (noising) and Phase 2 (depth) integration tests."""
import os

DEFAULT_MODEL = os.environ.get("AI_TOOLKIT_KREA_MODEL", "krea/Krea-2-Raw")
DEFAULT_DA2_MODEL = os.environ.get(
    "AI_TOOLKIT_DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Small-hf"
)
DEFAULT_DA2_INPUT_SIZE = int(os.environ.get("AI_TOOLKIT_DEPTH_INPUT_SIZE", "518"))


def build_krea_config(
    *,
    name: str,
    dataset_path: str,
    output_path: str,
    steps: int = 2,
    weight_noise: dict | None = None,
    gradient_noise: dict | None = None,
    gradient_accumulation: int = 1,
    gradient_accumulation_steps: int = 1,
    use_ema: bool = False,
    ema_decay: float = 0.99,
    pretrained_lora_path: str | None = None,
    model_path: str | None = None,
    lora_rank: int = 16,
    lora_alpha: int = 16,
    lr: float = 1e-4,
    batch_size: int = 1,
    resolution: int = 512,
    cache_latents_to_disk: bool = False,
    save_every: int = 1,
    seed: int = 42,
) -> dict:
    """Build a complete AI-Toolkit job config dict for a Krea 2 LoRA training run.

    Only weight_noise, gradient_noise, steps, accumulation, and EMA vary between
    matched runs. Everything else is fixed for determinism.
    """
    model = model_path or DEFAULT_MODEL
    network = {
        "type": "lora",
        "linear": lora_rank,
        "linear_alpha": lora_alpha,
    }
    if pretrained_lora_path is not None:
        network["pretrained_lora_path"] = pretrained_lora_path

    config = {
        "job": "extension",
        "config": {
            "name": name,
            "process": [
                {
                    "type": "sd_trainer",
                    "training_folder": output_path,
                    "device": "cuda:0",
                    "trigger_word": "ph0t0s4m",
                    "network": network,
                    "save": {
                        "dtype": "float16",
                        "save_every": save_every,
                        "max_step_saves_to_keep": 4,
                    },
                    "datasets": [
                        {
                            "folder_path": dataset_path,
                            "caption_ext": "txt",
                            "caption_dropout_rate": 0.0,
                            "shuffle_tokens": False,
                            "cache_latents_to_disk": cache_latents_to_disk,
                            "resolution": [resolution],
                            "buckets": True,
                            "bucket_tolerance": 16,
                        }
                    ],
                    "train": {
                        "batch_size": batch_size,
                        "steps": steps,
                        "gradient_accumulation": gradient_accumulation,
                        "gradient_accumulation_steps": gradient_accumulation_steps,
                        "noise_scheduler": "flowmatch",
                        "optimizer": "adamw",
                        "lr": lr,
                        "dtype": "bf16",
                        "seed": seed,
                        "disable_sampling": True,
                        "ema_config": {
                            "use_ema": use_ema,
                            "ema_decay": ema_decay,
                        },
                        "weight_noise": weight_noise or {"enabled": False},
                        "gradient_noise": gradient_noise or {"enabled": False},
                        "max_grad_norm": 1.0,
                    },
                    "model": {
                        "name_or_path": model,
                        "arch": "krea2",
                        "low_vram": False,
                        "quantize": False,
                        "dtype": "bf16",
                    },
                }
            ],
        },
    }
    return config


# ---------------------------------------------------------------------------
# Phase 2: depth-anchor configs
# ---------------------------------------------------------------------------

def _default_depth_block(
    *,
    loss_weight=0.001,
    model_id=None,
    input_size=None,
    mask_source="none",
    preview_every=0,
    preview_only=False,
    preview_max_keep=10,
    ssi_weight=1.0,
    grad_weight=0.5,
    grad_scales=4,
    pixel_blur_sigma=0.0,
    grad_checkpoint=True,
    loss_min_t=0.0,
    loss_max_t=1.0,
):
    """Build a depth_consistency process block with explicit, recorded values.

    The shipped production default is DA2-Small at 518 (see
    DepthConsistencyConfig in toolkit/config_modules.py). The accepted manual
    Phase 2 evidence used DA2-Large at 518 for the strict depth-only run; that
    is selected by passing model_id='depth-anything/Depth-Anything-V2-Large-hf'.
    Nothing here changes a production default -- it only records what the test
    selected.
    """
    return {
        "loss_weight": float(loss_weight),
        "loss_min_t": float(loss_min_t),
        "loss_max_t": float(loss_max_t),
        "model_id": model_id or DEFAULT_DA2_MODEL,
        "input_size": int(input_size or DEFAULT_DA2_INPUT_SIZE),
        "pixel_blur_sigma": float(pixel_blur_sigma),
        "ssi_weight": float(ssi_weight),
        "grad_weight": float(grad_weight),
        "grad_scales": int(grad_scales),
        "mask_source": mask_source,
        "grad_checkpoint": bool(grad_checkpoint),
        "preview_every": int(preview_every),
        "preview_only": bool(preview_only),
        "preview_max_keep": int(preview_max_keep),
    }


def build_krea_depth_config(
    *,
    name: str,
    dataset_path: str,
    output_path: str,
    steps: int = 2,
    # Depth activation. depth=None disables the feature entirely (no perceptor
    # load, no cache). A dict is placed verbatim under process.depth_consistency.
    depth: dict | None = None,
    # Per-dataset depth controls (None -> absent -> inherits process/global).
    dataset_depth_weight=None,
    dataset_loss_min_t=None,
    dataset_loss_max_t=None,
    dataset_loss_split=None,
    # Diffusion-zeroing for the strict depth-only proof. The accepted evidence
    # zeroes the diffusion term via dataset loss_multiplier=0.0 (the depth term
    # is added AFTER loss*loss_multiplier, so it survives). None leaves the
    # default (1.0) in place.
    loss_multiplier=None,
    global_loss_split="sentinel_absent",
    pretrained_lora_path: str | None = None,
    model_path: str | None = None,
    lora_rank: int = 16,
    lora_alpha: int = 16,
    optimizer: str = "adamw",
    lr: float = 1e-4,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    resolution=(512,),
    cache_latents_to_disk: bool = False,
    save_every: int = 1,
    max_step_saves_to_keep: int = 4,
    seed: int = 42,
    quantize: bool = False,
    low_vram: bool = False,
    use_lokr: bool = False,
):
    """Build a Krea 2 job config dict with a Phase 2 depth_consistency block.

    Mirrors the accepted manual evidence configs (see
    evidence/krea2-phase2-*.yml) while staying compatible with the Phase 1
    runner helpers. ``depth=None`` produces a depth-inactive job so the same
    builder covers the disabled-baseline test.
    """
    model = model_path or DEFAULT_MODEL
    if use_lokr:
        # Full-rank LoKr network -- the accepted Phase 2 evidence config (see
        # evidence/krea2-phase2-02-depth-only-*.yml). Proven to receive the
        # depth-anchor gradient (512 trainable tensors move under depth-only).
        network = {
            "type": "lora",
            "linear": 32,
            "linear_alpha": 32,
            "conv": 16,
            "conv_alpha": 16,
            "lokr_full_rank": True,
            "lokr_factor": -1,
            "network_kwargs": {"ignore_if_contains": []},
        }
    else:
        network = {
            "type": "lora",
            "linear": lora_rank,
            "linear_alpha": lora_alpha,
        }
    if pretrained_lora_path is not None:
        network["pretrained_lora_path"] = pretrained_lora_path

    dataset = {
        "folder_path": dataset_path,
        "caption_ext": "txt",
        "caption_dropout_rate": 0.0,
        "shuffle_tokens": False,
        "cache_latents_to_disk": cache_latents_to_disk,
        "resolution": list(resolution),
        "buckets": True,
        "bucket_tolerance": 16,
        "random_crop": False,
        "random_scale": False,
        "augments": [],
    }
    # Per-dataset depth scalars: only add keys that are explicitly set so an
    # absent key stays "inherit" rather than being forced to a value.
    if dataset_depth_weight is not None:
        dataset["depth_loss_weight"] = dataset_depth_weight
    if dataset_loss_min_t is not None:
        dataset["depth_loss_min_t"] = dataset_loss_min_t
    if dataset_loss_max_t is not None:
        dataset["depth_loss_max_t"] = dataset_loss_max_t
    if dataset_loss_split is not None:
        dataset["loss_split"] = dataset_loss_split
    if loss_multiplier is not None:
        dataset["loss_multiplier"] = loss_multiplier

    train = {
        "batch_size": batch_size,
        "steps": steps,
        "gradient_accumulation": 1,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "noise_scheduler": "flowmatch",
        "optimizer": optimizer,
        "lr": lr,
        "dtype": "bf16",
        "seed": seed,
        "disable_sampling": True,
        "ema_config": {"use_ema": False, "ema_decay": 0.99},
        "weight_noise": {"enabled": False},
        "gradient_noise": {"enabled": False},
        "max_grad_norm": 1.0,
    }
    # train.loss_split: a sentinel keeps the key ABSENT (Auto). Passing None
    # explicitly writes loss_split: null (force-off / sum). A string forces it.
    if global_loss_split != "sentinel_absent":
        train["loss_split"] = global_loss_split

    process = {
        "type": "sd_trainer",
        "training_folder": output_path,
        "device": "cuda:0",
        "trigger_word": "ph0t0s4m",
        "network": network,
        "save": {
            "dtype": "bf16",
            "save_every": save_every,
            "max_step_saves_to_keep": max_step_saves_to_keep,
        },
        "datasets": [dataset],
        "train": train,
        "model": {
            "name_or_path": model,
            "arch": "krea2",
            "low_vram": low_vram,
            "quantize": quantize,
            "dtype": "bf16",
        },
    }
    if depth is not None:
        process["depth_consistency"] = depth

    return {"job": "extension", "config": {"name": name, "process": [process]}}


def depth_disabled_block():
    """A depth block that is fully inert (loss_weight 0, preview off)."""
    return _default_depth_block(loss_weight=0.0, preview_only=False, preview_every=0)


def depth_active_block(**overrides):
    """A depth block with depth anchoring on, mask_source none, previews off."""
    base = _default_depth_block(loss_weight=0.001, preview_every=0, preview_only=False)
    base.update(overrides)
    return base


def depth_preview_only_block(**overrides):
    """A depth block in preview-only mode (DA2 loads, no loss contribution)."""
    base = _default_depth_block(
        loss_weight=0.0, preview_only=True, preview_every=1, preview_max_keep=10
    )
    base.update(overrides)
    return base
