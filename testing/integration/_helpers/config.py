"""Krea 2 config builder for Phase 1 integration tests."""
import os

DEFAULT_MODEL = os.environ.get("AI_TOOLKIT_KREA_MODEL", "krea/Krea-2-Raw")


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
