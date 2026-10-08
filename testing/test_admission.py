"""CPU-only static Krea/H3 admission contract tests."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from toolkit.admission import (
    RULE_DOPSD_BATCH,
    RULE_DOPSD_VARIANT,
    RULE_FUSED_BACKWARD,
    RULE_FUSED_CLIP_NOOP,
    RULE_H3_AUDIO,
    RULE_H3_FAST_CONDITIONING,
    RULE_H3_FRAME_GEOMETRY,
    RULE_H3_FPS,
    RULE_H3_IMAGE_AUXILIARY,
    RULE_KREA_CONFLICT,
    RULE_KREA_LOW_VRAM_AUX,
    RULE_QWEN_LOW_VRAM_AUX,
    RULE_QWEN_RGBA_ANCHOR,
    RULE_LATENT_CACHE_AUGMENT,
    RULE_TEMPORARY_SAFETY,
    RULE_LEGACY_ACCUMULATION,
    RULE_MEAN_FLOW,
    RULE_RESUME_DETERMINISM,
    RULE_TARGET_COLLISION,
    RULE_TEXT_CACHE_MIX,
    RULE_VAE_ANCHOR,
    collect_admission_diagnostics,
)


FIXTURE = Path(__file__).parent / "fixtures" / "admission" / "accepted_h3_character.yaml"
KREA_FIXTURE = Path(__file__).parent / "fixtures" / "admission" / "accepted_krea_lora.yaml"


def _process(**overrides):
    process = {
        "type": "diffusion_trainer",
        "model": {
            "name_or_path": "scrubbed-checkpoint",
            "arch": "minimax_h3_ref2va",
            "model_kwargs": {"partition": "ref2va_pruned"},
        },
        "train": {"optimizer": "adamw", "loss_type": "mse", "batch_size": 1},
        "datasets": [{"dataset_path": "not-present", "buckets": True}],
    }
    for key, value in overrides.items():
        process[key] = value
    return process


def _ids(result):
    return {item["rule_id"] for item in result["diagnostics"]}


def test_scrubbed_accepted_h3_recipe_is_admitted_without_runtime_claims():
    raw = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    result = collect_admission_diagnostics(raw, source="fixture")
    assert result["valid"]
    # no error diagnostics; the fused automagic3 recipe does carry the (truthful)
    # clip-noop warning -- grad-norm clipping never runs under fused backward
    assert not [d for d in result["diagnostics"] if d["severity"] == "error"]
    assert {d["rule_id"] for d in result["diagnostics"]} == {RULE_FUSED_CLIP_NOOP}
    assert result["resolved"][0]["variant"] == "minimax_h3_ref2va"
    assert any(item["rule_id"] == "admission.h3_audio_duration_runtime" for item in result["deferred"])
    assert any(item["rule_id"] == "admission.runtime_deferred" for item in result["deferred"])


def test_scrubbed_accepted_krea_lora_recipe_keeps_preview_admission():
    raw = yaml.safe_load(KREA_FIXTURE.read_text(encoding="utf-8"))
    result = collect_admission_diagnostics(raw, source="fixture")
    assert result["valid"]
    assert not result["diagnostics"]
    effective = result["resolved"][0]["effective"]
    assert effective["trainability_roles"] == ["network"]
    assert effective["is_fine_tuning"] is False


def test_process_text_cache_default_resolves_before_mixed_cache_check():
    process = _process(
        train={
            "optimizer": "adamw",
            "cache_text_embeddings": True,
        },
        datasets=[
            {"dataset_path": "one", "cache_text_embeddings": False},
            {"dataset_path": "two", "cache_text_embeddings": True},
        ],
    )
    result = collect_admission_diagnostics({"process": [process]})
    assert result["valid"]
    assert RULE_TEXT_CACHE_MIX not in _ids(result)
    assert result["resolved"][0]["effective"]["cache_text_embeddings"] == [True, True]


def test_diagnostics_are_side_effect_free_and_cover_every_process():
    raw = {
        "config": {
            "process": [
                _process(),
                _process(train={"optimizer": "automagic2", "gradient_accumulation": 2}),
            ]
        }
    }
    before = copy.deepcopy(raw)
    result = collect_admission_diagnostics(raw)
    assert raw == before
    assert not result["valid"]
    fused = [d for d in result["diagnostics"] if d["rule_id"] == RULE_FUSED_BACKWARD]
    assert fused and any("config.process[1].train.gradient_accumulation" in fused[0]["fields"] for _ in [0])
    assert result["resolved"][0]["process_index"] == 0
    assert result["resolved"][1]["process_index"] == 1


def test_fused_backward_clip_noop_warns_even_on_implicit_default():
    # the runtime default (max_grad_norm=1.0) also never clips under fused
    # backward, and the UI never writes the key -- key presence alone would
    # miss every UI job
    raw = {"config": {"process": [_process(train={"optimizer": "automagic2"})]}}
    assert RULE_FUSED_CLIP_NOOP in _ids(collect_admission_diagnostics(raw))

    raw = {"config": {"process": [_process(train={"optimizer": "automagic2", "max_grad_norm": 1.0})]}}
    result = collect_admission_diagnostics(raw)
    assert RULE_FUSED_CLIP_NOOP in _ids(result)
    warn = [d for d in result["diagnostics"] if d["rule_id"] == RULE_FUSED_CLIP_NOOP][0]
    assert warn["severity"] == "warning"


def test_fused_backward_clip_noop_silent_when_disabled_or_step_time():
    raw = {"config": {"process": [_process(train={"optimizer": "automagic2", "max_grad_norm": 0})]}}
    assert RULE_FUSED_CLIP_NOOP not in _ids(collect_admission_diagnostics(raw))

    raw = {"config": {"process": [_process(train={"optimizer": "adamw8bit", "max_grad_norm": 1.0})]}}
    assert RULE_FUSED_CLIP_NOOP not in _ids(collect_admission_diagnostics(raw))


def test_permanent_accumulation_mean_flow_and_target_collision_rules():
    process = _process(
        train={
            "optimizer": "automagic3",
            "optimizer_params": {"fused": True},
            "gradient_accumulation_steps": -1,
            "gradient_accumulation": 1,
            "loss_type": "mean_flow",
            "loss_target": "source",
            "t0_loss_target": True,
            "do_guidance_loss": True,
        }
    )
    ids = _ids(collect_admission_diagnostics({"process": [process]}))
    assert {RULE_LEGACY_ACCUMULATION, RULE_FUSED_BACKWARD, RULE_MEAN_FLOW, RULE_TARGET_COLLISION} <= ids


def test_h3_auxiliary_rejects_process_and_dataset_overrides_including_stills():
    process = _process(
        datasets=[
            {"dataset_path": "photo", "num_frames": 1, "depth_loss_weight": 0.1},
            {"dataset_path": "photo2", "num_frames": 1, "identity_loss_weight": 0.2},
        ],
        depth_consistency={"preview_only": True},
        normal_id={"loss_weight": 0.0, "preview_only": True},
    )
    result = collect_admission_diagnostics({"process": [process]})
    assert RULE_H3_IMAGE_AUXILIARY in _ids(result)
    fields = next(d["fields"] for d in result["diagnostics"] if d["rule_id"] == RULE_H3_IMAGE_AUXILIARY)
    assert "config.process[0].datasets[0].depth_loss_weight" in fields
    assert "config.process[0].datasets[1].identity_loss_weight" in fields


def test_h3_clock_geometry_and_fast_conditioning_are_precise():
    process = _process(
        model={"name_or_path": "fast", "arch": "minimax_h3_vsa"},
        datasets=[
            {
                "dataset_path": "video",
                "num_frames": 23,
                "fps": 30,
                "auto_frame_count": False,
                "do_i2v": True,
                "control_path": "control.png",
            }
        ],
    )
    ids = _ids(collect_admission_diagnostics({"process": [process]}))
    assert {RULE_H3_FPS, RULE_H3_FRAME_GEOMETRY, RULE_H3_FAST_CONDITIONING} <= ids


def test_h3_audio_is_not_blanket_blocked_but_known_voice_files_are_checked():
    process = _process(
        datasets=[{"dataset_path": "unknown-at-static-time", "do_audio": True, "buckets": False}]
    )
    result = collect_admission_diagnostics({"process": [process]})
    assert result["valid"]
    assert RULE_H3_AUDIO not in _ids(result)
    assert any(item["rule_id"] == "admission.h3_audio_duration_runtime" for item in result["deferred"])


def test_dopsd_static_variant_and_batch_rules():
    process = _process(
        model={"name_or_path": "base", "arch": "minimax_h3", "model_kwargs": {"dopsd": True, "dopsd_ref_mode": "other"}},
        train={"optimizer": "adamw", "batch_size": 2, "loss_type": "mse"},
    )
    ids = _ids(collect_admission_diagnostics({"process": [process]}))
    assert RULE_DOPSD_VARIANT in ids
    assert RULE_DOPSD_BATCH in ids


def test_krea_conflicts_cache_augmentation_low_vram_and_vae_anchor():
    process = {
        "model": {"name_or_path": "krea", "arch": "krea2", "low_vram": True},
        "train": {
            "optimizer": "adamw",
            "diff_output_preservation": True,
            "blank_prompt_preservation": True,
        },
        "datasets": [
            {
                "dataset_path": "images",
                "cache_text_embeddings": True,
                "cache_latents": True,
                "augments": ["random_crop"],
            },
            {"dataset_path": "images2", "cache_text_embeddings": False},
        ],


        "depth_consistency": {"loss_weight": 0.1},
        "vae_anchor": {"loss_weight": 0.1, "vae_model_path": "random.safetensors"},
    }
    ids = _ids(collect_admission_diagnostics({"process": [process]}))
    assert {RULE_KREA_CONFLICT, RULE_KREA_LOW_VRAM_AUX, RULE_LATENT_CACHE_AUGMENT, RULE_VAE_ANCHOR} <= ids


def test_qwen_vae_anchor_backend_low_vram_and_rgba_rules(tmp_path):
    # Qwen-Image 2.1 now gets the same fail-closed VAE-anchor backend check as
    # Krea 2, plus its own low_vram (tiled decode) and RGBA (4-channel decode
    # vs the 3-channel Flux 2 encoder) incompatibilities.
    process = {
        "model": {
            "name_or_path": "Comfy-Org/Qwen-Image-2.1",
            "arch": "qwen_image_2",
            "low_vram": True,
            "model_kwargs": {"rgba": True},
        },
        "train": {"optimizer": "adamw", "loss_type": "mse", "batch_size": 1},
        "datasets": [{"dataset_path": "images"}],
        "vae_anchor": {"loss_weight": 0.1, "vae_model_path": "missing.safetensors"},
    }
    ids = _ids(collect_admission_diagnostics({"process": [process]}))
    assert {RULE_VAE_ANCHOR, RULE_QWEN_LOW_VRAM_AUX, RULE_QWEN_RGBA_ANCHOR} <= ids

    # clean job: low_vram off, rgba off, existing local checkpoint -> the
    # backend rule defers its load-time contract instead of erroring
    ckpt = tmp_path / "ae.safetensors"
    ckpt.write_bytes(b"x")
    process["model"]["low_vram"] = False
    process["model"]["model_kwargs"] = {}
    process["vae_anchor"]["vae_model_path"] = str(ckpt)
    result = collect_admission_diagnostics({"process": [process]})
    clean_ids = _ids(result)
    assert {RULE_VAE_ANCHOR, RULE_QWEN_LOW_VRAM_AUX, RULE_QWEN_RGBA_ANCHOR}.isdisjoint(clean_ids)
    assert any(item["rule_id"] == RULE_VAE_ANCHOR for item in result["deferred"])


def test_qwen_rgba_without_vae_anchor_is_not_blocked():
    # RGBA on its own (transparency training) stays valid; only the anchor
    # combination is rejected.
    process = {
        "model": {
            "name_or_path": "Comfy-Org/Qwen-Image-2.1",
            "arch": "qwen_image_2",
            "model_kwargs": {"rgba": True},
        },
        "train": {"optimizer": "adamw", "loss_type": "mse", "batch_size": 1},
        "datasets": [{"dataset_path": "images"}],
    }
    ids = _ids(collect_admission_diagnostics({"process": [process]}))
    assert RULE_QWEN_RGBA_ANCHOR not in ids


def test_temporary_safety_blocks_are_scoped_to_enabled_paths():
    process = _process(
        model={
            "name_or_path": "krea",
            "arch": "krea2:o_edit",
            "model_kwargs": {"edit": True},
        },
        network={"network_kwargs": {"module_dropout": 0.2}},
        train={
            "optimizer": "adamw",
            "loss_target": "source",
            "do_guidance_loss_cfg_zero": True,
        },
        sample={"prompts": ["preview"], "guidance_scale": 4},
    )
    ids = _ids(collect_admission_diagnostics({"process": [process]}))
    assert RULE_TEMPORARY_SAFETY in ids


def test_false_fast_controls_are_disabled():
    process = _process(
        model={"name_or_path": "fast", "arch": "minimax_h3_vsa"},
        datasets=[
            {
                "dataset_path": "video",
                "do_i2v": False,
                "control_path": False,
                "control_path_1": "",
                "clip_image_path": None,
                "inpaint_path": False,
            }
        ],
    )
    result = collect_admission_diagnostics({"process": [process]})
    assert RULE_H3_FAST_CONDITIONING not in _ids(result)


def test_dormant_audio_files_do_not_activate_standalone_voice_rules(tmp_path):
    audio_path = tmp_path / "unused.wav"
    audio_path.write_bytes(b"not decoded by static admission")
    process = _process(
        datasets=[
            {
                "dataset_path": str(tmp_path),
                "do_audio": False,
                "buckets": False,
                "caption_dropout_rate": 0.5,
            }
        ],
    )
    result = collect_admission_diagnostics({"process": [process]})
    assert result["valid"]
    assert result["resolved"][0]["dataset_audio_indices"] == []
    assert RULE_H3_AUDIO not in _ids(result)

def test_nested_standalone_audio_activates_voice_policy(tmp_path):
    nested = tmp_path / "clips"
    nested.mkdir()
    (nested / "voice.wav").write_bytes(b"audio marker")

    process = _process(
        datasets=[
            {
                "dataset_path": str(tmp_path),
                "do_audio": True,
                "buckets": False,
            }
        ]
    )
    result = collect_admission_diagnostics({"process": [process]})

    assert RULE_H3_AUDIO in _ids(result)
    assert result["resolved"][0]["dataset_audio_indices"] == [0]

def test_large_audio_dataset_defers_unresolved_presence(tmp_path):
    import toolkit.admission as admission

    for index in range(admission._AUDIO_SCAN_MAX_ENTRIES + 1):
        (tmp_path / f"caption-{index}.txt").write_text("caption")
    voices = tmp_path / "voices"
    voices.mkdir()
    (voices / "voice.wav").write_bytes(b"audio marker")
    process = _process(
        datasets=[{
            "dataset_path": str(tmp_path),
            "do_audio": True,
            "buckets": False,
        }]
    )
    result = collect_admission_diagnostics({"process": [process]})

    assert result["resolved"][0]["dataset_audio_indices"] == []
    assert RULE_H3_AUDIO not in _ids(result)
    assert admission.RULE_H3_AUDIO_DURATION in {
        diagnostic["rule_id"] for diagnostic in result["deferred"]
    }



def test_edit_cfg_checks_each_effective_preview_item_and_legacy_prompts():
    per_item = _process(
        model={
            "name_or_path": "krea",
            "arch": "krea2:o_edit",
            "model_kwargs": {"edit": True},
        },
        network={"type": "lora"},
        sample={
            "guidance_scale": 1,
            "samples": [
                {"prompt": "safe", "guidance_scale": 1},
                {"prompt": "unsafe", "guidance_scale": 2},
            ],
        },
    )
    result = collect_admission_diagnostics({"process": [per_item]})
    assert RULE_TEMPORARY_SAFETY in _ids(result)
    cfg = next(
        item for item in result["diagnostics"]
        if item["rule_id"] == RULE_TEMPORARY_SAFETY
        and "guidance_scale" in " ".join(item["fields"])
    )
    assert "config.process[0].sample.samples[1].guidance_scale" in cfg["fields"]

    legacy = _process(
        model={"name_or_path": "krea", "arch": "krea2:o_edit"},
        network={"type": "lora"},
        sample={"prompts": ["legacy prompt"]},
    )
    legacy_result = collect_admission_diagnostics({"process": [legacy]})
    assert RULE_TEMPORARY_SAFETY in _ids(legacy_result)


def test_fft_and_independent_noisy_transform_remain_eligible_but_inverse_collisions_do_not():
    standalone_fft = _process(
        model={"name_or_path": "krea", "arch": "krea2"},
        train={"optimizer": "adamw", "loss_target": "noise", "do_fft_loss": True},
    )
    fft_result = collect_admission_diagnostics({"process": [standalone_fft]})
    assert fft_result["valid"]
    assert RULE_TARGET_COLLISION not in _ids(fft_result)

    independent_noise = _process(
        model={"name_or_path": "krea", "arch": "krea2"},
        train={"optimizer": "adamw", "loss_target": "noise", "noisy_latent_multiplier": 1.2},
    )
    assert collect_admission_diagnostics({"process": [independent_noise]})["valid"]

    collision = _process(
        model={"name_or_path": "krea", "arch": "krea2"},
        train={
            "optimizer": "adamw",
            "loss_target": "noise",
            "noisy_latent_multiplier": 1.2,
            "do_fft_loss": True,
        },
    )
    assert RULE_TARGET_COLLISION in _ids(
        collect_admission_diagnostics({"process": [collision]})
    )


def test_exact_resume_admits_only_the_proven_deterministic_scope():
    process = _process(
        device="cpu",
        train={"optimizer": "adamw", "resume_mode": "exact"},
        datasets=[{"dataset_path": "not-present", "num_workers": 0, "buckets": False}],
    )
    assert collect_admission_diagnostics({"process": [process]})["valid"]
    process["device"] = "cuda:0"
    result = collect_admission_diagnostics({"process": [process]})
    assert not result["valid"]
    assert RULE_RESUME_DETERMINISM in _ids(result)
    assert any(
        "config.process[0].device" in diagnostic["fields"]
        for diagnostic in result["diagnostics"]
    )


def test_gpu_continuation_defers_rank_count_instead_of_reading_unserialized_keys():
    for mode in (None, "auto", "continue"):
        train = {"optimizer": "adamw", "gradient_accumulation": 2}
        if mode is not None:
            train["resume_mode"] = mode
        process = _process(
            device="cuda:0",
            train=train,
            datasets=[{"dataset_path": "not-present", "num_workers": 2, "buckets": True}],
        )
        result = collect_admission_diagnostics({"process": [process]})
        assert result["valid"], result["diagnostics"]
        assert RULE_RESUME_DETERMINISM not in _ids(result)
        deferred = [item for item in result["deferred"] if item["rule_id"] == RULE_RESUME_DETERMINISM]
        assert any("rank count" in item["reason"] for item in deferred)
        assert any("output folder" in item["remedy"] for item in deferred)


@pytest.mark.parametrize("mode", [None, "auto", "continue", "exact"])
def test_resumable_modes_reject_merged_network_saves(mode):
    train = {"optimizer": "adamw", "merge_network_on_save": True}
    datasets = [{"dataset_path": "not-present", "buckets": True}]
    device = "cuda:0"
    if mode == "exact":
        device = "cpu"
        datasets = [{"dataset_path": "not-present", "num_workers": 0, "buckets": False}]
    if mode is not None:
        train["resume_mode"] = mode
    result = collect_admission_diagnostics({"process": [_process(device=device, train=train, datasets=datasets)]})
    assert not result["valid"]
    assert RULE_RESUME_DETERMINISM in _ids(result)
    assert any("merge_network_on_save" in " ".join(item["fields"]) for item in result["diagnostics"])
    assert any("new, separate output folder" in item["remedy"] for item in result["diagnostics"])


def test_weights_only_merged_export_is_statically_admitted_and_destination_is_deferred():
    process = _process(train={
        "optimizer": "adamw",
        "resume_mode": "weights_only",
        "merge_network_on_save": True,
    })
    result = collect_admission_diagnostics({"process": [process]})
    assert result["valid"], result["diagnostics"]
    assert RULE_RESUME_DETERMINISM not in _ids(result)
    deferred = [item for item in result["deferred"] if item["rule_id"] == RULE_RESUME_DETERMINISM]
    assert any("new, separate output folder" in item["remedy"] for item in deferred)


def test_ignored_audio_extension_does_not_trigger_voice_restrictions(tmp_path):
    (tmp_path / "unused.opus").write_bytes(b"not enumerated by the loader")
    process = _process(datasets=[{
        "dataset_path": str(tmp_path), "do_audio": True,
        "buckets": False, "caption_dropout_rate": 0.1,
    }])
    assert collect_admission_diagnostics({"process": [process]})["valid"]
    (tmp_path / "voice.WAV").write_bytes(b"runtime decoding is deferred")
    result = collect_admission_diagnostics({"process": [process]})
    assert not result["valid"]
    assert RULE_H3_AUDIO in _ids(result)


@pytest.mark.parametrize("options", [
    {"dopsd_ref_mode": "other"},
    {"dopsd_ref_mode": "unknown"},
    {"dopsd_group_by": "class"},
    {"dopsd_identity_first": True},
])
def test_dormant_teacher_options_reject_known_runtime_failures(options):
    process = _process(model={
        "arch": "minimax_h3_ref2va", "name_or_path": "base",
        "model_kwargs": {"dopsd": False, **options},
    })
    result = collect_admission_diagnostics({"process": [process]})
    assert not result["valid"]
    assert RULE_DOPSD_VARIANT in _ids(result)
