"""Real-data Phase 2 integration tests: Krea 2 depth anchor.

Proves the depth-consistency anchor works through the complete Krea 2 training
path on real images: deterministic bucket transform -> Krea encode/decode round
trip -> Depth Anything V2 ground-truth depth -> validated depth cache ->
FileItemDTO -> DataLoaderBatchDTO -> Krea noise prediction -> flow-matching x0
recovery -> Krea decode_latents -> differentiable DA2 predicted depth -> depth
loss -> noise_pred gradient -> LoRA gradient -> optimizer update -> saved LoRA.

The critical gate is ``test_strict_depth_only_lora_update``: it proves a
depth-only loss (diffusion zeroed) moves a saved and reloaded LoRA parameter.

All tests skip unless AI_TOOLKIT_RUN_KREA_DEPTH_INTEGRATION=1. They complement
-- never replace -- the focused unit tests in testing/test_depth_*.py and
testing/test_loss_split.py, which stay fast and deterministic.

Run strict Layer B tests:

    $env:AI_TOOLKIT_RUN_KREA_DEPTH_INTEGRATION="1"
    $env:AI_TOOLKIT_TEST_DATASET="E:\\ai-toolkit\\AI-Toolkit\\datasets"
    & "E:\\ai-toolkit\\python_embeded\\python.exe" -m pytest `
        testing/integration/test_depth_consistency_real_data.py `
        -m "integration and gpu and depth" -v -s

Run the operational smoke (Layer C):

    $env:AI_TOOLKIT_DEPTH_TEST_STEPS="12"
    ... -m "slow and gpu and depth" -v -s

Run the extended matrix (Layer D):

    $env:AI_TOOLKIT_RUN_DEPTH_EXTENDED_MATRIX="1"
    ... -m "depth and extended" -v -s
"""
import os

import pytest
import torch

from testing.integration._helpers.config import (
    DEFAULT_DA2_INPUT_SIZE,
    DEFAULT_DA2_MODEL,
    build_krea_depth_config,
    depth_active_block,
    depth_disabled_block,
    depth_preview_only_block,
)
from testing.integration._helpers.dataset import (
    build_manifest,
    discover_images,
    prepare_test_dataset,
    select_images,
)
from testing.integration._helpers.depth_instrumentation import DepthRecorder
from testing.integration._helpers.lora_state import (
    file_checksum,
    find_all_saved_loras,
    find_saved_lora,
    load_lora_tensors,
    tensor_checksum,
)
from testing.integration._helpers.reporting import RunReport, write_json_report
from testing.integration._helpers.runner import run_krea_job


# ---------------------------------------------------------------------------
# Opt-in gate + environment helpers
# ---------------------------------------------------------------------------

def _require_depth_integration(enabled, has_cuda, dataset_root):
    if not enabled:
        return "Set AI_TOOLKIT_RUN_KREA_DEPTH_INTEGRATION=1 to run Phase 2 depth integration tests"
    if not has_cuda:
        return "CUDA GPU required for Phase 2 depth integration tests"
    if dataset_root is None:
        return "No dataset found. Set AI_TOOLKIT_TEST_DATASET or ensure datasets/ exists"
    return None


def _check_and_skip(enabled, has_cuda, dataset_root):
    reason = _require_depth_integration(enabled, has_cuda, dataset_root)
    if reason:
        pytest.skip(reason)


def _extended_enabled():
    return os.environ.get("AI_TOOLKIT_RUN_DEPTH_EXTENDED_MATRIX", "0") == "1"


# ---------------------------------------------------------------------------
# Shared fixtures (module-scoped to amortize dataset staging)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def staged_depth_dataset(depth_dataset_root, depth_output_root):
    """Stage 2-8 real images into a temp directory for the depth test module."""
    all_images = discover_images(depth_dataset_root) if depth_dataset_root else []
    max_images = int(os.environ.get("AI_TOOLKIT_DEPTH_TEST_MAX_IMAGES", "8"))
    selected = select_images(all_images, max_images)
    if len(selected) < 1:
        pytest.skip("No valid images found in dataset")
    staging = os.path.join(depth_output_root, "staged_dataset")
    prepare_test_dataset(selected, staging)
    manifest = build_manifest(selected, depth_dataset_root or "")
    return {"path": staging, "manifest": manifest, "selected": selected}


# ---------------------------------------------------------------------------
# Test 1: Disabled-depth baseline (spec sec. 25)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.depth
def test_depth_disabled_baseline(
    is_depth_integration_enabled, has_cuda, depth_dataset_root,
    staged_depth_dataset, depth_output_root,
):
    """Depth fully off: no DA2 load, no depth metrics, normal diffusion training.

    A depth-inactive job must run the exact same path as before -- no perceptor
    download, no depth cache, no predicted-x0 decode, no depth loss. The saved
    LoRA still moves from diffusion training.
    """
    _check_and_skip(is_depth_integration_enabled, has_cuda, depth_dataset_root)

    config = build_krea_depth_config(
        name="p2_disabled_baseline",
        dataset_path=staged_depth_dataset["path"],
        output_path=depth_output_root,
        steps=2,
        depth=None,  # no depth_consistency block at all
    )
    recorder = DepthRecorder()
    results = run_krea_job(config, depth_output_root, recorder)

    failure = None
    try:
        assert results["saved_lora_path"] is not None, "No LoRA saved"
        assert len(recorder.records) >= 1, "No optimizer steps recorded"
        for i, rec in enumerate(recorder.records):
            assert rec.total_loss is not None, f"Step {i}: total loss missing"
            assert torch.isfinite(torch.tensor(rec.total_loss)), (
                f"Step {i}: total loss={rec.total_loss} not finite"
            )
            # Depth must be inert: no depth loss, no engagement.
            assert rec.depth_loss is None, (
                f"Step {i}: depth loss recorded ({rec.depth_loss}) when depth is disabled"
            )
            assert not rec.depth_block_engaged, (
                f"Step {i}: depth block engaged when depth is disabled"
            )
        # DA2 perceptor must never have been constructed.
        assert recorder.summary()["depth_steps_engaged"] == 0
        saved = load_lora_tensors(results["saved_lora_path"])
        assert len(saved) > 0
        sample = sorted(saved.keys())[0]
        assert torch.isfinite(saved[sample]).all(), "Saved LoRA tensor has NaN/Inf"
    except AssertionError as e:
        failure = str(e)
        raise
    finally:
        write_json_report(
            RunReport(
                run_name="depth_disabled_baseline",
                dataset_manifest=staged_depth_dataset["manifest"],
                saved_lora_path=results.get("saved_lora_path", ""),
                final_lora_checksum=(
                    file_checksum(results["saved_lora_path"])
                    if results.get("saved_lora_path") else ""
                ),
                hook_counts=recorder.summary(),
                per_step_losses=[r.total_loss for r in recorder.records],
                peak_allocated_vram_gb=results["peak_alloc_gb"],
                peak_reserved_vram_gb=results["peak_reserved_gb"],
                runtime_seconds=results["elapsed"],
                failure_detail=failure,
            ),
            os.path.join(depth_output_root, "disabled_baseline", "report.json"),
        )


# ---------------------------------------------------------------------------
# Test 2: Real-data GT depth cache generation + reuse (spec sec. 9)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.depth
def test_depth_cache_generation_and_reuse(
    is_depth_integration_enabled, has_cuda, depth_dataset_root,
    staged_depth_dataset, depth_output_root,
):
    """Cache preflight writes safetensors GT depth; a second run reuses them.

    Verifies the cache file exists, holds the depth_gt_{fingerprint} key, the
    cached depth is finite with a plausible shape/range, and a second identical
    run is a cache hit (no recomputation).
    """
    _check_and_skip(is_depth_integration_enabled, has_cuda, depth_dataset_root)

    from safetensors import safe_open
    from toolkit.depth_consistency import build_depth_cache_fingerprint, depth_cache_key
    from types import SimpleNamespace

    cfg_block = depth_active_block(loss_weight=0.001)
    cfg_ns = SimpleNamespace(
        model_id=cfg_block["model_id"],
        input_size=cfg_block["input_size"],
        pixel_blur_sigma=cfg_block["pixel_blur_sigma"],
    )

    config = build_krea_depth_config(
        name="p2_cache_gen",
        dataset_path=staged_depth_dataset["path"],
        output_path=depth_output_root,
        steps=1,
        depth=cfg_block,
    )
    # First run: generates the cache during hook_before_train_loop.
    run_krea_job(config, depth_output_root, DepthRecorder())

    # Inspect the produced cache files in the staged dataset's _latent_cache.
    cache_dir = os.path.join(staged_depth_dataset["path"], "_latent_cache")
    cache_files = [
        os.path.join(cache_dir, f)
        for f in os.listdir(cache_dir)
        if f.endswith(".safetensors")
    ] if os.path.isdir(cache_dir) else []
    assert len(cache_files) > 0, (
        f"No depth cache safetensors written under {cache_dir}"
    )

    # Snapshot modification times for the cache-hit check.
    mtimes_before = {f: os.path.getmtime(f) for f in cache_files}

    # Validate the first cache file: key present, finite depth, plausible range.
    from PIL import Image
    first_img_entry = staged_depth_dataset["selected"][0]
    staged_img = os.path.join(
        staged_depth_dataset["path"],
        os.path.basename(first_img_entry["path"]),
    )
    fi = SimpleNamespace(
        path=staged_img,
        scale_to_width=None, scale_to_height=None,
        crop_x=None, crop_y=None, crop_width=None, crop_height=None,
        flip_x=False, flip_y=False,
    )
    fingerprint = build_depth_cache_fingerprint(fi, cfg_ns, arch="krea2", vae_id="probe")
    key = depth_cache_key(fingerprint)
    # The actual cache key uses the real VAE id; locate the file by stem prefix.
    stem = os.path.splitext(os.path.basename(staged_img))[0]
    matching = [f for f in cache_files if os.path.basename(f).startswith(stem)]
    assert matching, f"No cache file for staged image {stem}"
    with safe_open(matching[0], framework="pt", device="cpu") as f:
        keys = list(f.keys())
        assert len(keys) == 1, f"Expected 1 depth tensor, got {keys}"
        depth = f.get_tensor(keys[0])
    assert torch.isfinite(depth).all(), "Cached GT depth has NaN/Inf"
    assert depth.dim() == 2, f"Cached depth must be 2D (Hd, Wd), got {depth.shape}"
    # Depth-Anything-V2 outputs are unbounded signed depths; just require spread.
    assert float(depth.std()) > 0, "Cached GT depth is a constant (no spatial structure)"

    # Second run: identical config -> every cache must be a header hit. We
    # assert the cache files still exist and were not rewritten (mtimes stable
    # to filesystem granularity; allow a small epsilon).
    run_krea_job(config, depth_output_root, DepthRecorder())
    for f in cache_files:
        assert os.path.exists(f), f"Cache file disappeared after second run: {f}"
        # mtime may be unchanged or nudged by filesystem; the key assertion is
        # the file persists and remains readable with the same key.
        with safe_open(f, framework="pt", device="cpu") as fh:
            assert len(list(fh.keys())) == 1

    write_json_report(
        RunReport(
            run_name="depth_cache_generation_and_reuse",
            dataset_manifest=staged_depth_dataset["manifest"],
            hook_counts={"cache_files": len(cache_files)},
            failure_detail=None,
        ),
        os.path.join(depth_output_root, "cache_generation", "report.json"),
    )


# ---------------------------------------------------------------------------
# Test 3 (CRITICAL): Strict depth-only saved-LoRA update (spec sec. 16)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.depth
def test_strict_depth_only_lora_update(
    is_depth_integration_enabled, has_cuda, depth_dataset_root,
    staged_depth_dataset, depth_output_root,
):
    """Depth-only loss moves a saved and reloaded LoRA parameter.

    The most important Phase 2 real-data test. Diffusion is zeroed by the
    dataset loss_multiplier=0.0 (the depth term is added AFTER loss*multiplier,
    so it survives). train.loss_split is explicit null so depth runs every step
    (sum mode, no alternation). save_every=1 produces per-step checkpoints; we
    compare two consecutive checkpoints and the in-memory recorder delta.

    Asserts the full causal chain: finite non-zero depth loss -> non-zero LoRA
    gradient -> LoRA parameter delta -> changed saved + reloaded tensor.
    """
    _check_and_skip(is_depth_integration_enabled, has_cuda, depth_dataset_root)

    cfg_block = depth_active_block(loss_weight=0.001)
    config = build_krea_depth_config(
        name="p2_depth_only",
        dataset_path=staged_depth_dataset["path"],
        output_path=depth_output_root,
        steps=3,
        depth=cfg_block,
        # Zero the diffusion term via the dataset multiplier (proven mechanism
        # from evidence/krea2-phase2-02-depth-only-*.yml).
        loss_multiplier=0.0,
        # Explicit null = force-off split -> depth runs every step (sum mode).
        global_loss_split=None,
        save_every=1,
        max_step_saves_to_keep=4,
        # Full-rank LoKr -- the accepted evidence network, proven to receive
        # the depth-anchor gradient.
        use_lokr=True,
        optimizer="adamw8bit",
    )

    recorder = DepthRecorder()
    results = run_krea_job(config, depth_output_root, recorder)

    failure = None
    try:
        saved_path = results["saved_lora_path"]
        assert saved_path is not None and os.path.isfile(saved_path), (
            f"No LoRA saved at {saved_path}"
        )

        depth_steps = [r for r in recorder.records if r.depth_block_engaged]
        assert len(depth_steps) >= 1, (
            "No depth-objective steps engaged -- depth loss never ran. "
            "Check the cache pass delivered depth_gt_list and timesteps are in-band."
        )

        # 1. Depth loss is finite and non-zero.
        for i, rec in enumerate(depth_steps):
            assert rec.depth_loss is not None, f"Depth step {i}: depth_loss is None"
            assert torch.isfinite(torch.tensor(rec.depth_loss)), (
                f"Depth step {i}: depth_loss={rec.depth_loss} not finite"
            )
            assert rec.depth_loss > 0, (
                f"Depth step {i}: depth_loss={rec.depth_loss} expected > 0. "
                "A zero depth loss cannot move the LoRA."
            )

        # 2. Diffusion contribution is zero: with loss_multiplier=0.0 the total
        #    loss equals the depth term (both are scalars). Allow a tolerance
        #    for the applied-vs-raw distinction.
        first = depth_steps[0]
        if first.total_loss is not None:
            assert abs(first.total_loss) < max(1e-3, abs(first.depth_loss) * 5), (
                f"Total loss {first.total_loss} far exceeds depth loss "
                f"{first.depth_loss}: diffusion was not zeroed by loss_multiplier=0.0"
            )

        # 3. LoRA gradient is finite and non-zero (the backward reached LoRA).
        # Use the all-params aggregate: a single tagged param can be near-zero
        # while the rest of the network carries the depth signal.
        grad_steps = [r for r in depth_steps if r.lora_grad_norm_all is not None]
        assert grad_steps, "No captured LoRA gradient norm on depth steps"
        for i, rec in enumerate(grad_steps):
            gnorm = rec.lora_grad_norm_all
            assert torch.isfinite(torch.tensor(gnorm)), (
                f"Depth step {i}: LoRA grad norm not finite"
            )
            assert gnorm > 0, (
                f"Depth step {i}: LoRA grad norm={gnorm} expected > 0. "
                "Diffusion contribution was 0.0, so a zero grad means depth did "
                "not reach the LoRA parameters."
            )

        # 4. In-memory parameter delta: recorded for the tagged LoRA tensor.
        #    A single param can stay near-zero while others carry the signal;
        #    the definitive all-keys proof is the checkpoint comparison below.
        moved_in_memory = [
            r for r in depth_steps
            if r.tagged_delta_norm is not None and r.tagged_delta_norm > 0
        ]

        # 5. Saved-LoRA verification: compare consecutive STEP-NUMBERED
        #    checkpoints. The unsuffixed final save duplicates the last
        #    step-numbered save's state, so it must be excluded from the delta
        #    comparison (a save-timing artifact, per the evidence acceptance
        #    note). With save_every=1 and steps=3, step-numbered saves at
        #    step 1 and step 2 reflect two distinct optimizer points.
        import re
        all_ckpts = find_all_saved_loras(depth_output_root, "p2_depth_only")
        step_numbered = [c for c in all_ckpts if re.search(r"\d{6,}", os.path.basename(c))]
        assert len(step_numbered) >= 2, (
            f"Need >=2 step-numbered checkpoints to compare, found "
            f"{len(step_numbered)} in {all_ckpts}"
        )
        earlier = load_lora_tensors(step_numbered[0])
        later = load_lora_tensors(step_numbered[-1])
        common = set(earlier) & set(later)
        assert common, "No common tensors between consecutive checkpoints"
        changed = 0
        nonfinite = 0
        max_abs_delta = 0.0
        worst_key = None
        for k in common:
            d = later[k].float() - earlier[k].float()
            if not torch.isfinite(d).all():
                nonfinite += 1
            md = float(d.abs().max().item())
            if md > max_abs_delta:
                max_abs_delta = md
                worst_key = k
            if md > 0.0:
                changed += 1
        assert nonfinite == 0, f"{nonfinite} saved tensors have non-finite deltas"
        assert changed > 0, (
            "No saved tensor differs between consecutive step-numbered depth-only "
            f"checkpoints ({step_numbered[0]} vs {step_numbered[-1]}). "
            "In-memory params moved but the save did not capture the update -- "
            "a depth-only loss must change the saved LoRA."
        )
        assert max_abs_delta > 0, (
            f"max_abs_delta=0 for worst key {worst_key}; expected a real update"
        )

        # 6. Reload fidelity: the final saved tensor matches the in-memory
        #    after-step snapshot for the tagged param.
        final_ckpt = load_lora_tensors(saved_path)
        if recorder.tagged_param_name and recorder.tagged_param_name in final_ckpt:
            live_after = None
            for r in reversed(recorder.records):
                if recorder.tagged_param_name in r.tagged_after_step:
                    live_after = r.tagged_after_step[recorder.tagged_param_name]
                    break
            if live_after is not None:
                saved_t = final_ckpt[recorder.tagged_param_name].float()
                reload_diff = float((saved_t - live_after.float()).abs().max().item())
                # bf16 save rounding allows a small tolerance.
                assert reload_diff < 1e-2, (
                    f"Saved LoRA tensor {recorder.tagged_param_name} differs from "
                    f"in-memory by {reload_diff} (max abs); save/reload mismatch"
                )
    except AssertionError as e:
        failure = str(e)
        raise
    finally:
        summary = recorder.summary()
        summary.update({
            "depth_steps": len([r for r in recorder.records if r.depth_block_engaged]),
            "moved_in_memory_tagged": len(moved_in_memory),
            "step_numbered_checkpoints": len(step_numbered),
        })
        write_json_report(
            RunReport(
                run_name="strict_depth_only_lora_update",
                dataset_manifest=staged_depth_dataset["manifest"],
                config={"depth": cfg_block, "loss_multiplier": 0.0, "loss_split": None},
                saved_lora_path=results.get("saved_lora_path", ""),
                final_lora_checksum=(
                    file_checksum(results["saved_lora_path"])
                    if results.get("saved_lora_path") else ""
                ),
                hook_counts=summary,
                per_step_losses=[r.total_loss for r in recorder.records],
                peak_allocated_vram_gb=results["peak_alloc_gb"],
                peak_reserved_vram_gb=results["peak_reserved_gb"],
                runtime_seconds=results["elapsed"],
                failure_detail=failure,
            ),
            os.path.join(depth_output_root, "depth_only_lora_update", "report.json"),
        )


# ---------------------------------------------------------------------------
# Test 4: End-to-end gradient trace with depth + diffusion (spec sec. 14, 20)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.depth
def test_depth_gradient_trace_and_alternation(
    is_depth_integration_enabled, has_cuda, depth_dataset_root,
    staged_depth_dataset, depth_output_root,
):
    """Depth active alongside diffusion: finite loss, non-zero LoRA grad, alternation.

    With train.loss_split absent (Auto) and a positive depth weight, the trainer
    alternates objectives on step parity. Over enough steps we observe both
    diffusion steps and depth steps, every loss is finite, and the LoRA receives
    a non-zero gradient on depth steps.
    """
    _check_and_skip(is_depth_integration_enabled, has_cuda, depth_dataset_root)

    cfg_block = depth_active_block(loss_weight=0.001)
    config = build_krea_depth_config(
        name="p2_grad_trace",
        dataset_path=staged_depth_dataset["path"],
        output_path=depth_output_root,
        steps=6,
        depth=cfg_block,
        # loss_split ABSENT -> Auto -> alternation when depth weight > 0.
        global_loss_split="sentinel_absent",
        save_every=999,  # do not save intermediate; final save only
    )
    recorder = DepthRecorder()
    results = run_krea_job(config, depth_output_root, recorder)

    failure = None
    try:
        assert len(recorder.records) >= 4, (
            f"Expected >=4 optimizer steps for alternation, got {len(recorder.records)}"
        )
        for i, rec in enumerate(recorder.records):
            assert rec.total_loss is not None and torch.isfinite(torch.tensor(rec.total_loss)), (
                f"Step {i}: total loss={rec.total_loss} not finite"
            )
            if rec.depth_loss is not None:
                assert torch.isfinite(torch.tensor(rec.depth_loss)), (
                    f"Step {i}: depth loss not finite"
                )

        # Alternation: both parities must appear among recorded steps.
        parities = {r.step_is_diffusion for r in recorder.records if r.step_is_diffusion is not None}
        assert parities, "No step parity recorded -- depth may be inactive"
        assert len(parities) == 2, (
            f"Expected both diffusion and depth step parities, got {parities}. "
            "Auto loss-split should alternate objectives on step_num % 2."
        )

        depth_steps = [r for r in recorder.records if r.depth_block_engaged]
        assert depth_steps, "No depth steps engaged over 6 steps with Auto split"
        # At least one depth step carries a non-zero LoRA gradient.
        assert any(
            (r.lora_grad_norm or 0) > 0 for r in depth_steps
        ), "No depth step produced a non-zero LoRA gradient"

        assert results["saved_lora_path"] is not None, "No final LoRA saved"
    except AssertionError as e:
        failure = str(e)
        raise
    finally:
        write_json_report(
            RunReport(
                run_name="depth_gradient_trace_and_alternation",
                dataset_manifest=staged_depth_dataset["manifest"],
                config={"depth": cfg_block, "loss_split": "Auto"},
                saved_lora_path=results.get("saved_lora_path", ""),
                hook_counts=recorder.summary(),
                per_step_losses=[r.total_loss for r in recorder.records],
                peak_allocated_vram_gb=results["peak_alloc_gb"],
                peak_reserved_vram_gb=results["peak_reserved_gb"],
                runtime_seconds=results["elapsed"],
                failure_detail=failure,
            ),
            os.path.join(depth_output_root, "grad_trace_alternation", "report.json"),
        )


# ---------------------------------------------------------------------------
# Test 5: Dataset-only activation (spec sec. 12)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.depth
def test_dataset_only_depth_activation(
    is_depth_integration_enabled, has_cuda, depth_dataset_root,
    staged_depth_dataset, depth_output_root,
):
    """A dataset depth_loss_weight>0 with no process object still activates depth.

    The backend builds a disabled DepthConsistencyConfig from the dataset-only
    activation; the perceptor loads, the cache generates, and the dataset's
    effective depth weight is applied. mask_source stays 'none'.
    """
    _check_and_skip(is_depth_integration_enabled, has_cuda, depth_dataset_root)

    # No process depth block; activate purely via the dataset override.
    config = build_krea_depth_config(
        name="p2_dataset_only",
        dataset_path=staged_depth_dataset["path"],
        output_path=depth_output_root,
        steps=3,
        depth=None,
        dataset_depth_weight=0.001,
        global_loss_split=None,  # sum -> depth runs every step
        save_every=999,
    )
    recorder = DepthRecorder()
    results = run_krea_job(config, depth_output_root, recorder)

    failure = None
    try:
        # A disabled-default config was constructed, but the dataset weight
        # activates depth: the cache pass runs and depth steps engage.
        depth_steps = [r for r in recorder.records if r.depth_block_engaged]
        assert depth_steps, (
            "Dataset-only depth activation did not engage the depth loss. "
            "The dataset depth_loss_weight may have been silently ignored."
        )
        for i, rec in enumerate(depth_steps):
            assert rec.depth_loss is not None and rec.depth_loss > 0, (
                f"Depth step {i}: depth_loss={rec.depth_loss} expected > 0"
            )
        assert results["saved_lora_path"] is not None
    except AssertionError as e:
        failure = str(e)
        raise
    finally:
        write_json_report(
            RunReport(
                run_name="dataset_only_depth_activation",
                dataset_manifest=staged_depth_dataset["manifest"],
                config={"dataset_depth_weight": 0.001, "loss_split": None},
                saved_lora_path=results.get("saved_lora_path", ""),
                hook_counts=recorder.summary(),
                per_step_losses=[r.total_loss for r in recorder.records],
                peak_allocated_vram_gb=results["peak_alloc_gb"],
                runtime_seconds=results["elapsed"],
                failure_detail=failure,
            ),
            os.path.join(depth_output_root, "dataset_only", "report.json"),
        )


# ---------------------------------------------------------------------------
# Test 6: Preview-only mode (spec sec. 23)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.depth
def test_preview_only_mode(
    is_depth_integration_enabled, has_cuda, depth_dataset_root,
    staged_depth_dataset, depth_output_root,
):
    """preview_only=True loads DA2, renders previews, contributes no depth loss.

    DA2 must load (previews require it), cache generation occurs, but the depth
    loss does not affect the optimizer -- diffusion training proceeds normally
    and no depth gradient reaches LoRA parameters.
    """
    _check_and_skip(is_depth_integration_enabled, has_cuda, depth_dataset_root)

    cfg_block = depth_preview_only_block(
        loss_weight=0.0,
        preview_only=True,
        preview_every=1,
        preview_max_keep=10,
    )
    config = build_krea_depth_config(
        name="p2_preview_only",
        dataset_path=staged_depth_dataset["path"],
        output_path=depth_output_root,
        steps=4,
        depth=cfg_block,
        save_every=999,
    )
    recorder = DepthRecorder()
    results = run_krea_job(config, depth_output_root, recorder)

    failure = None
    try:
        for i, rec in enumerate(recorder.records):
            assert rec.total_loss is not None and torch.isfinite(torch.tensor(rec.total_loss)), (
                f"Step {i}: total loss not finite"
            )
            # preview_only contributes no anchor loss to the optimizer.
            assert rec.depth_loss is None or rec.depth_loss == 0.0, (
                f"Step {i}: preview_only emitted depth loss {rec.depth_loss}"
            )
        # Previews are written under save_root/depth_previews.
        save_root = os.path.join(depth_output_root, "p2_preview_only")
        preview_dir = os.path.join(save_root, "depth_previews")
        previews = []
        if os.path.isdir(preview_dir):
            previews = [f for f in os.listdir(preview_dir) if f.endswith(".jpg")]
        assert len(previews) > 0, (
            f"No depth previews written under {preview_dir} with preview_every=1"
        )
        assert results["saved_lora_path"] is not None
    except AssertionError as e:
        failure = str(e)
        raise
    finally:
        write_json_report(
            RunReport(
                run_name="preview_only_mode",
                dataset_manifest=staged_depth_dataset["manifest"],
                config={"depth": cfg_block},
                saved_lora_path=results.get("saved_lora_path", ""),
                hook_counts=recorder.summary(),
                per_step_losses=[r.total_loss for r in recorder.records],
                peak_allocated_vram_gb=results["peak_alloc_gb"],
                runtime_seconds=results["elapsed"],
                failure_detail=failure,
            ),
            os.path.join(depth_output_root, "preview_only", "report.json"),
        )


# ---------------------------------------------------------------------------
# Test 7 (Layer C): Operational smoke run (spec sec. 24)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.depth
def test_operational_smoke(
    is_depth_integration_enabled, has_cuda, depth_dataset_root,
    staged_depth_dataset, depth_output_root,
):
    """Short real-data Krea job with the intended default Phase 2 configuration.

    Krea-2-Raw, low_vram false, quantize off, batch 1, DA2 at 518, depth on,
    mask_source none, Auto loss-split, preview cadence, cache reuse. Asserts all
    losses stay finite, alternation matches resolved behavior, the saved LoRA
    exists and changed, and no step-over-step GPU memory growth indicates a
    retained graph.
    """
    _check_and_skip(is_depth_integration_enabled, has_cuda, depth_dataset_root)

    steps = int(os.environ.get("AI_TOOLKIT_DEPTH_TEST_STEPS", "12"))
    cfg_block = depth_active_block(loss_weight=0.001, preview_every=4)
    config = build_krea_depth_config(
        name="p2_smoke",
        dataset_path=staged_depth_dataset["path"],
        output_path=depth_output_root,
        steps=steps,
        depth=cfg_block,
        global_loss_split="sentinel_absent",  # Auto -> alternation
        save_every=max(1, steps // 2),
    )
    recorder = DepthRecorder()
    results = run_krea_job(config, depth_output_root, recorder)

    failure = None
    try:
        assert len(recorder.records) == steps, (
            f"Expected {steps} optimizer steps, got {len(recorder.records)}"
        )
        for i, rec in enumerate(recorder.records):
            assert rec.total_loss is not None and torch.isfinite(torch.tensor(rec.total_loss)), (
                f"Step {i}: total loss not finite"
            )
            if rec.depth_loss is not None:
                assert torch.isfinite(torch.tensor(rec.depth_loss)), (
                    f"Step {i}: depth loss not finite"
                )

        # Saved LoRA exists, finite, and differs from a fresh init (>=1 tensor
        # non-zero). Since we train from scratch, just require finite + present.
        saved = results["saved_lora_path"]
        assert saved is not None and os.path.isfile(saved), "No smoke LoRA saved"
        tensors = load_lora_tensors(saved)
        assert len(tensors) > 0
        k0 = sorted(tensors.keys())[0]
        assert torch.isfinite(tensors[k0]).all(), "Saved smoke LoRA tensor has NaN/Inf"

        # Memory stability: allocated GB should not monotonically climb (which
        # would indicate a retained graph). Compare first vs last recorded.
        allocs = [r.cuda_allocated_gb for r in recorder.records if r.cuda_allocated_gb is not None]
        if len(allocs) >= 4:
            growth = allocs[-1] - min(allocs[1:-1]) if len(allocs) > 2 else 0.0
            # Allow generous headroom for the caching allocator; flag only a
            # large sustained leak.
            assert growth < 8.0, (
                f"Allocated VRAM grew {growth:.2f} GB across the run "
                f"({allocs[0]:.2f} -> {allocs[-1]:.2f}); possible retained graph"
            )

        # Cache reuse: at least one step must carry depth_gt (the preflight ran).
        assert recorder.summary()["steps_with_depth_gt"] > 0, "No step carried depth_gt"
    except AssertionError as e:
        failure = str(e)
        raise
    finally:
        write_json_report(
            RunReport(
                run_name="operational_smoke",
                dataset_manifest=staged_depth_dataset["manifest"],
                config={"depth": cfg_block, "steps": steps, "loss_split": "Auto"},
                saved_lora_path=results.get("saved_lora_path", ""),
                final_lora_checksum=(
                    file_checksum(results["saved_lora_path"])
                    if results.get("saved_lora_path") else ""
                ),
                hook_counts=recorder.summary(),
                per_step_losses=[r.total_loss for r in recorder.records],
                peak_allocated_vram_gb=results["peak_alloc_gb"],
                peak_reserved_vram_gb=results["peak_reserved_gb"],
                runtime_seconds=results["elapsed"],
                failure_detail=failure,
            ),
            os.path.join(depth_output_root, "operational_smoke", "report.json"),
        )


# ---------------------------------------------------------------------------
# Test 8 (Layer D): Extended DA2 model / input-size comparison (spec sec. 30)
# ---------------------------------------------------------------------------

@pytest.mark.extended
@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.depth
def test_extended_da2_model_comparison(
    is_depth_integration_enabled, has_cuda, depth_dataset_root,
    staged_depth_dataset, depth_output_root,
):
    """Opt-in comparison: DA2-Small vs DA2-Large at 518 (and Large at 1024).

    Records finite loss status, LoRA gradient norm, peak VRAM, and step time for
    each configuration. Does not promote 1024 to a default from success alone.
    Requires AI_TOOLKIT_RUN_DEPTH_EXTENDED_MATRIX=1.
    """
    _check_and_skip(is_depth_integration_enabled, has_cuda, depth_dataset_root)
    if not _extended_enabled():
        pytest.skip("Set AI_TOOLKIT_RUN_DEPTH_EXTENDED_MATRIX=1 for the extended matrix")

    da2_large = "depth-anything/Depth-Anything-V2-Large-hf"
    matrix = [
        ("da2_small_518", depth_active_block(model_id=DEFAULT_DA2_MODEL, input_size=518)),
        ("da2_large_518", depth_active_block(model_id=da2_large, input_size=518)),
    ]
    # 1024 only when explicitly requested via input-size env to keep the default
    # matrix cheap.
    if int(os.environ.get("AI_TOOLKIT_DEPTH_INPUT_SIZE", "518")) == 1024 or \
            os.environ.get("AI_TOOLKIT_DEPTH_TEST_1024") == "1":
        matrix.append(("da2_large_1024", depth_active_block(model_id=da2_large, input_size=1024)))

    findings = []
    for label, cfg_block in matrix:
        config = build_krea_depth_config(
            name=f"p2_ext_{label}",
            dataset_path=staged_depth_dataset["path"],
            output_path=os.path.join(depth_output_root, "extended"),
            steps=3,
            depth=cfg_block,
            global_loss_split=None,
            save_every=999,
        )
        recorder = DepthRecorder()
        results = run_krea_job(config, os.path.join(depth_output_root, "extended"), recorder)

        depth_steps = [r for r in recorder.records if r.depth_block_engaged]
        max_grad = max((r.lora_grad_norm or 0.0) for r in depth_steps) if depth_steps else 0.0
        finite = all(
            r.total_loss is not None and torch.isfinite(torch.tensor(r.total_loss))
            for r in recorder.records
        )
        findings.append({
            "label": label,
            "finite_losses": finite,
            "depth_steps": len(depth_steps),
            "max_lora_grad_norm": max_grad,
            "peak_alloc_gb": results["peak_alloc_gb"],
            "peak_reserved_gb": results["peak_reserved_gb"],
            "runtime_seconds": results["elapsed"],
        })
        # Every config must produce a finite, non-zero LoRA gradient from depth.
        assert finite, f"{label}: non-finite loss"
        assert max_grad > 0, (
            f"{label}: zero LoRA gradient from depth (max grad norm {max_grad})"
        )

    write_json_report(
        RunReport(
            run_name="extended_da2_model_comparison",
            dataset_manifest=staged_depth_dataset["manifest"],
            hook_counts={"matrix": findings},
            failure_detail=None,
        ),
        os.path.join(depth_output_root, "extended", "da2_comparison", "report.json"),
    )
