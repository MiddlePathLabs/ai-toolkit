"""Real-data Phase 1 integration tests: weight noising and gradient noising
inside actual short Krea 2 training runs.

All tests skip unless AI_TOOLKIT_RUN_KREA_INTEGRATION=1.

Run strict Layer A tests:
    $env:AI_TOOLKIT_RUN_KREA_INTEGRATION="1"
    $env:AI_TOOLKIT_TEST_DATASET="<repo>\\datasets"
    pytest testing/integration/test_perceptual_noising_real_data.py -m "integration and gpu" -v -s

Run operational smoke (Layer B):
    $env:AI_TOOLKIT_NOISING_TEST_STEPS="50"
    pytest testing/integration/test_perceptual_noising_real_data.py::test_matched_smoke_comparison -m "slow and gpu" -v -s
"""
import os

import pytest
import torch

from testing.integration._helpers.config import build_krea_config
from testing.integration._helpers.dataset import (
    build_manifest,
    discover_images,
    prepare_test_dataset,
    select_images,
)
from testing.integration._helpers.instrumentation import NoisingRecorder
from testing.integration._helpers.lora_state import file_checksum
from testing.integration._helpers.reporting import RunReport, write_json_report
from testing.integration._helpers.runner import create_seed_lora, run_krea_job


def _require_integration(is_integration_enabled, has_cuda, dataset_root):
    """Skip helper: returns a reason string or None if all checks pass."""
    if not is_integration_enabled:
        return "Set AI_TOOLKIT_RUN_KREA_INTEGRATION=1 to run Krea 2 integration tests"
    if not has_cuda:
        return "CUDA GPU required for Krea 2 integration tests"
    if dataset_root is None:
        return "No dataset found. Set AI_TOOLKIT_TEST_DATASET or ensure datasets/ exists"
    return None


def _check_and_skip(is_integration_enabled, has_cuda, dataset_root, initial_lora):
    """Module-level guard for every integration test."""
    reason = _require_integration(is_integration_enabled, has_cuda, dataset_root)
    if reason:
        pytest.skip(reason)
    if initial_lora is None or not os.path.isfile(initial_lora.get("path", "")):
        pytest.skip("Initial LoRA seed not available")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def staged_dataset(dataset_root, output_root):
    """Stage 2-4 images into a temp directory for the test module."""
    all_images = discover_images(dataset_root) if dataset_root else []
    selected = select_images(all_images, 4)
    if len(selected) < 1:
        pytest.skip("No valid images found in dataset")
    staging = os.path.join(output_root, "staged_dataset")
    prepare_test_dataset(selected, staging)
    manifest = build_manifest(selected, dataset_root or "")
    return {"path": staging, "manifest": manifest, "selected": selected}


@pytest.fixture(scope="module")
def initial_lora(staged_dataset, output_root, is_integration_enabled, has_cuda, dataset_root):
    """Run a 1-step seed job to create an initial LoRA file for all tests.

    Caches the seed LoRA path on disk so subsequent subprocess test runs
    can reuse it without recreating.
    """
    reason = _require_integration(is_integration_enabled, has_cuda, dataset_root)
    if reason:
        pytest.skip(reason)

    seed_marker = os.path.join(output_root, "seed_lora_path.txt")
    # Check for an existing cached seed LoRA from a prior run
    if os.path.isfile(seed_marker):
        cached_path = open(seed_marker, "r").read().strip()
        if os.path.isfile(cached_path):
            return {"path": cached_path, "checksum": file_checksum(cached_path)}

    base_config = build_krea_config(
        name="phase1_seed",
        dataset_path=staged_dataset["path"],
        output_path=output_root,
        steps=1,
    )
    seed_path = create_seed_lora(base_config, output_root)
    checksum = file_checksum(seed_path)
    # Cache the path for subprocess reuse
    os.makedirs(os.path.dirname(seed_marker), exist_ok=True)
    with open(seed_marker, "w") as f:
        f.write(seed_path)
    return {"path": seed_path, "checksum": checksum}


# ---------------------------------------------------------------------------
# Test 1: Disabled Baseline
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.gpu
def test_disabled_baseline(
    is_integration_enabled, has_cuda, dataset_root, staged_dataset, initial_lora, output_root
):
    """Both noising features disabled: no injector activity, finite loss, untagged unchanged."""
    _check_and_skip(is_integration_enabled, has_cuda, dataset_root, initial_lora)

    config = build_krea_config(
        name="phase1_disabled",
        dataset_path=staged_dataset["path"],
        output_path=output_root,
        steps=2,
        weight_noise={"enabled": False},
        gradient_noise={"enabled": False},
        pretrained_lora_path=initial_lora["path"],
    )

    recorder = NoisingRecorder()
    results = run_krea_job(config, output_root, recorder)
    summary = recorder.summary()

    failure = None
    try:
        assert results["saved_lora_path"] is not None, "No LoRA was saved"
        assert len(recorder.records) >= 1, f"Expected >= 1 optimizer step, got {len(recorder.records)}"
        for i, rec in enumerate(recorder.records):
            assert rec.loss is not None, f"Step {i}: loss is None"
            assert torch.isfinite(torch.tensor(rec.loss)), f"Step {i}: loss={rec.loss} is not finite"
            # When disabled, the methods are still CALLED by hook_train_loop but return early.
            # Assert no EFFECT: no metrics emitted, no parameter changes from noising.
            assert rec.grad_noise_norm is None, (
                f"Step {i}: grad noise emitted metric (norm={rec.grad_noise_norm}) when disabled"
            )
            assert rec.weight_noise_norm is None, (
                f"Step {i}: weight noise emitted metric (norm={rec.weight_noise_norm}) when disabled"
            )
            # Tagged params should be unchanged by weight noise when disabled
            if rec.tagged_after_optimizer and rec.tagged_after_weight_noise:
                for pname in rec.tagged_after_optimizer:
                    assert torch.equal(
                        rec.tagged_after_optimizer[pname], rec.tagged_after_weight_noise[pname]
                    ), f"Step {i}: tagged param {pname} changed from weight noise when disabled"
        assert summary["all_tagged_count"] > 0, "No tagged LoRA parameters found"

        if recorder.untagged_param is not None and recorder.records:
            rec = recorder.records[-1]
            if rec.untagged_after_optimizer and rec.untagged_after_weight_noise:
                for pname in rec.untagged_after_optimizer:
                    before = rec.untagged_after_optimizer[pname]
                    after = rec.untagged_after_weight_noise[pname]
                    assert torch.equal(before, after), (
                        f"Untagged param {pname} changed when noising is disabled"
                    )
    except AssertionError as e:
        failure = str(e)
        raise
    finally:
        report = RunReport(
            run_name="disabled_baseline",
            dataset_manifest=staged_dataset["manifest"],
            seed=42,
            initial_lora_checksum=initial_lora["checksum"],
            final_lora_checksum=(
                file_checksum(results["saved_lora_path"]) if results.get("saved_lora_path") else ""
            ),
            saved_lora_path=results.get("saved_lora_path", ""),
            hook_counts=summary,
            per_step_losses=[r.loss for r in recorder.records],
            peak_allocated_vram_gb=results["peak_alloc_gb"],
            peak_reserved_vram_gb=results["peak_reserved_gb"],
            runtime_seconds=results["elapsed"],
            failure_detail=failure,
        )
        write_json_report(report, os.path.join(output_root, "disabled_baseline", "report.json"))


# ---------------------------------------------------------------------------
# Test 2: Relative Weight Noise
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.gpu
def test_relative_weight_noise(
    is_integration_enabled, has_cuda, dataset_root, staged_dataset, initial_lora, output_root
):
    """Relative weight noise: runs once/step, after EMA, tagged-only, non-zero delta where RMS > 0."""
    _check_and_skip(is_integration_enabled, has_cuda, dataset_root, initial_lora)

    config = build_krea_config(
        name="phase1_rel_wnoise",
        dataset_path=staged_dataset["path"],
        output_path=output_root,
        steps=2,
        weight_noise={
            "enabled": True,
            "mode": "relative",
            "sigma": 0.00125,
            "bound_norm": False,
            "log_every": 1,
        },
        gradient_noise={"enabled": False},
        pretrained_lora_path=initial_lora["path"],
    )

    recorder = NoisingRecorder()
    results = run_krea_job(config, output_root, recorder)
    summary = recorder.summary()

    failure = None
    try:
        assert len(recorder.records) >= 1
        for i, rec in enumerate(recorder.records):
            assert rec.loss is not None and torch.isfinite(torch.tensor(rec.loss)), (
                f"Step {i}: loss not finite"
            )
            assert rec.weight_noise_invocations == 1, (
                f"Step {i}: weight noise ran {rec.weight_noise_invocations} times, expected 1"
            )
            assert rec.grad_noise_norm is None, (
                "Gradient noise emitted metric when disabled"
            )

            if rec.weight_noise_norm is not None:
                assert torch.isfinite(torch.tensor(rec.weight_noise_norm)), (
                    f"Step {i}: weight_noise_norm not finite"
                )
                assert rec.weight_noise_norm > 0, (
                    f"Step {i}: weight_noise_norm={rec.weight_noise_norm} expected > 0"
                )

            if rec.weight_norm_pre_noise is not None:
                assert torch.isfinite(torch.tensor(rec.weight_norm_pre_noise))

            if rec.tagged_after_optimizer and rec.tagged_after_weight_noise:
                pname = recorder.tagged_param_name
                after_opt = rec.tagged_after_optimizer[pname]
                after_noise = rec.tagged_after_weight_noise[pname]
                delta = (after_noise - after_opt).abs().max().item()
                if after_opt.norm().item() > 1e-8:
                    assert delta > 0, (
                        f"Step {i}: tagged param {pname} weight-noise delta is zero but RMS > 0"
                    )

            if rec.untagged_after_optimizer and rec.untagged_after_weight_noise:
                pname = recorder.untagged_param_name
                assert torch.equal(
                    rec.untagged_after_optimizer[pname],
                    rec.untagged_after_weight_noise[pname],
                ), f"Step {i}: untagged param {pname} changed from weight noise"

        assert summary["weight_noise_calls"] == len(recorder.records), (
            f"Weight noise calls {summary['weight_noise_calls']} != "
            f"optimizer steps {len(recorder.records)}"
        )
    except AssertionError as e:
        failure = str(e)
        raise
    finally:
        report = RunReport(
            run_name="relative_weight_noise",
            dataset_manifest=staged_dataset["manifest"],
            initial_lora_checksum=initial_lora["checksum"],
            final_lora_checksum=(
                file_checksum(results["saved_lora_path"]) if results.get("saved_lora_path") else ""
            ),
            saved_lora_path=results.get("saved_lora_path", ""),
            hook_counts=summary,
            per_step_losses=[r.loss for r in recorder.records],
            peak_allocated_vram_gb=results["peak_alloc_gb"],
            peak_reserved_vram_gb=results["peak_reserved_gb"],
            runtime_seconds=results["elapsed"],
            failure_detail=failure,
        )
        write_json_report(report, os.path.join(output_root, "relative_weight_noise", "report.json"))


# ---------------------------------------------------------------------------
# Test 3: Absolute Weight Noise
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.gpu
def test_absolute_weight_noise(
    is_integration_enabled, has_cuda, dataset_root, staged_dataset, initial_lora, output_root
):
    """Absolute weight noise: tagged-only, independent of RMS, finite next-step loss."""
    _check_and_skip(is_integration_enabled, has_cuda, dataset_root, initial_lora)

    config = build_krea_config(
        name="phase1_abs_wnoise",
        dataset_path=staged_dataset["path"],
        output_path=output_root,
        steps=2,
        weight_noise={
            "enabled": True,
            "mode": "absolute",
            "sigma": 0.01,
            "bound_norm": False,
            "log_every": 1,
        },
        gradient_noise={"enabled": False},
        pretrained_lora_path=initial_lora["path"],
    )

    recorder = NoisingRecorder()
    results = run_krea_job(config, output_root, recorder)
    summary = recorder.summary()

    failure = None
    try:
        assert len(recorder.records) >= 1
        for i, rec in enumerate(recorder.records):
            assert rec.loss is not None and torch.isfinite(torch.tensor(rec.loss)), (
                f"Step {i}: loss not finite"
            )
            assert rec.weight_noise_invocations == 1

            if rec.tagged_after_optimizer and rec.tagged_after_weight_noise:
                pname = recorder.tagged_param_name
                after_opt = rec.tagged_after_optimizer[pname]
                after_noise = rec.tagged_after_weight_noise[pname]
                delta = (after_noise - after_opt).abs().max().item()
                assert delta > 0, (
                    f"Step {i}: absolute weight noise produced zero delta on tagged param {pname}"
                )

            if rec.untagged_after_optimizer and rec.untagged_after_weight_noise:
                pname = recorder.untagged_param_name
                assert torch.equal(
                    rec.untagged_after_optimizer[pname],
                    rec.untagged_after_weight_noise[pname],
                ), f"Step {i}: untagged param {pname} changed from absolute weight noise"
    except AssertionError as e:
        failure = str(e)
        raise
    finally:
        report = RunReport(
            run_name="absolute_weight_noise",
            dataset_manifest=staged_dataset["manifest"],
            initial_lora_checksum=initial_lora["checksum"],
            final_lora_checksum=(
                file_checksum(results["saved_lora_path"]) if results.get("saved_lora_path") else ""
            ),
            saved_lora_path=results.get("saved_lora_path", ""),
            hook_counts=summary,
            per_step_losses=[r.loss for r in recorder.records],
            peak_allocated_vram_gb=results["peak_alloc_gb"],
            runtime_seconds=results["elapsed"],
            failure_detail=failure,
        )
        write_json_report(report, os.path.join(output_root, "absolute_weight_noise", "report.json"))


# ---------------------------------------------------------------------------
# Test 4: Bound Norm Weight Noise
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.gpu
def test_bound_norm_weight_noise(
    is_integration_enabled, has_cuda, dataset_root, staged_dataset, initial_lora, output_root
):
    """Bound norm: norm(after bounded noise) ~= norm(clean post-optimizer parameter)."""
    _check_and_skip(is_integration_enabled, has_cuda, dataset_root, initial_lora)

    config = build_krea_config(
        name="phase1_bound_wnoise",
        dataset_path=staged_dataset["path"],
        output_path=output_root,
        steps=2,
        weight_noise={
            "enabled": True,
            "mode": "absolute",
            "sigma": 0.01,
            "bound_norm": True,
            "log_every": 1,
        },
        gradient_noise={"enabled": False},
        pretrained_lora_path=initial_lora["path"],
    )

    recorder = NoisingRecorder()
    results = run_krea_job(config, output_root, recorder)
    summary = recorder.summary()

    failure = None
    try:
        assert len(recorder.records) >= 1
        for i, rec in enumerate(recorder.records):
            assert rec.loss is not None and torch.isfinite(torch.tensor(rec.loss))

            if rec.tagged_after_optimizer and rec.tagged_after_weight_noise:
                pname = recorder.tagged_param_name
                after_opt = rec.tagged_after_optimizer[pname]
                after_noise = rec.tagged_after_weight_noise[pname]
                norm_before = after_opt.norm().item()
                norm_after = after_noise.norm().item()
                if norm_before > 1e-8:
                    rel_diff = abs(norm_after - norm_before) / norm_before
                    assert rel_diff < 0.01, (
                        f"Step {i}: bound_norm failed. norm_before={norm_before:.6f}, "
                        f"norm_after={norm_after:.6f}, rel_diff={rel_diff:.6f}"
                    )
                delta = (after_noise - after_opt).abs().max().item()
                assert delta > 0 or norm_before == 0, (
                    f"Step {i}: bound_norm produced no direction change on non-zero param"
                )
                assert torch.isfinite(after_noise).all(), f"Step {i}: NaN/Inf in bounded param"
    except AssertionError as e:
        failure = str(e)
        raise
    finally:
        report = RunReport(
            run_name="bound_norm_weight_noise",
            dataset_manifest=staged_dataset["manifest"],
            initial_lora_checksum=initial_lora["checksum"],
            final_lora_checksum=(
                file_checksum(results["saved_lora_path"]) if results.get("saved_lora_path") else ""
            ),
            saved_lora_path=results.get("saved_lora_path", ""),
            hook_counts=summary,
            per_step_losses=[r.loss for r in recorder.records],
            peak_allocated_vram_gb=results["peak_alloc_gb"],
            runtime_seconds=results["elapsed"],
            failure_detail=failure,
        )
        write_json_report(report, os.path.join(output_root, "bound_norm_weight_noise", "report.json"))


# ---------------------------------------------------------------------------
# Test 5: Absolute Gradient Noise
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.gpu
def test_absolute_gradient_noise(
    is_integration_enabled, has_cuda, dataset_root, staged_dataset, initial_lora, output_root
):
    """Absolute gradient noise: runs once/step, post-clip, tagged-only, finite delta."""
    _check_and_skip(is_integration_enabled, has_cuda, dataset_root, initial_lora)

    config = build_krea_config(
        name="phase1_abs_gnoise",
        dataset_path=staged_dataset["path"],
        output_path=output_root,
        steps=2,
        weight_noise={"enabled": False},
        gradient_noise={"enabled": True, "mode": "absolute", "sigma": 0.001, "log_every": 1},
        pretrained_lora_path=initial_lora["path"],
    )

    recorder = NoisingRecorder()
    results = run_krea_job(config, output_root, recorder)
    summary = recorder.summary()

    failure = None
    try:
        assert len(recorder.records) >= 1
        for i, rec in enumerate(recorder.records):
            assert rec.loss is not None and torch.isfinite(torch.tensor(rec.loss)), (
                f"Step {i}: loss not finite"
            )
            assert rec.grad_noise_invocations == 1, (
                f"Step {i}: grad noise ran {rec.grad_noise_invocations} times"
            )
            assert rec.weight_noise_norm is None, (
                f"Step {i}: weight noise emitted metric when disabled"
            )
            if rec.grad_noise_norm is not None:
                assert torch.isfinite(torch.tensor(rec.grad_noise_norm))
                assert rec.grad_noise_norm > 0, f"Step {i}: grad_noise_norm expected > 0"
        assert summary["grad_noise_calls"] == len(recorder.records)
    except AssertionError as e:
        failure = str(e)
        raise
    finally:
        report = RunReport(
            run_name="absolute_gradient_noise",
            dataset_manifest=staged_dataset["manifest"],
            initial_lora_checksum=initial_lora["checksum"],
            final_lora_checksum=(
                file_checksum(results["saved_lora_path"]) if results.get("saved_lora_path") else ""
            ),
            saved_lora_path=results.get("saved_lora_path", ""),
            hook_counts=summary,
            per_step_losses=[r.loss for r in recorder.records],
            peak_allocated_vram_gb=results["peak_alloc_gb"],
            runtime_seconds=results["elapsed"],
            failure_detail=failure,
        )
        write_json_report(report, os.path.join(output_root, "absolute_gradient_noise", "report.json"))


# ---------------------------------------------------------------------------
# Test 6: Relative Gradient Noise
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.gpu
def test_relative_gradient_noise(
    is_integration_enabled, has_cuda, dataset_root, staged_dataset, initial_lora, output_root
):
    """Relative gradient noise: scales by gradient RMS, zero-grad-safe."""
    _check_and_skip(is_integration_enabled, has_cuda, dataset_root, initial_lora)

    config = build_krea_config(
        name="phase1_rel_gnoise",
        dataset_path=staged_dataset["path"],
        output_path=output_root,
        steps=2,
        weight_noise={"enabled": False},
        gradient_noise={"enabled": True, "mode": "relative", "sigma": 0.001, "log_every": 1},
        pretrained_lora_path=initial_lora["path"],
    )

    recorder = NoisingRecorder()
    results = run_krea_job(config, output_root, recorder)
    summary = recorder.summary()

    failure = None
    try:
        assert len(recorder.records) >= 1
        for i, rec in enumerate(recorder.records):
            assert rec.loss is not None and torch.isfinite(torch.tensor(rec.loss))
            assert rec.grad_noise_invocations == 1
            if rec.grad_noise_norm is not None:
                assert torch.isfinite(torch.tensor(rec.grad_noise_norm))
    except AssertionError as e:
        failure = str(e)
        raise
    finally:
        report = RunReport(
            run_name="relative_gradient_noise",
            dataset_manifest=staged_dataset["manifest"],
            initial_lora_checksum=initial_lora["checksum"],
            final_lora_checksum=(
                file_checksum(results["saved_lora_path"]) if results.get("saved_lora_path") else ""
            ),
            saved_lora_path=results.get("saved_lora_path", ""),
            hook_counts=summary,
            per_step_losses=[r.loss for r in recorder.records],
            peak_allocated_vram_gb=results["peak_alloc_gb"],
            runtime_seconds=results["elapsed"],
            failure_detail=failure,
        )
        write_json_report(report, os.path.join(output_root, "relative_gradient_noise", "report.json"))


# ---------------------------------------------------------------------------
# Test 7: Neelakantan Gradient Noise
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.gpu
def test_neelakantan_gradient_noise(
    is_integration_enabled, has_cuda, dataset_root, staged_dataset, initial_lora, output_root
):
    """Neelakantan gradient noise: decays per step using optimizer-step count."""
    _check_and_skip(is_integration_enabled, has_cuda, dataset_root, initial_lora)

    config = build_krea_config(
        name="phase1_neel_gnoise",
        dataset_path=staged_dataset["path"],
        output_path=output_root,
        steps=3,
        weight_noise={"enabled": False},
        gradient_noise={
            "enabled": True,
            "mode": "neelakantan",
            "eta": 0.01,
            "gamma": 0.55,
            "log_every": 1,
        },
        pretrained_lora_path=initial_lora["path"],
    )

    recorder = NoisingRecorder()
    results = run_krea_job(config, output_root, recorder)
    summary = recorder.summary()

    failure = None
    try:
        assert len(recorder.records) >= 2, (
            f"Need at least 2 steps for decay check, got {len(recorder.records)}"
        )
        for i, rec in enumerate(recorder.records):
            assert rec.loss is not None and torch.isfinite(torch.tensor(rec.loss)), (
                f"Step {i}: loss not finite"
            )
            assert rec.grad_noise_invocations == 1, (
                f"Step {i}: grad noise ran {rec.grad_noise_invocations} times"
            )

            eta = 0.01
            gamma = 0.55
            step = rec.optimizer_step
            expected_scale = eta / max(1.0, (1.0 + step) ** gamma)
            assert expected_scale > 0, (
                f"Step {i}: expected Neelakantan scale > 0, got {expected_scale}"
            )

        scales = [
            0.01 / max(1.0, (1.0 + rec.optimizer_step) ** 0.55) for rec in recorder.records
        ]
        if len(scales) >= 2:
            assert scales[1] < scales[0], (
                f"Neelakantan decay check failed: scale[0]={scales[0]:.6f} >= scale[1]={scales[1]:.6f}"
            )
    except AssertionError as e:
        failure = str(e)
        raise
    finally:
        report = RunReport(
            run_name="neelakantan_gradient_noise",
            dataset_manifest=staged_dataset["manifest"],
            initial_lora_checksum=initial_lora["checksum"],
            final_lora_checksum=(
                file_checksum(results["saved_lora_path"]) if results.get("saved_lora_path") else ""
            ),
            saved_lora_path=results.get("saved_lora_path", ""),
            hook_counts=summary,
            per_step_losses=[r.loss for r in recorder.records],
            peak_allocated_vram_gb=results["peak_alloc_gb"],
            runtime_seconds=results["elapsed"],
            failure_detail=failure,
        )
        write_json_report(report, os.path.join(output_root, "neelakantan_gradient_noise", "report.json"))


# ---------------------------------------------------------------------------
# Test 8: Gradient Accumulation
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.gpu
def test_gradient_accumulation(
    is_integration_enabled, has_cuda, dataset_root, staged_dataset, initial_lora, output_root
):
    """Gradient accumulation: injectors fire once per optimizer step, not per microbatch."""
    _check_and_skip(is_integration_enabled, has_cuda, dataset_root, initial_lora)

    config = build_krea_config(
        name="phase1_grad_accum",
        dataset_path=staged_dataset["path"],
        output_path=output_root,
        steps=4,
        gradient_accumulation=1,
        gradient_accumulation_steps=2,
        weight_noise={"enabled": True, "mode": "absolute", "sigma": 0.01, "log_every": 1},
        gradient_noise={"enabled": True, "mode": "absolute", "sigma": 0.001, "log_every": 1},
        use_ema=True,
        ema_decay=0.99,
        pretrained_lora_path=initial_lora["path"],
    )

    recorder = NoisingRecorder()
    results = run_krea_job(config, output_root, recorder)
    summary = recorder.summary()

    failure = None
    try:
        optimizer_steps = len(recorder.records)
        assert optimizer_steps >= 2, f"Expected >= 2 optimizer steps, got {optimizer_steps}"

        assert summary["microbatches"] == 4, f"Expected 4 microbatches, got {summary['microbatches']}"
        assert summary["grad_noise_calls"] == optimizer_steps, (
            f"Grad noise calls {summary['grad_noise_calls']} != optimizer steps {optimizer_steps}"
        )
        assert summary["weight_noise_calls"] == optimizer_steps, (
            f"Weight noise calls {summary['weight_noise_calls']} != optimizer steps {optimizer_steps}"
        )
        assert summary["fisher_calls"] == optimizer_steps, (
            f"Fisher calls {summary['fisher_calls']} != optimizer steps {optimizer_steps}"
        )
        assert summary["ema_calls"] == optimizer_steps, (
            f"EMA calls {summary['ema_calls']} != optimizer steps {optimizer_steps}"
        )

        for i, rec in enumerate(recorder.records):
            assert rec.loss is not None and torch.isfinite(torch.tensor(rec.loss)), (
                f"Opt step {i}: loss not finite"
            )
            assert rec.grad_noise_invocations == 1, (
                f"Opt step {i}: grad noise fired {rec.grad_noise_invocations} times"
            )
            assert rec.weight_noise_invocations == 1, (
                f"Opt step {i}: weight noise fired {rec.weight_noise_invocations} times"
            )
            assert rec.fisher_invocations == 1, (
                f"Opt step {i}: fisher fired {rec.fisher_invocations} times"
            )
    except AssertionError as e:
        failure = str(e)
        raise
    finally:
        report = RunReport(
            run_name="gradient_accumulation",
            dataset_manifest=staged_dataset["manifest"],
            initial_lora_checksum=initial_lora["checksum"],
            final_lora_checksum=(
                file_checksum(results["saved_lora_path"]) if results.get("saved_lora_path") else ""
            ),
            saved_lora_path=results.get("saved_lora_path", ""),
            hook_counts=summary,
            per_step_losses=[r.loss for r in recorder.records],
            peak_allocated_vram_gb=results["peak_alloc_gb"],
            runtime_seconds=results["elapsed"],
            failure_detail=failure,
        )
        write_json_report(report, os.path.join(output_root, "gradient_accumulation", "report.json"))


# ---------------------------------------------------------------------------
# Test 9: EMA Ordering
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.gpu
def test_ema_ordering(
    is_integration_enabled, has_cuda, dataset_root, staged_dataset, initial_lora, output_root
):
    """EMA ordering: shadow tracks clean post-optimizer, live param gets noise after."""
    _check_and_skip(is_integration_enabled, has_cuda, dataset_root, initial_lora)

    config = build_krea_config(
        name="phase1_ema_order",
        dataset_path=staged_dataset["path"],
        output_path=output_root,
        steps=2,
        weight_noise={
            "enabled": True,
            "mode": "absolute",
            "sigma": 0.01,
            "bound_norm": False,
            "log_every": 1,
        },
        gradient_noise={"enabled": False},
        use_ema=True,
        ema_decay=0.99,
        pretrained_lora_path=initial_lora["path"],
    )

    recorder = NoisingRecorder()
    results = run_krea_job(config, output_root, recorder)
    summary = recorder.summary()

    failure = None
    try:
        assert len(recorder.records) >= 1
        for i, rec in enumerate(recorder.records):
            assert rec.loss is not None and torch.isfinite(torch.tensor(rec.loss))
            assert rec.weight_noise_invocations == 1
            assert rec.ema_invocations == 1

            pname = recorder.tagged_param_name
            if rec.live_param_before_ema and rec.tagged_after_weight_noise:
                live_before_ema = rec.live_param_before_ema[pname]
                noised = rec.tagged_after_weight_noise[pname]

                # The live param must have received noise (differ from pre-noise state)
                if rec.tagged_after_optimizer:
                    clean = rec.tagged_after_optimizer[pname]
                    noised_delta = (noised - clean).abs().max().item()
                    assert noised_delta > 0, (
                        f"Step {i}: live param did not receive weight noise "
                        f"(delta from clean={noised_delta})"
                    )

                if rec.ema_shadow_after_update:
                    shadow = rec.ema_shadow_after_update[pname]
                    # Shadow must NOT equal the noised param (EMA ran on clean, not noised)
                    assert not torch.equal(shadow, noised), (
                        f"Step {i}: EMA shadow matches noised param "
                        f"-- EMA may be tracking noised weights"
                    )
                    # Shadow must be finite
                    assert torch.isfinite(shadow).all(), (
                        f"Step {i}: EMA shadow has NaN/Inf"
                    )
    except AssertionError as e:
        failure = str(e)
        raise
    finally:
        report = RunReport(
            run_name="ema_ordering",
            dataset_manifest=staged_dataset["manifest"],
            initial_lora_checksum=initial_lora["checksum"],
            final_lora_checksum=(
                file_checksum(results["saved_lora_path"]) if results.get("saved_lora_path") else ""
            ),
            saved_lora_path=results.get("saved_lora_path", ""),
            hook_counts=summary,
            per_step_losses=[r.loss for r in recorder.records],
            peak_allocated_vram_gb=results["peak_alloc_gb"],
            runtime_seconds=results["elapsed"],
            failure_detail=failure,
        )
        write_json_report(report, os.path.join(output_root, "ema_ordering", "report.json"))


# ---------------------------------------------------------------------------
# Test 10: Fisher Trace
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.gpu
def test_fisher_trace(
    is_integration_enabled, has_cuda, dataset_root, staged_dataset, initial_lora, output_root
):
    """Fisher trace: finite, non-zero, from optimizer state not stale grads, once/step."""
    _check_and_skip(is_integration_enabled, has_cuda, dataset_root, initial_lora)

    config = build_krea_config(
        name="phase1_fisher",
        dataset_path=staged_dataset["path"],
        output_path=output_root,
        steps=2,
        weight_noise={"enabled": True, "mode": "absolute", "sigma": 0.01, "log_every": 1},
        gradient_noise={"enabled": True, "mode": "absolute", "sigma": 0.001, "log_every": 1},
        pretrained_lora_path=initial_lora["path"],
    )

    recorder = NoisingRecorder()
    results = run_krea_job(config, output_root, recorder)
    summary = recorder.summary()

    failure = None
    try:
        assert len(recorder.records) >= 1
        for i, rec in enumerate(recorder.records):
            assert rec.loss is not None and torch.isfinite(torch.tensor(rec.loss))
            assert rec.fisher_invocations == 1, (
                f"Step {i}: fisher fired {rec.fisher_invocations} times"
            )

            assert rec.fisher_trace is not None, f"Step {i}: fisher_trace is None"
            assert torch.isfinite(torch.tensor(rec.fisher_trace)), (
                f"Step {i}: fisher_trace not finite"
            )
            assert rec.fisher_trace > 0, (
                f"Step {i}: fisher_trace={rec.fisher_trace} expected > 0. "
                f"If 0, fisher may be reading stale/zeroed optimizer state."
            )
    except AssertionError as e:
        failure = str(e)
        raise
    finally:
        report = RunReport(
            run_name="fisher_trace",
            dataset_manifest=staged_dataset["manifest"],
            initial_lora_checksum=initial_lora["checksum"],
            final_lora_checksum=(
                file_checksum(results["saved_lora_path"]) if results.get("saved_lora_path") else ""
            ),
            saved_lora_path=results.get("saved_lora_path", ""),
            hook_counts=summary,
            per_step_losses=[r.loss for r in recorder.records],
            peak_allocated_vram_gb=results["peak_alloc_gb"],
            runtime_seconds=results["elapsed"],
            failure_detail=failure,
        )
        write_json_report(report, os.path.join(output_root, "fisher_trace", "report.json"))


# ---------------------------------------------------------------------------
# Test 11: Saved LoRA Checksum and Reload Verification
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.gpu
def test_saved_lora_checksums_and_reload(
    is_integration_enabled, has_cuda, dataset_root, staged_dataset, initial_lora, output_root
):
    """Verify: final checksums differ from initial, saved tensors are finite and changed."""
    _check_and_skip(is_integration_enabled, has_cuda, dataset_root, initial_lora)

    from testing.integration._helpers.lora_state import load_lora_tensors

    config = build_krea_config(
        name="phase1_checksum",
        dataset_path=staged_dataset["path"],
        output_path=output_root,
        steps=2,
        weight_noise={"enabled": True, "mode": "absolute", "sigma": 0.01, "log_every": 1},
        pretrained_lora_path=initial_lora["path"],
    )

    recorder = NoisingRecorder()
    results = run_krea_job(config, output_root, recorder)

    failure = None
    try:
        saved_path = results["saved_lora_path"]
        assert saved_path is not None and os.path.isfile(saved_path), (
            f"Saved LoRA not found at {saved_path}"
        )

        final_checksum = file_checksum(saved_path)
        assert final_checksum != initial_lora["checksum"], (
            f"Final checksum {final_checksum} == initial {initial_lora['checksum']}: "
            f"LoRA did not change from initial state"
        )

        saved_tensors = load_lora_tensors(saved_path)
        assert len(saved_tensors) > 0, "Saved LoRA has no tensors"

        first_key = sorted(saved_tensors.keys())[0]
        first_tensor = saved_tensors[first_key]
        assert torch.isfinite(first_tensor).all(), f"Saved tensor {first_key} has NaN/Inf"

        initial_tensors = load_lora_tensors(initial_lora["path"])
        matching_keys = set(saved_tensors.keys()) & set(initial_tensors.keys())
        assert len(matching_keys) > 0, "No matching keys between initial and final LoRA"

        changed = False
        for key in list(matching_keys)[:5]:
            diff = (saved_tensors[key].float() - initial_tensors[key].float()).abs().max().item()
            if diff > 0:
                changed = True
                break
        assert changed, (
            "No saved tensor differs from initial -- training + noising had no effect on saved weights"
        )
    except AssertionError as e:
        failure = str(e)
        raise
    finally:
        report = RunReport(
            run_name="saved_lora_verification",
            dataset_manifest=staged_dataset["manifest"],
            initial_lora_checksum=initial_lora["checksum"],
            final_lora_checksum=(
                file_checksum(results["saved_lora_path"]) if results.get("saved_lora_path") else ""
            ),
            saved_lora_path=results.get("saved_lora_path", ""),
            hook_counts=recorder.summary(),
            per_step_losses=[r.loss for r in recorder.records],
            peak_allocated_vram_gb=results["peak_alloc_gb"],
            runtime_seconds=results["elapsed"],
            failure_detail=failure,
        )
        write_json_report(report, os.path.join(output_root, "saved_lora_verification", "report.json"))


# ---------------------------------------------------------------------------
# Test 12 (Layer B): Matched Operational Smoke Comparison
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.gpu
def test_matched_smoke_comparison(
    is_integration_enabled, has_cuda, dataset_root, staged_dataset, initial_lora, output_root
):
    """25-step matched comparison: baseline vs relative weight noise vs neelakantan gradient noise."""
    _check_and_skip(is_integration_enabled, has_cuda, dataset_root, initial_lora)

    steps = int(os.environ.get("AI_TOOLKIT_NOISING_TEST_STEPS", "25"))
    smoke_dir = os.path.join(output_root, "smoke")

    run_configs = {
        "baseline": {
            "weight_noise": {"enabled": False},
            "gradient_noise": {"enabled": False},
        },
        "rel_weight_noise": {
            "weight_noise": {
                "enabled": True,
                "mode": "relative",
                "sigma": 0.00125,
                "bound_norm": False,
                "log_every": 5,
            },
            "gradient_noise": {"enabled": False},
        },
        "neelakantan_grad_noise": {
            "weight_noise": {"enabled": False},
            "gradient_noise": {
                "enabled": True,
                "mode": "neelakantan",
                "eta": 0.01,
                "gamma": 0.55,
                "log_every": 5,
            },
        },
    }

    failures = []
    for run_name, noise_config in run_configs.items():
        config = build_krea_config(
            name=f"phase1_smoke_{run_name}",
            dataset_path=staged_dataset["path"],
            output_path=smoke_dir,
            steps=steps,
            pretrained_lora_path=initial_lora["path"],
            **noise_config,
        )

        recorder = NoisingRecorder()
        results = run_krea_job(config, smoke_dir, recorder)
        summary = recorder.summary()

        try:
            assert results["saved_lora_path"] is not None, f"{run_name}: no LoRA saved"
            assert len(recorder.records) == steps, (
                f"{run_name}: expected {steps} steps, got {len(recorder.records)}"
            )
            for i, rec in enumerate(recorder.records):
                assert rec.loss is not None and torch.isfinite(torch.tensor(rec.loss)), (
                    f"{run_name} step {i}: loss not finite"
                )

            if run_name != "baseline":
                if "weight" in run_name:
                    assert summary["weight_noise_calls"] > 0, f"{run_name}: no weight noise calls"
                    for rec in recorder.records:
                        if rec.weight_noise_norm is not None:
                            assert rec.weight_noise_norm > 0, f"{run_name}: weight_noise_norm is zero"
                            break
                if "grad" in run_name:
                    assert summary["grad_noise_calls"] > 0, f"{run_name}: no grad noise calls"
                    for rec in recorder.records:
                        if rec.grad_noise_norm is not None:
                            assert rec.grad_noise_norm > 0, f"{run_name}: grad_noise_norm is zero"
                            break
            else:
                # Disabled baseline: methods are called but produce no metrics
                for rec in recorder.records:
                    assert rec.weight_noise_norm is None, f"{run_name}: unexpected weight_noise_norm"
                    assert rec.grad_noise_norm is None, f"{run_name}: unexpected grad_noise_norm"

            assert summary["all_tagged_count"] > 0, f"{run_name}: no tagged params"
        except AssertionError as e:
            failures.append(f"{run_name}: {e}")

        report = RunReport(
            run_name=f"smoke_{run_name}",
            dataset_manifest=staged_dataset["manifest"],
            initial_lora_checksum=initial_lora["checksum"],
            final_lora_checksum=(
                file_checksum(results["saved_lora_path"]) if results.get("saved_lora_path") else ""
            ),
            saved_lora_path=results.get("saved_lora_path", ""),
            hook_counts=summary,
            per_step_losses=[r.loss for r in recorder.records],
            peak_allocated_vram_gb=results["peak_alloc_gb"],
            peak_reserved_vram_gb=results["peak_reserved_gb"],
            runtime_seconds=results["elapsed"],
        )
        write_json_report(report, os.path.join(smoke_dir, f"smoke_{run_name}", "report.json"))

    if failures:
        pytest.fail("Smoke comparison failures:\n" + "\n".join(failures))
