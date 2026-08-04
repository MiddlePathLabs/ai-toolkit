"""Shared test runner: launches a Krea 2 job with instrumentation and returns results."""
import gc
import json
import os
import shutil
import sys
import time

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def run_krea_job(
    config: dict,
    output_root: str,
    recorder=None,
) -> dict:
    """Run a Krea 2 training job to completion and return a results dict.

    Args:
        config: Full AI-Toolkit job config dict.
        output_root: Base directory for test outputs.
        recorder: Optional NoisingRecorder to install for instrumentation.

    Returns:
        dict with keys: saved_lora_path, save_root, elapsed,
                        peak_alloc_gb, peak_reserved_gb
    """
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    from toolkit.job import get_job
    from testing.integration._helpers.lora_state import find_saved_lora

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    name = config["config"]["name"]
    training_folder = config["config"]["process"][0]["training_folder"]
    save_root = os.path.join(training_folder, name)

    # Clean up any stale checkpoint from a previous test run so the trainer
    # starts fresh instead of resuming from a prior step.
    if os.path.isdir(save_root):
        shutil.rmtree(save_root, ignore_errors=True)

    job = get_job(config)
    process = job.process[0]

    if recorder is not None:
        recorder.install(process)

    start = time.time()
    try:
        job.run()
    finally:
        elapsed = time.time() - start

    peak_alloc = None
    peak_reserved = None
    if torch.cuda.is_available():
        peak_alloc = torch.cuda.max_memory_allocated() / 1e9
        peak_reserved = torch.cuda.max_memory_reserved() / 1e9

    saved_path = find_saved_lora(training_folder, name)

    results = {
        "saved_lora_path": saved_path,
        "save_root": save_root,
        "training_folder": training_folder,
        "elapsed": elapsed,
        "peak_alloc_gb": peak_alloc,
        "peak_reserved_gb": peak_reserved,
    }

    try:
        job.cleanup()
    except Exception:
        pass

    # Release the recorder's live GPU parameter references before GC.
    if recorder is not None:
        recorder.release_live_params()

    # Aggressive GPU memory cleanup: move models to CPU and null references
    # before GC so the next test can load a fresh model without OOM.
    try:
        if hasattr(process, "sd") and process.sd is not None:
            if hasattr(process.sd, "unet") and process.sd.unet is not None:
                process.sd.unet.cpu()
            if hasattr(process.sd, "vae") and process.sd.vae is not None:
                process.sd.vae.cpu()
            te = getattr(process.sd, "text_encoder", None)
            if te is not None:
                if isinstance(te, list):
                    for t in te:
                        t.cpu()
                else:
                    te.cpu()
            process.sd = None
        process.optimizer = None
        process.network = None
        process.ema = None
        process.modules_being_trained = []
    except Exception:
        pass

    del process
    del job
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return results


def create_seed_lora(
    config: dict,
    output_root: str,
) -> str:
    """Run a 1-step seed job to create an initial LoRA, return its path.

    The seed config uses disabled noising and steps=1 to produce a baseline
    LoRA from deterministic initialization. All comparison runs load this
    via pretrained_lora_path.
    """
    seed_config = json.loads(json.dumps(config))
    seed_config["config"]["name"] = seed_config["config"]["name"] + "_seed"
    proc = seed_config["config"]["process"][0]
    proc["train"]["steps"] = 1
    proc["train"]["weight_noise"] = {"enabled": False}
    proc["train"]["gradient_noise"] = {"enabled": False}
    proc["save"]["save_every"] = 1

    results = run_krea_job(seed_config, output_root, recorder=None)

    saved = results["saved_lora_path"]
    if saved is None or not os.path.isfile(saved):
        raise RuntimeError(f"Seed run did not produce a saved LoRA at {results['save_root']}")
    return saved
