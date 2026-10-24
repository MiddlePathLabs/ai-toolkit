"""Run only after correctness passes: fresh full-workload native/kitchen A/B jobs.

Example (eager first; generated artifacts stay outside the checkout):
  python scripts/benchmark_krea_convrot8.py --output E:/ai-toolkit/eval_out/kitchen_ab_eager
Repeat separately with --compiled and a different --output after eager passes.
Wall samples synchronize at existing train_loop boundaries. Optional CUDA-event
phase timings include checkpoint recomputation in backward, not forward.
"""

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import inspect
import io
import json
import os
import runpy
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT.parent / "KREA_AB_BASELINE.json"
GPU_UUID = "GPU-84ff6d69-fa9c-1800-7c35-0e5376a446aa"
GPU_FIELDS = "uuid,name,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu,clocks.sm,clocks.mem"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def smi(query, gpu=None):
    command = ["nvidia-smi", query, "--format=csv,noheader,nounits"]
    if gpu:
        command += ["--id", gpu]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return list(csv.reader(io.StringIO(result.stdout), skipinitialspace=True))


def idle_gpu(uuid):
    row = smi("--query-gpu=" + GPU_FIELDS, uuid)[0]
    apps = smi("--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory")
    if "RTX 5090" not in row[1]:
        raise RuntimeError(f"Expected RTX 5090, found {row}")
    if float(row[2]) > 256 or float(row[4]) > 5 or any(r[0] == uuid for r in apps):
        raise RuntimeError(
            f"GPU is busy; do not compare shared workloads: {row}; {apps}"
        )
    return row


def distribution(values):
    if not values:
        return None
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": ordered[0],
        "max": ordered[-1],
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p10": ordered[round((len(ordered) - 1) * 0.10)],
        "p90": ordered[round((len(ordered) - 1) * 0.90)],
        "p95": ordered[round((len(ordered) - 1) * 0.95)],
    }


def dataset_manifest(process):
    suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".txt", ".caption"}
    return [
        {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for dataset in process["datasets"]
        for path in sorted(Path(dataset["folder_path"]).rglob("*"))
        if path.is_file() and path.suffix.lower() in suffixes
    ]


def tensor_fingerprint(tensor):
    if tensor is None:
        return None
    import torch

    if not isinstance(tensor, torch.Tensor):
        return tensor
    raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
    digest = hashlib.sha256()
    for offset in range(0, raw.numel(), 16 * 1024 * 1024):
        chunk = raw[offset : offset + 16 * 1024 * 1024].cpu()
        digest.update(memoryview(chunk.numpy()))
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "bytes": raw.numel(),
        "sha256": digest.hexdigest(),
    }


def workload_invariant(config):
    workload = copy.deepcopy(config)
    workload["config"].pop("name", None)
    for process in workload["config"]["process"]:
        for key in ("name", "training_folder"):
            process.pop(key, None)
        for key in ("convrot8_backend", "convrot8_backend_strict"):
            process["model"].pop(key, None)
    return workload


def validate_pair(report, reference):
    invariants = report.get("paired_invariants", {})
    initial = invariants.get("initial_state") or {}
    failures = []
    if not initial.get("quantized_buffers") or not initial.get("trainable_parameters"):
        failures.append("missing initial quantized buffers or trainable parameters")
    if len(invariants.get("warmup_inputs", [])) != 2:
        failures.append("expected exactly two recorded warmup prediction inputs")
    if reference:
        expected = reference.get("paired_invariants", {})
        failures.extend(
            key
            for key in ("workload", "initial_state", "warmup_inputs")
            if invariants.get(key) != expected.get(key)
        )
    return {
        "matched_reference": bool(reference),
        "failures": failures,
        "passed": not failures,
    }


def gpu_sampler(uuid, stop, samples, path):
    with path.open("w", encoding="utf-8") as log:
        while not stop.is_set():
            try:
                row = {
                    "unix_s": time.time(),
                    "gpu": smi("--query-gpu=" + GPU_FIELDS, uuid)[0],
                    "processes": smi(
                        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory"
                    ),
                }
            except (OSError, subprocess.CalledProcessError) as exc:
                row = {"unix_s": time.time(), "error": str(exc)}
            samples.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            stop.wait(1.0)


def run_pair(args):
    source = json.loads(args.config.read_text(encoding="utf-8"))
    if len(source["config"]["process"]) != 1:
        raise ValueError("Benchmark requires exactly one original trainer process")
    original = source["config"]["process"][0]
    if (
        original["model"]["arch"] != "krea2"
        or original["model"].get("qtype") != "convrot8"
    ):
        raise ValueError("The baseline must be Krea2 with existing ConvRot8 weights")
    steps = int(original["train"]["steps"])
    if not 100 <= steps <= 200 or not 0 < args.warmup < steps:
        raise ValueError(
            "Preserve a 100-200-step baseline and exclude a nonzero warmup"
        )
    if args.output.resolve().is_relative_to(ROOT):
        raise ValueError("Place generated benchmark artifacts outside the checkout")
    idle = idle_gpu(args.gpu_uuid)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "baseline_path": str(args.config.resolve()),
        "baseline": source,
        "baseline_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "dataset": dataset_manifest(original),
        "gpu_idle_before": idle,
        "gpu_fields": GPU_FIELDS.split(","),
        "compiled": args.compiled,
        "warmup": args.warmup,
        "python": sys.executable,
        "method": "fresh run.py jobs; synchronized existing train_loop boundaries; completion intervals include data/logging; phase CUDA events optional",
    }
    write_json(args.output / "manifest.json", manifest)
    results = []
    for repeat in range(args.repeats):
        order = (
            ["native", "comfy_kitchen"]
            if repeat % 2 == 0
            else ["comfy_kitchen", "native"]
        )
        for backend in order:
            idle_gpu(args.gpu_uuid)
            directory = args.output / f"{repeat + 1:02d}_{backend}"
            directory.mkdir()
            config = copy.deepcopy(source)
            name = f"krea_convrot8_{'compiled' if args.compiled else 'eager'}_{repeat + 1}_{backend}"
            config["config"]["name"] = name
            process = config["config"]["process"][0]
            process["name"] = "[name]"
            process["training_folder"] = str(directory / "training")
            process["model"]["convrot8_backend"] = backend
            process["model"]["convrot8_backend_strict"] = backend == "comfy_kitchen"
            if args.compiled:
                process["model"]["compile"] = True
                process["model"]["block_compile"] = False
            config_path = directory / "config.json"
            write_json(config_path, config)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                str(config_path),
                "--output",
                str(directory),
                "--warmup",
                str(args.warmup),
            ]
            if args.phase_timing:
                command.append("--phase-timing")
            env = os.environ.copy()
            env.update(
                CUDA_VISIBLE_DEVICES=args.gpu_uuid,
                CUDA_DEVICE_ORDER="PCI_BUS_ID",
                SEED="42",
                PYTHONUNBUFFERED="1",
                PYTHONHASHSEED="42",
            )
            if args.compiled:
                env["TORCH_LOGS"] = "recompiles,graph_breaks"
            stop, samples = threading.Event(), []
            sampler = threading.Thread(
                target=gpu_sampler,
                args=(args.gpu_uuid, stop, samples, directory / "nvml.jsonl"),
                daemon=True,
            )
            sampler.start()
            started = time.perf_counter()
            try:
                with (directory / "console.log").open("w", encoding="utf-8") as log:
                    print(
                        f"Running {name}; log: {directory / 'console.log'}", flush=True
                    )
                    completed = subprocess.run(
                        command,
                        cwd=ROOT,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
            finally:
                stop.set()
                sampler.join()
            report_path = directory / "report.json"
            report = (
                json.loads(report_path.read_text(encoding="utf-8"))
                if report_path.exists()
                else {}
            )
            report.update(
                backend=backend,
                repeat=repeat + 1,
                returncode=completed.returncode,
                process_wall_s=time.perf_counter() - started,
            )
            steady = [
                r
                for r in samples
                if "gpu" in r
                and (report.get("steady_start_unix_s") or float("inf"))
                <= r["unix_s"]
                <= (report.get("steady_end_unix_s") or 0)
            ]
            report["nvml_device_memory_MiB"] = distribution(
                [float(r["gpu"][2]) for r in steady]
            )
            report["nvml_sample_errors"] = [r for r in samples if "error" in r]
            report["competing_processes"] = [
                r
                for r in steady
                if any(
                    p[0] == args.gpu_uuid and int(p[1]) != report.get("pid")
                    for p in r.get("processes", [])
                )
            ]
            report["paired_validation"] = validate_pair(
                report, results[0] if results else None
            )
            if not report["paired_validation"]["passed"]:
                report["complete"] = False
            if report["competing_processes"]:
                report["complete"] = False
                report["interference_error"] = (
                    "Other GPU compute processes were observed during steady timing"
                )
            write_json(report_path, report)
            results.append(report)
            write_json(args.output / "results.json", results)
            if completed.returncode or not report.get("complete"):
                raise RuntimeError(f"Benchmark failed; inspect {directory}")


def run_worker(args):
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    started = time.perf_counter()
    entry = runpy.run_path(str(ROOT / "run.py"), run_name="benchmark_entry")
    import torch

    import toolkit.util.convrot_kitchen as kitchen
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer

    config = json.loads(args.worker.read_text(encoding="utf-8"))
    process_config = config["config"]["process"][0]
    backend = process_config["model"]["convrot8_backend"]
    setup_times, rows, phase_events = [], [], {}
    state = {"trainer": None, "previous_end": None, "step": 0}
    original_configure = kitchen.configure_krea_convrot8

    def configure(*a, **kw):
        begin = time.perf_counter()
        result = original_configure(*a, **kw)
        setup_times.append(time.perf_counter() - begin)
        return result

    kitchen.configure_krea_convrot8 = configure
    original_before = SDTrainer.hook_before_train_loop

    def before(trainer):
        original_before(trainer)
        state["trainer"] = trainer
        state["diagnostics"] = getattr(trainer.sd, "convrot8_diagnostics", None)
        timer_start, timer_stop = trainer.timer.start, trainer.timer.stop
        state["trainable_parameters"] = [
            {
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
            }
            for name, parameter in trainer.network.named_parameters()
            if parameter.requires_grad
        ]
        hash_started = time.perf_counter()
        quantized_modules = [
            (name, module)
            for name, module in trainer.sd.unet.named_modules()
            if isinstance(
                getattr(module, "ostris_quantizer", None), kitchen.ConvRotInt8Quantizer
            )
            and module.ostris_quantizer.qtype == "convrot8"
        ]
        state["native_coverage"] = {
            "requested_backend": backend,
            "selected_backend": "native",
            "total_layers": len(quantized_modules),
            "selected_layer_counts": {
                "native": len(quantized_modules),
                "comfy_kitchen": 0,
            },
            "observed_training_layer_counts": {},
            "runtime_observation": "native quantizer types counted; no per-layer execution hooks",
        }
        state["initial_state"] = {
            "quantized_buffers": [
                {
                    "name": name + "." + buffer,
                    **tensor_fingerprint(getattr(module, buffer)),
                }
                for name, module in quantized_modules
                for buffer in ("cr8_qdata", "cr8_scales")
            ],
            "trainable_parameters": [
                {"name": name, **tensor_fingerprint(parameter)}
                for name, parameter in trainer.network.named_parameters()
                if parameter.requires_grad
            ],
        }
        state["initial_state_hash_s"] = time.perf_counter() - hash_started
        state["warmup_inputs"] = []
        process_batch = trainer.process_general_training_batch
        original_predict = trainer.predict_noise
        predict_signature = inspect.signature(original_predict)

        def process_general_batch(batch, *a, **kw):
            result = process_batch(batch, *a, **kw)
            if state["step"] <= args.warmup and len(state["warmup_inputs"]) < 2:
                state["pending_warmup_batch"] = {
                    "sample_paths": [str(item.path) for item in batch.file_items],
                    "clean_latents": tensor_fingerprint(batch.latents),
                    "noise": tensor_fingerprint(result[1]),
                    "processed_timesteps": tensor_fingerprint(result[2]),
                    "conditioned_prompts": list(result[3]),
                }
            return result

        def predict(*a, **kw):
            if state["step"] <= args.warmup and len(state["warmup_inputs"]) < 2:
                arguments = predict_signature.bind(*a, **kw).arguments
                record = dict(state.get("pending_warmup_batch", {}))
                record.update(
                    step=state["step"],
                    noisy_latents=tensor_fingerprint(arguments["noisy_latents"]),
                    timesteps=tensor_fingerprint(arguments.get("timesteps", 1)),
                )
                state["warmup_inputs"].append(record)
            return original_predict(*a, **kw)

        trainer.process_general_training_batch = process_general_batch
        trainer.predict_noise = predict

        def start(name):
            if name == "train_loop":
                torch.cuda.synchronize()
                state["begin"] = time.perf_counter()
                state.setdefault("init_s", state["begin"] - started)
                state["step"] += 1
                if state["step"] == 1:
                    state["initialization_peak_allocated_bytes"] = (
                        torch.cuda.max_memory_allocated()
                    )
                    state["initialization_peak_reserved_bytes"] = (
                        torch.cuda.max_memory_reserved()
                    )
                phase_events.clear()
                if state["step"] == args.warmup + 1:
                    torch.cuda.reset_peak_memory_stats()
                    state["steady_start_unix_s"] = time.time()
                    diagnostics = getattr(trainer.sd, "convrot8_diagnostics", None)
                    if backend == "comfy_kitchen":
                        diagnostics.require_complete()
                    if process_config["model"].get("compile"):
                        print("BENCH_STEADY_START", flush=True)
            return timer_start(name)

        def stop(name):
            result = timer_stop(name)
            if name != "train_loop":
                return result
            torch.cuda.synchronize()
            end = time.perf_counter()
            row = {
                "step": state["step"],
                "unix_s": time.time(),
                "loop_wall_s": end - state["begin"],
                "completion_interval_s": end - state["previous_end"]
                if state["previous_end"]
                else None,
                "allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            for phase, events in phase_events.items():
                row[phase + "_cuda_ms"] = sum(a.elapsed_time(b) for a, b in events)
            if not trainer._last_optimizer_update_success:
                raise RuntimeError(
                    "A skipped/OOM optimizer update invalidates this comparison"
                )
            rows.append(row)
            state["previous_end"] = end
            return result

        trainer.timer.start, trainer.timer.stop = start, stop
        if args.phase_timing:

            def timed(function, phase):
                def call(*a, **kw):
                    begin, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    begin.record()
                    result = function(*a, **kw)
                    end.record()
                    phase_events.setdefault(phase, []).append((begin, end))
                    return result

                return call

            trainer.predict_noise = timed(trainer.predict_noise, "forward")
            trainer.accelerator.backward = timed(
                trainer.accelerator.backward, "backward_with_recompute"
            )

    SDTrainer.hook_before_train_loop = before
    error = None
    try:
        sys.argv = [
            str(ROOT / "run.py"),
            str(args.worker),
            "--log",
            str(args.output / "training.log"),
        ]
        entry["main"]()
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        steady = rows[args.warmup :]
        if process_config["model"].get("compile"):
            print("BENCH_STEADY_END", flush=True)
        diagnostics = state.get("diagnostics")
        coverage = (
            diagnostics.report()
            if diagnostics
            else state.get("native_coverage", {"requested_backend": backend})
        )
        complete = error is None and len(rows) == process_config["train"]["steps"]
        coverage_error = None
        if backend == "comfy_kitchen" and complete:
            try:
                diagnostics.require_complete()
            except RuntimeError as exc:
                complete, coverage_error = False, str(exc)
        metrics = [
            "loop_wall_s",
            "completion_interval_s",
            "forward_cuda_ms",
            "backward_with_recompute_cuda_ms",
        ]
        report = {
            "pid": os.getpid(),
            "complete": complete,
            "error": error,
            "coverage_error": coverage_error,
            "coverage": coverage,
            "startup_to_first_step_s": state.get("init_s"),
            "backend_setup_s": setup_times,
            "initialization_peak_allocated_bytes": state.get(
                "initialization_peak_allocated_bytes"
            ),
            "initialization_peak_reserved_bytes": state.get(
                "initialization_peak_reserved_bytes"
            ),
            "trainable_parameters": state.get("trainable_parameters"),
            "initial_state_hash_s": state.get("initial_state_hash_s"),
            "startup_without_state_hash_s": state.get("init_s", 0)
            - state.get("initial_state_hash_s", 0),
            "paired_invariants": {
                "workload": workload_invariant(config),
                "initial_state": state.get("initial_state"),
                "warmup_inputs": state.get("warmup_inputs", []),
            },
            "warmup_steps": args.warmup,
            "steady_steps": len(steady),
            "steady_start_unix_s": state.get("steady_start_unix_s"),
            "steady_end_unix_s": steady[-1]["unix_s"] if steady else None,
            "steady": {
                key: distribution([r[key] for r in steady if r.get(key) is not None])
                for key in metrics
            },
            "steady_peak_allocated_bytes": max(
                (r["peak_allocated_bytes"] for r in steady), default=None
            ),
            "steady_peak_reserved_bytes": max(
                (r["peak_reserved_bytes"] for r in steady), default=None
            ),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "packages": {
                name: importlib.metadata.version(name)
                for name in (
                    "comfy-kitchen",
                    "triton-windows",
                    "diffusers",
                    "transformers",
                    "accelerate",
                )
            },
            "phase_timing": args.phase_timing,
            "compiled": process_config["model"].get("compile", False),
            "compile_note": "Inspect console.log TORCH_LOGS by step; steady recompiles invalidate compiled throughput",
        }
        write_json(args.output / "steps.json", rows)
        write_json(args.output / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BASELINE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid", default=GPU_UUID)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="Alternate AB then BA, each arm preserves all baseline steps",
    )
    parser.add_argument(
        "--compiled",
        action="store_true",
        help="Separate full-transformer compile experiment, not mixed with eager",
    )
    parser.add_argument(
        "--phase-timing",
        action="store_true",
        help="Separate instrumented run; CUDA events around predict_noise and backward",
    )
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.worker:
        run_worker(args)
    else:
        run_pair(args)


if __name__ == "__main__":
    main()
