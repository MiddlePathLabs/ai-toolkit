# Real-Data Integration Tests

Tests that prove weight noising / gradient noising (Phase 1) and the depth-consistency anchor (Phase 2) work inside actual short Krea 2 training runs.

## Prerequisites

- CUDA GPU with >= 24 GB VRAM (tested on RTX PRO 6000 Blackwell 102 GB)
- Krea-2-Raw + Depth-Anything-V2 models cached locally
- Dataset with images + captions under `datasets/` or path set via `AI_TOOLKIT_TEST_DATASET`
- **Runtime:** run pytest under `E:\ai-toolkit\python_embeded\python.exe` (has all deps; the system Python lacks `diffusers`). Set `$env:PYTHONPATH="."`.

## Fast unit tests (no GPU needed)

```bash
& "E:\ai-toolkit\python_embeded\python.exe" -m pytest testing/test_perceptual_noising.py testing/test_depth_config.py testing/test_loss_split.py -q
```

## Phase 1: noising integration tests

```powershell
$env:AI_TOOLKIT_RUN_KREA_INTEGRATION="1"
$env:AI_TOOLKIT_TEST_DATASET="E:\ai-toolkit\AI-Toolkit\datasets"
& "E:\ai-toolkit\python_embeded\python.exe" -m pytest testing/integration/test_perceptual_noising_real_data.py -m "integration and gpu" -v -s
```

## Phase 2: depth-anchor integration tests

Each test loads the full model stack; run them process-isolated (the transformers
framework retains refs that `gc.collect()` cannot break in one pytest process, so
a single-process batch OOMs after ~2 tests). The runner gives each test its own
process and aggregates results:

```powershell
# Canonical: full suite, process-isolated
powershell -ExecutionPolicy Bypass -File testing\integration\run_phase2_depth_tests.ps1

# Or one test in isolation (each is a fresh process):
$env:PYTHONPATH="."; $env:AI_TOOLKIT_RUN_KREA_DEPTH_INTEGRATION="1"
$env:AI_TOOLKIT_TEST_DATASET="E:\ai-toolkit\AI-Toolkit\datasets"
& "E:\ai-toolkit\python_embeded\python.exe" -m pytest `
    testing/integration/test_depth_consistency_real_data.py::test_strict_depth_only_lora_update `
    -m "depth" -v
```

The critical gate is `test_strict_depth_only_lora_update`: it proves a depth-only
loss (diffusion zeroed via dataset `loss_multiplier: 0.0`) moves a saved and
reloaded LoRA parameter.

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `AI_TOOLKIT_RUN_KREA_INTEGRATION` | `0` | Must be `1` to run Phase 1 tests |
| `AI_TOOLKIT_RUN_KREA_DEPTH_INTEGRATION` | `0` | Must be `1` to run Phase 2 tests |
| `AI_TOOLKIT_RUN_DEPTH_EXTENDED_MATRIX` | `0` | Must be `1` for the Layer D extended matrix |
| `AI_TOOLKIT_TEST_DATASET` | repo `datasets/` | Dataset root with images + captions |
| `AI_TOOLKIT_KREA_MODEL` | `krea/Krea-2-Raw` | HuggingFace model ID |
| `AI_TOOLKIT_DEPTH_MODEL` | `depth-anything/Depth-Anything-V2-Small-hf` | DA2 perceptor (shipped default is Small@518) |
| `AI_TOOLKIT_DEPTH_INPUT_SIZE` | `518` | DA2 input size |
| `AI_TOOLKIT_TEST_OUTPUT` | `test_outputs/phase1_real_data` | Phase 1 output dir |
| `AI_TOOLKIT_DEPTH_TEST_OUTPUT` | `test_outputs/phase2_real_data` | Phase 2 output dir |
| `AI_TOOLKIT_NOISING_TEST_STEPS` | `25` | Phase 1 smoke step count |
| `AI_TOOLKIT_DEPTH_TEST_STEPS` | `12` | Phase 2 smoke step count |
| `AI_TOOLKIT_DEPTH_TEST_MAX_IMAGES` | `8` | Images staged for depth tests |
| `AI_TOOLKIT_KEEP_TEST_OUTPUTS` | `0` | Set `1` to preserve large artifacts |

## Output Artifacts

Each run writes a JSON report under `test_outputs/phase{1,2}_real_data/<run_name>/report.json` capturing losses, hook counts, peak VRAM, runtime, and any failure detail.
