# Phase 1 Real-Data Integration Tests

Tests that prove weight noising and gradient noising work inside actual short Krea 2 training runs.

## Prerequisites

- CUDA GPU with >= 24 GB VRAM (tested on RTX PRO 6000 Blackwell 96 GB)
- Krea-2-Raw model cached locally (`krea/Krea-2-Raw`)
- Dataset with images + captions under `datasets/` or path set via `AI_TOOLKIT_TEST_DATASET`

## Fast unit tests (no GPU needed)

```bash
pytest testing/test_perceptual_noising.py testing/test_perceptual_noising_qa.py -q
```

## Strict real-data integration tests (Layer A)

```powershell
$env:AI_TOOLKIT_RUN_KREA_INTEGRATION="1"
$env:AI_TOOLKIT_TEST_DATASET="<repo>\datasets"
pytest testing/integration/test_perceptual_noising_real_data.py -m "integration and gpu" -v -s
```

## Operational smoke test (Layer B)

```powershell
$env:AI_TOOLKIT_RUN_KREA_INTEGRATION="1"
$env:AI_TOOLKIT_TEST_DATASET="<repo>\datasets"
$env:AI_TOOLKIT_NOISING_TEST_STEPS="50"
pytest testing/integration/test_perceptual_noising_real_data.py::test_matched_smoke_comparison -m "slow and gpu" -v -s
```

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `AI_TOOLKIT_RUN_KREA_INTEGRATION` | `0` | Must be `1` to run integration tests |
| `AI_TOOLKIT_TEST_DATASET` | repo `datasets/` | Dataset root with images + captions |
| `AI_TOOLKIT_KREA_MODEL` | `krea/Krea-2-Raw` | HuggingFace model ID |
| `AI_TOOLKIT_TEST_OUTPUT` | `test_outputs/phase1_real_data` | Output directory for artifacts |
| `AI_TOOLKIT_NOISING_TEST_STEPS` | `25` | Step count for smoke test |
| `AI_TOOLKIT_KEEP_TEST_OUTPUTS` | `0` | Set `1` to preserve large artifacts |

## Output Artifacts

Each run writes a JSON report under `test_outputs/phase1_real_data/<run_name>/report.json`.
