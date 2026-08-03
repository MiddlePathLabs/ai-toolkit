# Krea 2 Phase 1 Noising Acceptance

Date: 2026-08-03

## Result

Two matched 50-step Krea 2 Raw training jobs completed successfully. The
baseline kept noising disabled; the comparison enabled relative weight
noising at `sigma: 0.00125`. Both runs kept finite loss throughout, and every
recorded noised-run cadence emitted a positive `weight_noise_norm`.

| Setting | Baseline | Relative weight noise |
|---|---:|---:|
| Steps | 50 | 50 |
| Exit code | 0 | 0 |
| Parsed loss range | 0.02956–0.2940 | 0.03294–0.4427 |
| All parsed losses finite | Yes | Yes |
| `weight_noise_norm` | Not emitted | 0.02332–0.03148 |
| Final checkpoint saved | Yes | Yes |

The noised run emitted positive weight-noise norms at each configured cadence:
`0.02332`, `0.02568`, `0.02757`, `0.02934`, and `0.03148`. The progress logger
rendered the first value twice while refreshing the same step; that duplicate
is not a second optimizer cadence.

## Matched Runtime

| Field | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition |
| GPU memory | 95.59 GiB |
| Python | 3.12.10 embedded runtime |
| PyTorch | 2.13.0+cu130 |
| Model | `krea/Krea-2-Raw` |
| Architecture | `krea2` |
| Precision | BF16 |
| Quantization | `convrot8` transformer and text encoder |
| `low_vram` | `false` |
| Batch size | 1 |
| Dataset | 37 local images, 512/768/1024 buckets |
| Adapter | full-rank LoKr, 781,160,448 trainable parameters |
| Optimizer | `automagic2` |
| EMA | enabled, decay 0.99 |
| Seed | 42 |
| Sampling | disabled |

The two temporary CLI configurations were compared structurally before the
runs. They differed only in the run name and
`train.weight_noise.enabled`. Relative-mode untagged-parameter isolation is
covered separately by the focused test suite; the runtime job validates the
real Krea trainer path, finite optimization, EMA ordering, metrics, and final
save.

## Commands

```powershell
& E:\ai-toolkit\python_embeded\python.exe run.py phase1_krea_baseline.json -l output/phase1_krea_baseline.log
& E:\ai-toolkit\python_embeded\python.exe run.py phase1_krea_noised.json -l output/phase1_krea_noised.log
```

Both commands ran with `SEED=42` from the isolated Phase 1 validation
worktree. Generated checkpoints, optimizer states, temporary configs, and raw
logs were removed after this evidence was recorded.
