# Krea 2 Perceptual Phase 1 Design

## Goal

Add the model-agnostic weight-noising and gradient-noising features from the
Krea 2 perceptual plan, with safe defaults, LoRA-only targeting, observable
metrics, persisted UI configuration, and focused regression tests. Run the
temporary Phase 0 Krea decode-gradient probe before the Phase 1 implementation
is treated as ready for later depth work.

## Scope

This branch covers:

- The temporary Krea 2 gradient-flow probe described by the implementation
  guide. It must prove a finite, non-zero gradient from decoded pixels back to
  `noise_pred`, record the model/runtime settings and peak memory, and then be
  removed from production code.
- Weight noising in `SDTrainer`, applied after the optimizer step and after the
  optional EMA update.
- Gradient noising in `SDTrainer`, applied after gradient clipping and before
  the optimizer step.
- `_is_lora` tagging at both the LoRA module and network levels so all adapter
  parameter types are covered.
- Python configuration defaults and metric cadence.
- TypeScript defaults, migration for existing SQLite jobs, Advanced-card
  controls, and documentation entries.
- Focused Python tests for targeting, modes, norm bounding, metrics, and
  default configuration, plus a UI production build.

The depth anchor, `rose` optimizer, loss splitting, depth caching, dataloader
fields, subject masking, and additional perceptors remain out of scope. They
retain the phase gates in `PERCEPTUAL_KREA2_IMPLEMENTATION_GUIDE.md`.

## Design

### Configuration

`TrainConfig` owns two nested configurations:

```python
WeightNoiseConfig(
    enabled=False,
    mode="relative",
    sigma=0.00125,
    bound_norm=False,
    log_every=50,
)

GradientNoiseConfig(
    enabled=False,
    mode="neelakantan",
    sigma=0.001,
    eta=0.01,
    gamma=0.55,
    log_every=50,
)
```

The backend accepts the nested objects when absent, so existing YAML jobs
remain disabled and receive the documented defaults. The UI default job
contains complete objects, while migration fills each missing object without
changing unrelated saved values.

Weight modes are `absolute` and `relative`; relative noise scales the
configured sigma by each parameter's RMS. Gradient modes are `absolute`,
`relative`, and `neelakantan`; the latter uses `eta / (1 + step) ** gamma`.
Unknown modes are treated as invalid configuration in the new test contract
and must not silently affect training.

### Parameter targeting

Trainable LoRA parameters receive `_is_lora = True` during `LoRAModule`
construction. The network-level traversal marks compatible adapter parameters
as a defense-in-depth measure for LoKr, LoHa, and related adapter variants.
The injectors filter exclusively on this marker. Untagged trainable parameters
must remain unchanged.

The gradient injector iterates only tagged parameters with populated gradients.
The weight injector iterates all tagged parameters because it runs after
`optimizer.zero_grad(set_to_none=True)` and therefore cannot depend on
`p.grad`.

### Training-loop ordering

The existing `SDTrainer.hook_train_loop` remains the single integration point
for both `DiffusionTrainer` and `UITrainer`:

```text
clip_grad_norm_
  -> inject gradient noise
  -> optimizer.step
  -> optimizer.zero_grad
  -> optional EMA update
  -> record Fisher trace
  -> inject weight noise
```

All operations remain inside the non-gradient-accumulation-step branch. Weight
noise is deliberately after EMA so the EMA shadow follows the clean optimizer
trajectory rather than the perturbed live weights.

### Metrics

At the configured cadence, the trainer records:

- `_last_grad_noise_norm`
- `_last_grad_noise_snr`
- `_last_weight_noise_norm`
- `_last_weight_norm`
- `_last_fisher_trace` when the optimizer exposes an `exp_avg_sq` state

The loss dictionary exposes values that are present. A cadence of `0` disables
metric emission while preserving the noise behavior.

### UI and migration

Both feature groups live in the existing model-agnostic `Advanced` card. Each
group has an enable checkbox and reveals only the controls relevant to the
selected mode. All controls use the existing `setJobConfig` path helper.

The migration adds missing nested noising objects to old jobs loaded from
SQLite. It must not add or rewrite `train.loss_split`; that future setting's
absent value has meaningful Auto semantics.

Each non-obvious control receives a `ui/src/docs.tsx` entry explaining the
noise scale, relative-mode zero-initialized LoRA-up behavior, norm bounding,
metric cadence, and the distinction between gradient and weight noise.

### Phase 0 probe

Before depth implementation, run one representative Krea 2 training sample
with `low_vram: false`, batch size 1, and the actual target dtype/preset. The
temporary probe calls `self.sd.decode_latents(noisy_latents - t * noise_pred)`
and uses `torch.autograd.grad(decoded_pixels.float().mean(), noise_pred)`.
It fails unless the decoded output is `(B, 3, H, W)` and the gradient is
finite and non-zero. It records model preset, resolution, dtype,
quantization, batch size, allocated peak, and reserved peak, then is removed
from the source before the Phase 1 commit.

## Validation

The focused Python test will verify:

1. Gradient noise changes only tagged populated gradients.
2. Weight noise changes only tagged parameters even when gradients are absent.
3. Absolute and relative modes use the documented scales.
4. `bound_norm` restores each parameter's pre-noise norm.
5. Metrics appear at a positive cadence and disappear at `log_every=0`.
6. `TrainConfig()` keeps both injectors disabled and uses the documented
   defaults.

Verification uses the embedded target Python runtime and includes the existing
LoRA regression test, the new focused test, `git diff --check`, and
`ui/npm run build`. The Phase 0 evidence is recorded separately and is not
claimed from static inspection.

