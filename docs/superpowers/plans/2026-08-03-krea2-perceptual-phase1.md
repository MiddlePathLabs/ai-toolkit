# Krea 2 Perceptual Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add tested LoRA weight and gradient noising with safe Python/UI configuration, and run the temporary Krea 2 decode-gradient probe that gates future depth work.

**Architecture:** Keep all training behavior in `extensions_built_in/sd_trainer/SDTrainer.py`, with `_is_lora` parameter tags supplied by both LoRA construction paths. Keep persisted settings nested under `TrainConfig`, mirror them as optional TypeScript fields, migrate old jobs without touching future `loss_split` semantics, and render controls in the existing model-agnostic Advanced card.

**Tech Stack:** Python 3.12 embedded runtime, PyTorch 2.13 CUDA, pytest, TypeScript, React/Next.js, SQLite-backed job configuration.

## Global Constraints

- Weight-noise UI default is `sigma: 0.00125` and `log_every: 50`.
- Gradient-noise UI default is `mode: neelakantan`, `sigma: 0.001`, `eta: 0.01`, `gamma: 0.55`, and `log_every: 50`.
- Both injectors are disabled by default and affect only parameters tagged `_is_lora`.
- Gradient noise runs after gradient clipping and before `optimizer.step()`.
- Weight noise runs after `ema.update()` and after `optimizer.zero_grad(set_to_none=True)`.
- `bound_norm` rescales each noised tensor to its pre-noise norm.
- A metric cadence of `0` emits no noising metrics.
- The Phase 0 probe must use a real Krea decode and `torch.autograd.grad(decoded_pixels.mean(), noise_pred)`; static inspection cannot satisfy the gate.
- Depth, `rose`, loss splitting, caching, masking, and remaining perceptors are outside this branch.

---

## File Map

- Create: `docs/superpowers/specs/2026-08-03-krea2-perceptual-phase1-design.md` — approved phase design.
- Create: `testing/test_perceptual_noising.py` — focused config, tagging, injector, and metric regression tests.
- Modify: `toolkit/config_modules.py` — `WeightNoiseConfig`, `GradientNoiseConfig`, and `TrainConfig` fields.
- Modify: `toolkit/lora_special.py` — module-level and network-level `_is_lora` tags.
- Modify: `extensions_built_in/sd_trainer/SDTrainer.py` — injector methods, loop hooks, metric flush, and temporary Phase 0 probe.
- Modify: `ui/src/types.ts` — optional noising interfaces and `TrainConfig` fields.
- Modify: `ui/src/app/jobs/new/jobConfig.ts` — complete defaults and old-job migration.
- Modify: `ui/src/app/jobs/new/SimpleJob.tsx` — Advanced-card controls.
- Modify: `ui/src/docs.tsx` — noising help text.

### Task 1: Establish the failing noising contract

**Files:**
- Create: `testing/test_perceptual_noising.py`

**Interfaces:**
- Consumes: `TrainConfig`, `LoRAModule`, and the `SDTrainer` injector method names listed below.
- Produces: tests that define the expected defaults, `_is_lora` filtering, absolute/relative/Neelakantan behavior, norm bounding, and metric cadence.

- [ ] **Step 1: Write the focused tests before implementation.**

Use deterministic `torch.manual_seed(7)` fixtures and a lightweight trainer
instance created with `SDTrainer.__new__(SDTrainer)`, then assign only
`params`, `train_config`, `step_num`, and the metric attributes needed by each
test. Cover these exact cases:

```python
def test_gradient_noise_only_changes_tagged_gradients():
    tagged = torch.nn.Parameter(torch.ones(8))
    tagged._is_lora = True
    untagged = torch.nn.Parameter(torch.ones(8))
    tagged.grad = torch.ones_like(tagged)
    untagged.grad = torch.ones_like(untagged)
    trainer.params = [tagged, untagged]
    trainer.train_config = TrainConfig(
        gradient_noise={"enabled": True, "mode": "absolute", "sigma": 0.5, "log_every": 1}
    )
    trainer.step_num = 0
    before_untagged = untagged.grad.clone()
    trainer._inject_gradient_noise()
    assert not torch.equal(tagged.grad, torch.ones_like(tagged))
    assert torch.equal(untagged.grad, before_untagged)
    assert trainer._last_grad_noise_norm > 0
```

Add equivalent assertions for relative and Neelakantan sigma selection,
weight-noise operation after gradients are absent, untagged parameter
stability, `bound_norm=True`, `log_every=0`, and all `TrainConfig()` defaults.
Use `LoRAModule` with a small `torch.nn.Linear` to assert module-level tags;
use a small fake adapter module and the network registration path to assert
network-level tags.

- [ ] **Step 2: Run the focused test and confirm it fails for missing config, tags, and injector methods.**

Run:

```powershell
& E:\ai-toolkit\python_embeded\python.exe -m pytest testing/test_perceptual_noising.py -q
```

Expected: collection or assertion failures because the target checkout does
not yet define the new configuration, tags, and injector methods.

### Task 2: Add backend configuration and complete LoRA tagging

**Files:**
- Modify: `toolkit/config_modules.py` near `TrainConfig`.
- Modify: `toolkit/lora_special.py:120-140` and the network constructor after adapter lists are assembled.

**Interfaces:**
- Consumes: existing `TrainConfig`, `LoRAModule`, and adapter parameter lists.
- Produces: `TrainConfig.weight_noise`, `TrainConfig.gradient_noise`, and `_is_lora=True` on every relevant adapter parameter.

- [ ] **Step 1: Add `WeightNoiseConfig` and `GradientNoiseConfig`.**

Implement the documented defaults exactly:

```python
class WeightNoiseConfig:
    def __init__(self, **kwargs):
        self.enabled = bool(kwargs.get("enabled", False))
        self.mode = str(kwargs.get("mode", "relative"))
        self.sigma = float(kwargs.get("sigma", 0.00125))
        self.bound_norm = bool(kwargs.get("bound_norm", False))
        self.log_every = int(kwargs.get("log_every", 50))


class GradientNoiseConfig:
    def __init__(self, **kwargs):
        self.enabled = bool(kwargs.get("enabled", False))
        self.mode = str(kwargs.get("mode", "neelakantan"))
        self.sigma = float(kwargs.get("sigma", 0.001))
        self.eta = float(kwargs.get("eta", 0.01))
        self.gamma = float(kwargs.get("gamma", 0.55))
        self.log_every = int(kwargs.get("log_every", 50))
```

Wire them into `TrainConfig.__init__` with `(kwargs.get(name, {}) or {})`
so omitted legacy jobs remain disabled.

- [ ] **Step 2: Tag module-level LoRA parameters.**

Immediately after LoRA projection initialization, tag
`lora_down.weight`, `lora_up.weight` when present, and `lora_up.bias` when
present. Preserve the full-rank `IdentityModule` path by checking attributes
before assigning.

- [ ] **Step 3: Tag all registered adapter parameters at network level.**

After `self.text_encoder_loras` and `self.unet_loras` are assembled, iterate
both lists and set `_is_lora=True` on every parameter returned by
`adapter.parameters()`. This covers adapter implementations that do not pass
through `LoRAModule`.

- [ ] **Step 4: Run the focused tests.**

Run the same pytest command from Task 1. The default/tagging tests should pass;
injector tests remain failing until Task 3.

- [ ] **Step 5: Commit the backend contract.**

```powershell
git add toolkit/config_modules.py toolkit/lora_special.py testing/test_perceptual_noising.py
git commit -m "feat: add perceptual noising config and LoRA tags"
```

### Task 3: Implement injector methods, loop ordering, metrics, and probe

**Files:**
- Modify: `extensions_built_in/sd_trainer/SDTrainer.py` near the training-loop helpers and `hook_train_loop`.

**Interfaces:**
- Consumes: `TrainConfig.weight_noise`, `TrainConfig.gradient_noise`, tagged optimizer parameters, and existing `self.params`, `self.optimizer`, `self.step_num`, and `self.ema`.
- Produces: `_iter_lora_params_with_grad()`, `_inject_gradient_noise()`, `_record_fisher_trace()`, `_inject_weight_noise()`, and loss-dict metrics.

- [ ] **Step 1: Add `_iter_lora_params_with_grad`.**

Support both optimizer parameter-list shapes already accepted by the trainer:
list of parameter-group dictionaries and a flat parameter iterable. Yield only
parameters with `_is_lora=True` and non-`None` gradients.

- [ ] **Step 2: Add gradient injection.**

For each yielded parameter, calculate sigma from the selected mode, add
`torch.randn_like(grad) * sigma` in place, and accumulate gradient/noise norms
only when `log_every > 0` and `step_num % log_every == 0`. Emit
`_last_grad_noise_norm` and `_last_grad_noise_snr`; disabled or non-positive
sigma is a no-op.

- [ ] **Step 3: Add Fisher trace and weight injection.**

Sum `optimizer.state[p]["exp_avg_sq"]` only for tagged parameters when that
state exists. For weight noise, iterate every tagged parameter even when its
gradient is `None`, calculate absolute or RMS-relative sigma, add noise to
`p.data`, and when `bound_norm` is enabled rescale to the pre-noise norm.
Emit `_last_weight_noise_norm` and `_last_weight_norm` at cadence.

- [ ] **Step 4: Wire exact training-loop ordering.**

In the existing `if not self.is_grad_accumulation_step` block, call gradient
noise immediately after the two gradient-clipping branches. Call Fisher trace
and weight noise after the optional EMA update, while retaining
`optimizer.zero_grad(set_to_none=True)` before weight noise.

- [ ] **Step 5: Flush transient metrics into `loss_dict`.**

After constructing the existing `OrderedDict`, copy each present noising/Fisher
metric into the dictionary and reset it to `None` so stale values cannot leak to
the next training step. Preserve all existing loss and gradient metrics.

- [ ] **Step 6: Run the Phase 0 probe before removing it.**

Temporarily add the guide's Krea-only guard inside `calculate_loss` after
`noise_pred` exists. Use the scheduler's configured timestep count, call
`self.sd.decode_latents(noisy_latents - t * noise_pred)`, then assert
`torch.autograd.grad(decoded_pixels.float().mean(), noise_pred, retain_graph=True,
allow_unused=True)` is finite and non-zero and the result is `(B, 3, H, W)`.
Run one real Krea2 job with `low_vram=false`, batch 1, and the target dtype;
record model preset, resolution, dtype, quantization, allocated peak, and
reserved peak. Remove the temporary block whether the probe passes or is
blocked by the environment, and report the evidence accurately.

- [ ] **Step 7: Run the focused tests and commit.**

```powershell
& E:\ai-toolkit\python_embeded\python.exe -m pytest testing/test_perceptual_noising.py -q
git diff --check
git add extensions_built_in/sd_trainer/SDTrainer.py
git commit -m "feat: add LoRA weight and gradient noising"
```

### Task 4: Add TypeScript types, defaults, and migration

**Files:**
- Modify: `ui/src/types.ts` near `TrainConfig`.
- Modify: `ui/src/app/jobs/new/jobConfig.ts` defaults and `migrateJobConfig`.

**Interfaces:**
- Consumes: Python config paths and existing `JobConfig` migration behavior.
- Produces: optional `WeightNoiseConfig` and `GradientNoiseConfig` TypeScript interfaces, complete default job values, and backward-compatible migration.

- [ ] **Step 1: Add optional typed interfaces and fields.**

Use literal unions for the mode fields and add optional fields to `TrainConfig`
so old saved job objects remain type-safe.

- [ ] **Step 2: Add complete defaults to `defaultJobConfig`.**

Place `weight_noise` and `gradient_noise` inside
`config.process[0].train` using the exact backend defaults.

- [ ] **Step 3: Add migration clauses.**

If either object is absent, add the complete disabled object. Use nullish field
merging only if the migration needs to fill a partial nested object; preserve
existing saved values. Do not add `loss_split`.

- [ ] **Step 4: Run TypeScript compilation.**

```powershell
npm run build
```

### Task 5: Add Advanced-card controls and help text

**Files:**
- Modify: `ui/src/app/jobs/new/SimpleJob.tsx` after the existing Differential Guidance controls in the Advanced card.
- Modify: `ui/src/docs.tsx` with the matching `docKey` entries.

**Interfaces:**
- Consumes: typed/defaulted `jobConfig.config.process[0].train` noising objects.
- Produces: controls for every Phase 1 setting and user-facing semantics.

- [ ] **Step 1: Add the Weight Noising group.**

Add enable, mode, sigma, bound-norm, and cadence controls. Reveal the four
secondary controls only when enabled; use `setJobConfig` with the exact nested
paths and the defaults when the existing value is nullish.

- [ ] **Step 2: Add the Gradient Noising group.**

Add enable, mode, sigma, eta, gamma, and cadence controls. Show sigma for
absolute/relative modes and eta/gamma for Neelakantan mode. Keep all settings
under `config.process[0].train.gradient_noise.*`.

- [ ] **Step 3: Add docs entries.**

Document relative versus absolute scale, zero-initialized LoRA-up behavior,
norm bounding, metric cadence, and Neelakantan annealing. Keep the help text
clear that these affect LoRA adapter parameters only.

- [ ] **Step 4: Build the UI.**

```powershell
npm run build
```

### Task 6: Final verification and handoff

**Files:**
- No additional source files; update the plan checkboxes only if this plan is tracked during execution.

- [ ] **Step 1: Run focused Python verification.**

```powershell
& E:\ai-toolkit\python_embeded\python.exe -m pytest testing/test_perceptual_noising.py -q
& E:\ai-toolkit\python_embeded\python.exe -m pytest testing/test_lora_compile_scalars.py -q
```

- [ ] **Step 2: Run UI and repository checks.**

```powershell
Push-Location ui
npm run build
Pop-Location
git diff --check
git status --short --branch
```

- [ ] **Step 3: Review the branch diff.**

Confirm the Phase 0 probe is absent from production code, no depth/rose/Phase
3 files were added, migration does not write `loss_split`, and the branch has
separate commits for the backend noising work and UI work if the changes were
implemented as planned.

- [ ] **Step 4: Report exact evidence.**

Report the branch path, commit hashes, focused test counts, UI build result,
known warnings, and whether the real Krea Phase 0 probe passed or could not be
run in the available environment. Do not claim Phase 2 readiness from static
inspection or unit tests alone.

