# LoRA Identity Isolation — implementation and validation report

## Status and scope

Implemented opt-in backend and UI integration for declared Krea 2 / Krea-2-Raw LoRA training, within the restrictions below. **511 focused behavioral/regression tests passed.** Synthetic models/encoders and generated color tiles establish control flow, gradients, caching, optimizer updates, and checkpoint mechanics—not full Krea model correctness or identity-isolation effectiveness.

No real-data training, real-data inference comparisons, ArcFace/SFace scoring, or A/B experiments were executed. No portrait collection or model weights were downloaded. The production pinned portrait preparation path and full Krea/GPU training remain for the user. CUDA is available locally; the existing synthetic CUDA continuation regression ran successfully. This is not a claim that hardware was unavailable or that a real Krea training run was validated.

## Reference and provenance

- Initial AI-Toolkit HEAD: `13cbb75f5a2402fe3a178cc382fc7734654bd37f`; working tree was clean before edits.
- Fizgig inspected HEAD: `d63ca56dd307ccda820a0304c54473c9ba7ce29b`, described as `v8.4.0-5-gd63ca56d-dirty`.
- Expected tag `v8.4.0`: `e2f7ccfe33b754910a11981978a6ee1658f18923`.
- Read-only comparison against the tag enumerated only unrelated tracked differences; the isolation implementation files have no differences against that tag. Existing reference-local untracked documentation and removed output `.gitkeep` were preserved. Fizgig was never modified.
- Detailed ten-mechanism mapping, defaults, conflicts, tokenizer evidence, and adaptations: [reference audit](identity_isolation_reference.md).
- Apache-2.0 source attribution/license retained in [NOTICE](licenses/Fizgig-NOTICE.txt) and [license text](licenses/Fizgig-Apache-2.0.txt). Stranger assets are separately CC BY 4.0; their pinned dataset card was inspected without fetching portraits.

## Files changed

| Area | Files | Responsibility |
|---|---|---|
| Pure configuration/captions | `toolkit/identity_isolation.py` | Explicit preset/component resolution, mapping validation, boundary-aware fusion and identity-free preservation captions. |
| Assets/compositor | `toolkit/identity_isolation_composites.py` | Explicit pinned download command; offline hashes/provenance checks; faithful panel RNG/caption ordering; atomic content-addressed output caches. |
| Dataset preparation/reporting | `toolkit/identity_isolation_data.py` | Static settings resolution, source binding, generated datasets, approximate repeats, content recipe identity, actual DTO caption inspection and exposure reporting. |
| Training guards/helpers | `toolkit/identity_isolation_training.py` | Shared declared/runtime guards, successful-update DOP cadence, homogeneous category checks and preservation MSE. |
| Existing training integration | `jobs/process/BaseSDTrainProcess.py`, `extensions_built_in/sd_trainer/SDTrainer.py` | Extend existing DOP, preview conditioning, existing optimizer LR windows, startup/per-window instrumentation, and existing raw resume state. |
| Existing data/config/admission integration | `toolkit/config_modules.py`, `toolkit/dataloader_mixins.py`, `toolkit/data_loader.py`, `toolkit/admission.py` | Opt-in metadata; caption/dropout/cache lifecycle; category partitioning; stateful sampling; stdlib-only static admission. |
| UI | `ui/src/types.ts`, `ui/src/app/jobs/new/SimpleJob.tsx`, `ui/src/components/IdentityIsolationPanel.tsx`, `ui/src/app/api/identity-isolation/route.ts` | Optional settings, presets/overrides, subjects, backend caption/exposure inspection, local preparation/thumbnail display, shell-free Python bridge and actionable errors. |
| Local inspection command | `ui_scripts/identity_isolation.py` | Caption/inspect/prepare requests; never starts training or downloads assets. |
| Behavioral tests | `testing/test_identity_isolation_captions.py`, `testing/test_identity_isolation_composites.py`, `testing/test_identity_isolation_data.py`, `testing/test_identity_isolation_training.py`, `testing/test_identity_isolation_conditioning.py`, `testing/test_identity_isolation_admission.py`, `testing/test_identity_isolation_resume.py` | Synthetic/offline caption, asset/cache, data, gradients, optimizer, admission and checkpoint regressions. |
| Example/handoff | `config/examples/train_lora_krea2_identity_isolation.yaml`, `docs/identity_isolation.md`, `docs/identity_isolation_reference.md`, `docs/identity_isolation_experiments.md`, this report, `docs/licenses/Fizgig-Apache-2.0.txt`, `docs/licenses/Fizgig-NOTICE.txt`, `README.md`, `lora-identity-isolation.md` | Usage, attribution, source mapping, manual checklist, unexecuted experiment protocol and completion record. |

No optimizer implementation was globally changed. No ordinary-training cache namespace was version-bumped merely because Isolation exists.

## Verified behavior

- Raw/exact-phrase/already-fused captions; repeated identities; same/different classes; overlapping names; substring boundaries; ambiguous/cascading mapping rejection; transformation order and idempotence.
- All configured subject mappings apply across mapped and additional datasets. Own subject triggers do not overwrite other mapped subjects; unrelated normal datasets retain their legacy global trigger default; composites never inherit it.
- Live and disk-cached training/preservation embeddings agree using the actual text-cache producer/consumer, including caption dropout. Source caption edits refresh conditioning/cache keys through the existing lifecycle.
- Disabled preparation is mutation-, source-access-, and RNG-inert; ordinary/off cache keys match. Existing single-subject DOP behavior is retained outside opt-in mapping-aware behavior. This is focused parity evidence, not bitwise equivalence of full training runs.
- Composites are 1024×512 RGB, with two 512-square panels, reference RNG/placement/stranger-first captions and preserved originals. Content changes produce new identities; incomplete/corrupt outputs are rejected. Known byte/provenance overlap is rejected; distinct real identities are not certified.
- Actual `SDTrainer.get_prior_prediction` disables the synthetic trainable adapter under `no_grad`, preserves matched noisy input/timestep/conditioning/model kwargs, and restores model/adapter state on success/failure. The production preservation path includes the adapter and backpropagates to its trainable synthetic parameter; the frozen reference parameter has no gradient.
- DOP selection is constant over an accumulation window and uses persisted `completed_update_id`. Tests cover every-step/strided selection, sample-normalized accumulation, skipped updates and raw-checkpoint phase continuity.
- Actual AdamW and this fork's ROSE updates exercise temporary group-LR scaling, relative group ratios, restoration and scheduler progression. Unsupported concrete optimizers and mixed-category scaled updates are rejected rather than loss-weighted/averaged.
- Raw checkpoint tests restore optimizer/scheduler/sampler state and DOP phase, and reject an actual changed source-caption recipe. GPU/bucket numerical replay is not asserted.
- Existing focused manual-mask/full-frame DOP, depth loss, optimizer-runtime, ROSE, resume, H3 cache, caption admission, dataset-path and preview-state regressions passed.

## Verification actually run

### Final automated run

Packaged training interpreter: `E:/ai-toolkit/python_embeded/python.exe`. Set `HF_HUB_OFFLINE=1` and `NO_ALBUMENTATIONS_UPDATE=1` before importing the training stack. Ran pytest with `-q -p no:cacheprovider` over the seven new Isolation test modules listed above and:

- `testing/test_manual_mask_dop.py`
- `testing/test_optimizer_runtime.py`
- `testing/test_rose_optimizer.py`
- `testing/test_resume.py`
- `testing/test_depth_loss_gating.py`
- `testing/test_h3_cache_contracts.py`
- `testing/test_uncaptioned_guard.py`
- `testing/test_admission.py`
- `testing/test_python_admission.py`
- `testing/test_dataset_config_paths.py`
- `testing/test_depth_cache_identity.py`
- `testing/test_preview_eager_stance.py`
- `testing/test_preview_state_restoration.py`

**Observed result: 511 passed, 69 warnings, 69.33 seconds.** Warnings were PyTorch JIT deprecations, an existing mask-test tensor-to-scalar warning, and an optional Triton import warning. No full-repository test-suite claim is made.

### Static checks

- Ruff, isolated configuration, `E4,E7,E9,F,I`, over the four new backend modules, bridge and seven new test modules: **passed**. Scoped import ordering and Ruff formatting applied only to new Python files.
- `py_compile` over the new backend modules/bridge and modified admission/config/data/training modules: **passed**.
- `npm run typecheck` (worker and application TypeScript configurations): **passed**.
- Scoped Prettier check for the new panel and route: **passed**. Existing unrelated UI files were not wholesale reformatted.
- `npm run lint -- --no-cache`: **not available as a noninteractive check**. `next lint` is deprecated and the checkout has no ESLint configuration; it exited 1 with a setup prompt. No ESLint configuration/dependency was added solely to suppress that result.
- Programmatic canonical admission on the synthetic full configuration with the default Python interpreter: static-valid result, three deferred runtime checks, and **no Torch import**. Static admission is not runtime/model certification.

### Executable smoke scenarios

1. **Inspection CLI:** actual `ui_scripts/identity_isolation.py --request ...` transformed a repeated two-subject caption into fused training text and identity-free preservation text, without source/model access.
2. **Integrated synthetic preparation/training mechanics:** a throwaway script generated two subject color-tile datasets and a separately authenticated private synthetic stranger pin (never labeled as production assets). It ran actual configuration preparation, composites, repeat balancing, bucketed loading, disk VAE-latent caching using an average-pooling toy encoder, disk training/preservation text caching, existing teacher/student prediction helpers, AdamW runtime LR windows, scheduler restoration, sampler-state restore, and post-loader composite reuse. Eight synthetic update iterations observed alternating DOP selection and category-appropriate 1×/2× LRs, with frozen-reference gradients absent. This was a mechanics loop over one tiny loader epoch, not a full Krea trainer run or a configured compute-budget experiment.
3. **Locally cached Krea tokenizer:** both production tokenizer constructors used `local_files_only=True`. The actual instruction prefix encoded to 34 tokens, matching the production slice. `zwxemwoman` split into `z`, `wx`, `em`, `woman`; representative multi-subject/preservation splits are recorded in the reference audit.
4. **Live UI:** launched Next directly on loopback without the queue/training worker. Browser actions exercised presets/overrides, two subjects, repeated caption previews, four-subject limit, Off retaining legacy DOP/weight/optimizer/step settings, subjectless DOP-only preview, balance-only minimum subject guard, real API dataset/exposure tables, worker-count errors and missing-asset instructions. Screenshots verified the actual surface.
5. **Composite UI rendering:** four real synthetic compositor/bridge thumbnails were displayed through an explicitly intercepted synthetic test response. The live production route did not accept a synthetic pin. This verifies thumbnail rendering, not production-asset preparation.
6. **Final real API requests:** caption and dataset inspection returned 200; missing pinned assets returned 422 with explicit preparation instructions; invalid action returned 400. No training job was created or started.

Both standalone Next dev sessions needed a reload after an initially blank first navigation; the rendered form and API paths then worked. Root cause was not established, so this is recorded as a verification-environment limitation rather than claimed fixed or attributed to unrelated code. The dev server also emitted an optional macOS temperature-sensor module warning on Windows. Dedicated browser tabs/services were closed, and throwaway scripts/fixtures were removed.

### Defects found and corrected during validation

- Unconditional new window-hook dispatch broke disabled-path synthetic resume hosts: guarded the call; all 45 existing resume tests then passed, followed by the final combined run.
- Initial generated-cache allowlisting omitted `.aitk_size.json` written by the actual loader: reproduced post-loader cache reuse failure, allowed only known trainer metadata/cache locations, retained strict recipe-image/caption hashes, and confirmed post-loader reuse with both text and VAE caches.
- Review found unrelated normal datasets lost the process trigger default: narrowed suppression to composites and added an actual constructor regression.
- Inspection now uses the actual DTO caption path, preventing differences for uppercase already-fused captions and unrelated global-trigger inheritance. Balancing-only ablations activate explicitly rather than silently doing nothing.

## Compatibility restrictions and deviations

- Declared Krea2 Raw text-to-image LoRA only. Turbo/edit/control/fine-tune/other model-family support is not claimed. A local file's real model role cannot be certified from its filename; actual model loading remains manual validation.
- Actual torch AdamW or toolkit ROSE only in this exercised boundary. Fused-backward/unknown optimizers are rejected. Nonunit composite LR requires both accumulation controls1 and homogeneous updates; pooling is rejected in that mode, and nonbucketed batches must be size1.
- Active Isolation uses zero-worker loaders. Existing raw checkpoint continuation is retained; exact GPU/bucket replay is not promised. Use the original unprepared job configuration when resuming, not internally derived dataset blocks.
- Explicit source-only masks/controls cannot be silently transferred to generated sheets; incompatible dataset auxiliaries are rejected. Compatible global masking/perceptual objectives and instrumentation remain in the existing paths. Real auxiliary-model interactions on Krea remain unverified.
- Conflicting reference objectives, adapters/embeddings/decorators, unsupported guidance/mean-flow/text-encoder training, and reg-dataset paths are rejected rather than silently disabled.
- Person composites only; source-compatible selection reads immediate folder photos. Existing Toolkit JPEG/PNG/WebP source support is used. Multiple trained identities share one adapter through their own datasets/triggers; composites still contain only one trained identity plus a stranger.
- Explicit asset preparation and content-hashed cache integrity are strengthened extensions. No automatic asset/model download is required by ordinary training/tests.
- All-subject cross-folder fusion fixes a source limitation. Adjective fallback follows actual source behavior, not its overstated documentation. Fusion is a text operation, never a single-token guarantee.
- DOP preserves Toolkit's ordinary noisy state rather than Fizgig's independently re-noised extra pass. Reference/student within preservation remain matched. Successful-update-window striding differs from Fizgig's consumed-loader-batch counter, especially with accumulation/skips.
- Composite LR never averages categories as Fizgig can under accumulation; restricted homogeneous updates provide actual optimizer-LR scaling. Other explicitly enabled runtime scales may compose and must be disclosed in experiments.
- Reference balancing is optional/approximate; raw Toolkit repeats are multiplied by the balancing factor, generated blocks remain repeat1, and resolution/bucket effects are reported. No optimization-step increase is automatic.

## Manual checks awaiting the user

Follow [usage/manual checklist](identity_isolation.md) and [example Krea configuration](../config/examples/train_lora_krea2_identity_isolation.yaml):

1. Explicitly prepare the pinned portrait assets, verify provenance/license, or supply a complete verified local directory.
2. Substitute your own consented datasets/model/encoder/VAE locations; inspect every mapping and generated sheet.
3. Run the short real Krea2/Raw training smoke yourself. Verify real tokenizer/encoder/VAE/denoiser integration, caches, auxiliary losses, DOP gradients/cadence, category LR logs, previews, VRAM and runtime.
4. Save/resume the unchanged raw checkpoint; inspect phase, sampler/category progression and scheduled LRs. Distinguish continuation from weights-only warm start and from exact numerical replay.
5. Execute the [reproducible experiment plan](identity_isolation_experiments.md) yourself, including fixed-compute/exposure accounting, separate stride/LR ablations, two-identity balancing control, and ArcFace/SFace plus visual evaluation.

**Untested:** full Krea denoiser/text encoder/VAE execution, real production portrait download/preparation, actual AMP overflow/OOM behavior with a full Krea model, full perceptual/masking model interactions on real data, distributed training, unsupported optimizers/families, and empirical identity-isolation effectiveness. No improvement claim is made before your results.
