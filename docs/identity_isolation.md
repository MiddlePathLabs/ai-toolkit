# LoRA Identity Isolation

An opt-in port of Fizgig's fused conditioning and synthetic stranger-composite recipe, extending **this fork's existing DOP**. Primary intended model: Krea 2 / Krea-2-Raw LoRA training. **Improved identity isolation has not been established.** Synthetic/local tests verify mechanics, not real-data training quality.

Read the [source audit and deviations](identity_isolation_reference.md), [user-run experiment protocol](identity_isolation_experiments.md), and [implementation/validation report](identity_isolation_implementation_report.md) before training.

## Asset preparation (explicit, optional network operation)

Ordinary training and tests never fetch stranger assets. Full Isolation requires the exact collection `ShootTheSound/fizgig-multi-person-strangers` at revision `8aeb77502234781cb5dca524994c80efcaa51024`; another collection is not substituted.

Run from the repository with the Python interpreter used for training:

```text
python -m toolkit.identity_isolation_composites download-assets --asset-dir data/identity_isolation/strangers
```

This is an **explicit download command**, not part of an ordinary run. It fetches only the pinned portrait PNGs, `strangers.json`, and dataset card, writes the revision marker, and verifies pinned metadata/image hashes. It does not fetch a diffusion model or start training. Use a fresh destination; an existing incomplete/tampered directory is rejected rather than merged silently. Preserve a suspect directory for diagnosis and choose a fresh destination.

An existing local asset directory is supported through `stranger_asset_dir`. It must contain the pinned PNGs, `strangers.json`, `README.md`, and `fizgig_revision.txt` containing the revision above. A complete Fizgig operational cache may be usable; all bytes still must pass verification. Merely writing the revision marker cannot bless arbitrary files. The same offline verification runs before composite creation.

Assets: CC BY 4.0, ShootTheSound / Peter Neill, [pinned dataset card](https://huggingface.co/datasets/ShootTheSound/fizgig-multi-person-strangers/raw/8aeb77502234781cb5dca524994c80efcaa51024/README.md). They are 32 synthetic Qwen Image 2.1 portraits; provenance/licensing are separate from Fizgig's Apache-2.0 source. See [attribution](licenses/Fizgig-NOTICE.txt). Cropping, resizing, and pairing are adaptations. Confirm rights to your own photos separately.

Content hashes prevent byte-identical source/stranger overlap. Optional provenance declarations can reject known identity overlap; filenames do not prove distinct identity. Neither asset metadata nor these checks can certify that a stranger does not resemble a real trained person.

For declared overlap checks, a source or asset directory may also contain `identity_isolation_provenance.json`, for example `{"identity_ids":["consented-subject-one"],"provenance_ids":["my-source-collection"]}`. Matching declared IDs/provenance across source and strangers are rejected. These are user-supplied declarations, not face-recognition verification; missing declarations cannot establish distinct identities.

## Folder/caption layout

Choose actual folders yourself; symbolic locations below are not existing datasets:

```text
<subject-one-folder>/
  image01.png
  image01.txt      # zwxem, a woman smiling in a park
<subject-two-folder>/
  image01.jpg
  image01.txt      # qrxt, a man standing on a beach
<stranger-asset-directory>/
  README.md
  strangers.json
  fizgig_revision.txt
  stranger_00.png ... stranger_31.png
<separate-generated-directory>/
  <content-hash>/
    fizgig_mp_00.png
    fizgig_mp_00.txt
    ...
    identity_isolation_manifest.json
```

Each subject has a distinct folder, trigger, and explicit one-word class. Up to four subjects. Two subjects can share the same class. Original photos and caption files are never rewritten. Composites contain one trained subject plus one stranger—not two trained subjects.

Training sources use Toolkit's JPEG/PNG/WebP formats. For the source-compatible compositor, put eligible subject photos directly in each subject folder: its sorted source selection is nonrecursive, like Fizgig. Ordinary Toolkit dataset scanning may include nested images, but those are not selected for composites.

Supported caption forms include `zwxem, a woman`, `zwxem a woman`, standalone `zwxem`, and already-fused `zwxemwoman`. Exact phrase rules run before standalone-trigger replacement; all configured subjects are handled globally before text encoding. Literal word boundaries prevent substring replacement. Similar trigger prefixes are resolved longest-first; collisions/cascading aliases are rejected. Capitalization is matched case-insensitively. Fixed literal syntax is intentional: arbitrary punctuation/whitespace variants fall back to the bare-trigger rule rather than a speculative natural-language parser.

For adjectives, source-compatible behavior is `zwxem, a young blonde woman` → `zwxemwoman, a young blonde woman`, not deletion of the descriptive phrase. Fused words are **not single tokenizer tokens**; see the locally observed Krea tokenizer examples in the reference audit. Group-photo captions can name multiple identities regardless of which subject folder holds the photo.

## Presets and precedence

Process key: `identity_isolation`. Presets initialize explicit components; they do not hide an alternate training algorithm.

| Preset | Fused captions | Composites | DOP interval / weight | Composite LR |
|---|---|---|---|---|
| `off` | No | No | No new DOP settings | Inactive |
| `dop_only` | No | No | 1 / 1 | Inactive |
| `full` | Yes | 16 per subject | 2 / 2 | 2× |

Explicit component fields override preset defaults. `balance_subjects` defaults false, matching optional reference balancing. `seed` defaults to 42. Fusion-only and balancing-only ablations are supported with preset off plus the selected component and explicit subjects. **Off never disables separately configured legacy DOP.** Active contradictory legacy DOP and Isolation settings are errors; remove duplicate ownership or make the values agree. In the UI, an existing legacy DOP checkbox value of false is explicit: enable it and match its multiplier to the Isolation weight, or remove the duplicate legacy fields in advanced configuration. Selecting an Isolation preset does not silently change those older controls.

Isolation `seed` controls compositor randomness only; it does not reseed the existing training/sampling RNG. Use the process-level `training_seed` for the training run and the normal sample seed settings for previews. Checkpointed sampler/RNG state governs continuation.

### Trigger inheritance and verified asset paths

With subject mappings and DOP enabled, every nonempty effective dataset trigger must match a configured subject trigger. Validation includes the process-level `trigger_word` when a dataset's trigger is absent or `null`. An explicit dataset trigger overrides the process trigger; an empty string disables inheritance. Map the inherited trigger or override it on unrelated datasets before inspection or training.

Composite preparation resolves `stranger_asset_dir` once and uses that absolute directory for verification, rendering, and the post-render input check. A working-directory change during rendering cannot redirect portrait reads to an unverified relative directory.


## Example Krea 2 process

Substitute your actual model and folders. Do not use symbolic angle-bracket values unchanged. This is a process block to place under `config.process` in an ordinary `job: extension` configuration. Keep your normal save/sample/device configuration.

```yaml
type: diffusion_trainer
training_folder: output
device: cuda
training_seed: 42
network:
  type: lora
  linear: 16
  linear_alpha: 16
model:
  arch: krea2
  name_or_path: <your-Krea-2-Raw-checkpoint-or-model-directory>
  quantize: true
train:
  steps: 1000
  batch_size: 1
  gradient_accumulation: 1
  gradient_accumulation_steps: 1
  optimizer: adamw
  lr: 0.0001
  lr_scheduler: constant
  noise_scheduler: flowmatch
  dtype: bf16
  cache_text_embeddings: true
  pool_datasets: false
identity_isolation:
  preset: full
  subjects:
    - folder_path: <subject-one-folder>
      trigger: zwxem
      class_name: woman
    - folder_path: <subject-two-folder>
      trigger: qrxt
      class_name: man
  balance_subjects: false
  fused_captions: true
  composites: true
  composite_count: 16
  composite_lr_multiplier: 2.0
  dop_enabled: true
  dop_interval: 2
  dop_weight: 2.0
  stranger_asset_dir: <prepared-pinned-asset-directory>
  generated_dir: <separate-generated-directory>
  seed: 42
datasets:
  - folder_path: <subject-one-folder>
    caption_ext: txt
    caption_dropout_rate: 0.0
    cache_latents_to_disk: true
    resolution: [512, 1024]
    num_repeats: 1
    num_workers: 0
  - folder_path: <subject-two-folder>
    caption_ext: txt
    caption_dropout_rate: 0.0
    cache_latents_to_disk: true
    resolution: [512, 1024]
    num_repeats: 1
    num_workers: 0
sample:
  sampler: flowmatch
  sample_every: 100
  width: 1024
  height: 1024
  prompts:
    - "a man with round glasses on the left, and zwxem, a woman on the right, two people side by side"
```

The step count is illustrative, not a training recommendation. Adding subjects/composites never automatically increases it. The complete Full recipe uses the restrictions below. For caption-only/DOP ablations, turn components off explicitly rather than changing multiple variables unintentionally.

## Inspection without training

The local bridge reads one JSON request file (or stdin). It never starts training. A request contains your process dictionary, not the surrounding job:

```json
{"action":"caption","process":{"identity_isolation":{"preset":"off","fused_captions":true,"subjects":[{"folder_path":"<subject-one-folder>","trigger":"zwxem","class_name":"woman"}]}},"caption":"zwxem, a woman smiling in a park"}
```

```text
python ui_scripts/identity_isolation.py --request <your-request.json>
```

`caption` reports original/training/preservation text with no dataset access. `inspect` validates your configured source folders and reports caption/exposure previews without generating composites or downloading assets. `prepare` explicitly builds/verifies generated composites using already-prepared assets and returns representative composite thumbnails/captions. Inspect all generated sheets yourself, not only thumbnails. Preview prompt transformations use the same policy as training; finished LoRA inference prompts should use fused strings unless the inference client applies this transform.

### Composite face eligibility

Preparation checks every detection after the largest-area face, rejecting the photo if any rival reaches 60% of the primary face's visible height. Subject photos must also pass the frontal-pose gate.

Square crops must contain the entire clamped face box. Photos whose face cannot fit an in-bounds square are excluded from the usability pool with reason `face_crop`; preparation does not pad or publish clipped faces. The `face-square-frontal-v2` recipe uses a new cache identity, so caches built before these checks are not reused.

## Cache and exposure semantics

Generated composite directory names hash source image contents, captions, mappings, pinned assets, seed/count, and compositor/preprocessing version. Source/input changes produce a different generated dataset identity. Missing, unexpected, or changed recipe outputs at an existing identity are errors, not silent cache hits. Explicitly allowlisted Toolkit latent/text/perceptual cache directories and `.aitk_size.json` are excluded from recipe outputs so training does not invalidate its own composites. Do not edit generated sheets/captions in place; keep source changes in the source folder and generate a new identity.

Active text embedding keys include mapping/transform identity. Training and preservation conditioning (including caption dropout) share one policy. Ordinary disabled-feature cache keys are not invalidated. Existing VAE caches are reused through the normal loader, and generated content-addressed paths prevent old auxiliary targets from being accidentally reused under stable composite filenames.

Balancing uses normal dataset repeats, approximately matching the largest subject folder; composites stay repeat1. Reports distinguish originals, effective samples/loader slots, normal/composite ratios, per-subject probabilities, and expected exposures over the configured run. Bucket tail padding, batch size, repeats, and multiple resolutions can change these values. Pre-loader reports are estimates; sampler probability is not a promise of exact equality in a finite run.

`source_original_count` counts source files before repeats/resolution expansion; `generated_original_count` counts generated composites. `original_count` is their combined un-repeated total. The per-dataset rows distinguish categories, resolution variants, effective repeats, padded sample slots, sample and loader-draw probabilities, and estimated exposures. `expected_optimizer_updates` and `optimizer_updates_with_partial_flush` bound the non-skipped update count for legacy accumulation; `gradient_accumulation_steps` delays updates without adding input draws.

## DOP and optimizer logs

DOP uses the existing matched-input teacher/student preservation path, with the trainable identity adapter disabled only for the reference. The cadence clock is **successful optimizer updates**: window0 fires, then interval2 fires on windows2,4,...; all contributing microbatches share a decision. Overflow/empty/OOM skips do not advance the successful-update phase. This differs from Fizgig's consumed-batch clock. Striding is not mathematically identical to every-step preservation.

Raw preservation loss and weighted contribution are logged only for actual preservation passes, along with pass counts, interval, and successful-update clock. A skipped pass is not a measured zero loss. Composite optimizer updates temporarily scale scheduled parameter-group LRs (2× by default), then restore scheduled values before scheduler progression. This is not loss weighting. Log the category and effective group LRs; relative group LRs remain intact.

| Log/metric | Interpretation |
|---|---|
| `[identity-isolation exposure]` | Startup JSON report using actual prepared loader slots; exposures remain estimates. |
| `[identity-isolation DOP]` | Effective interval/weight and successful-update counter semantics. |
| `isolation/update_id`, `isolation/dop_interval`, `isolation/dop_selected` | Window clock, effective cadence, and selected-window decision. |
| `isolation/preservation_raw`, `isolation/preservation_weighted` | Sample-weighted means of measured MSE and its weighted contribution; absent on skipped windows. |
| `isolation/preservation_pass_count` | Actual preservation forward passes in this update window, not a lifetime counter. |
| `isolation/category_composite`, `isolation/category_mixed` | Category flags: both zero means normal; mixed is only permitted with unit composite LR. |
| `isolation/lr_multiplier`, `isolation/effective_lr_group_N` | Composite factor and effective scheduled LR for each parameter group; other explicitly enabled runtime scales may compose. |


## Supported boundary and conflict handling

- Intended model path is Krea2 declared Raw LoRA training, not Turbo, edit/control, slider, fine-tune, text-encoder training, or another unvalidated family.
- Dataset `num_workers: 0` is required for the exercised sampler-cursor resume boundary. Worker-prefetched sampling is not claimed. Use the original unprepared job configuration when resuming; derived internal dataset blocks are not accepted as fresh input.
- Composite LR scaling is locally exercised for actual torch AdamW and this fork's ROSE; accepting an optimizer name is not sufficient. Unsupported concrete/fused-backward optimizers are rejected. ROSE implementation is not globally changed.
- Nonunit composite LR requires both accumulation controls1 and category-homogeneous batches. Use nonpooled buckets, or batch-size1 for nonbucketed loading. Mixed normal/composite batching or accumulation is rejected, never averaged or converted to loss weighting.
- Existing ordinary masks and supported auxiliary losses remain in the loss path. Generated composites cannot inherit source-only mask/control files; explicit incompatible configurations are rejected rather than silently stripped. Avoid identity-reference assumptions that assign a single subject to every unrelated face.
- Conflicting prior objectives (e.g. inverted-mask prior/other teacher conditioning), adapters, textual embeddings, or unsupported guidance/training modes produce validation errors. Turn off the conflicting component deliberately or retain your original non-Isolation workflow.
- Raw checkpoints preserve successful-update DOP phase and existing sampler/optimizer/scheduler state. Data/configuration identity changes reject compatible-resume claims. GPU numerical determinism is not promised; existing exact-replay restrictions still apply. A weights-only warm start is a new trajectory, not continuation of the original phase.

## User-run manual training checklist

1. Prepare the pinned assets explicitly, or point to a verified complete local directory. Keep license/provenance metadata.
2. Back up your run configuration. Choose one or two consented subject folders and inspect original captions. Do not change existing caption files for fusion.
3. Use caption/inspect actions (or UI controls) to inspect all identity mappings and DOP substitutions. Check repeated triggers, multi-subject captions, and already-fused captions.
4. Prepare composites explicitly. Verify dimensions1024×512, random left/right placement, stranger-first captions, identity correctness, no source duplication, and unmodified originals. Check the generated manifest and exposures.
5. **You run** a short Krea2/Raw training smoke using your normal environment and chosen real data. Start from a fresh output name; use a short input-step budget, sample/save intervals, and batch/GA restrictions above. This assistant does not launch it.
6. Confirm logs alternate selected DOP windows and skipped preservation without fake zero-loss measurements. Confirm composite category updates use the multiplier once and normal updates return to scheduled LR; verify scheduler advances only on successful updates.
7. Confirm cached and uncached captions match, including dropout if enabled. Inspect actual standard previews for transformed prompts, subject likeness, and unrelated faces; no effectiveness conclusion from one smoke image.
8. Save at a completed update boundary. Resume the unchanged run with its raw training-state file and compatible resume mode. Confirm completed-update count, DOP phase, sampler cursor, category progression, and LRs continue. Changing mappings/source/assets should reject reuse of the old trajectory.
9. Compare logs with the exposure report. Record any skipped update, OOM, or bucket-tail duplication. Do not increase steps automatically to compensate without documenting it.
10. Only after these checks, execute the separate real-data experiment plan yourself. Share results before any experiment-driven changes.

## Troubleshooting and evidence to collect

Missing assets: run only the explicit preparation command or supply the exact complete pinned directory. Hash failure: retain metadata/path/error and use a fresh destination; do not bypass verification. Generated-cache error: retain the manifest and expected/observed files; explicitly remove only the affected generated cache after diagnosis. Mapping error: use unique invented triggers and explicit nouns; inspect collisions instead of substring substitutions. Mixed-category error: use supported bucketing/pooling/GA settings, not loss scaling.

Collect: exact run config (redact tokens), source/generated manifests and hashes, asset revision, Toolkit/Fizgig revisions, Python/PyTorch/CUDA/driver/optimizer versions, raw checkpoint metadata, completed-update/sampler counters, preservation pass logs, raw/weighted losses, category/effective-LR logs, exposure report, exception traceback, VRAM/time, and representative prompts/seeds/images you consent to share. Do not publish private source photos or secrets by default.

Local validation proves only the paths listed in the implementation report. Full Krea denoiser/text-encoder/VAE execution and real-data effectiveness require the user's manual GPU validation. No quality/isolation improvement is claimed before those results.
