# Ostris AI Toolkit

AI Toolkit is an easy to use all in one training suite for diffusion models. I try to support all the latest models on consumer grade hardware. Image and video models. It can be run as a GUI or CLI. It is designed to be easy to use but still have every feature imaginable. Free and open source.

> [!NOTE]
> This is a fork of [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit). See [What's different from upstream](#whats-different-from-upstream) below for the additions in this fork.

## What's different from upstream

This fork tracks `upstream/main` closely (currently 0 commits behind) and adds the following on top:

> [!NOTE]
> The perceptual anchors, weight/gradient noising, rose optimizer, and auto-masking features were ported from [BuffaloBuffaloBuffaloBuffalo/ai-toolkit-perceptual](https://github.com/BuffaloBuffaloBuffaloBuffalo/ai-toolkit-perceptual) (MIT licensed) and adapted to work with Krea 2 on top of this fork's newer upstream base. Full credit to that project for the original implementation.

### Krea 2 perceptual anchor training

Auxiliary losses that can anchor a Krea 2 LoRA to structural ground truth; they do not guarantee identity/geometry usefulness without the required model weights and runtime evidence. Each anchor can be enabled independently only when its dependency/admission gate is satisfied, with per-dataset weight and timestep-range overrides (`*_loss_weight`, `*_loss_min_t/max_t`):

- **Depth anchor** — Depth Anything V2 (incl. DA2-Large option) depth-GT loss with a caching pipeline (depth maps are precomputed and round-tripped through the Krea VAE so train-time decode matches), unified latent decode, and depth-step-based preview rendering.
- **Face identity anchor** — ArcFace embedding loss.
- **Body proportion anchor** — ViTPose keypoint-based loss with an attached confidence-shortfall gradient; synthetic CPU graph evidence does not certify useful real-checkpoint gradients, so the licensed dependency/runtime gate remains required.
- **Surface normal anchor** — Sapiens normal-map loss.
- **Body shape anchor** — HybrIK-based loss.
- **Cross-VAE anchor** — perceptual loss through a frozen Flux 2 VAE encoder; requires a matching licensed local `ae.safetensors` checkpoint and remains unavailable when that backend is not present.
- **Auto subject masking + region-weighted loss** — YOLO person detection + SegFormer semantics restrict anchor losses to the subject.
- A **loss split resolver** and **loss watch** instrumentation manage how the anchor losses combine with the base flow-matching loss, and a **Perceptual Anchors UI panel** (with safe depth-config migration) exposes all of it in the GUI.

Extra dependencies live in `requirements_perceptual.txt`.

### Perceptual noising

Noise injection on LoRA weights and/or gradients (`train.weight_noise` / `train.gradient_noise`) with UI controls. Zero-RMS parameters are preserved (skipped) so sparse/perceptual parameters are not corrupted.

### Per-image adaptive learning rate

`per_image_adaptive_lr` tracks each dataset image's loss trend across update windows, with logging, warmup windows (`per_image_adaptive_lr_warmup_windows`), resolution-aware adjustment scaling, and an observation-only `per_image_adaptive_lr_stats_only` mode. The default `train.per_image_adaptive_lr_mode: loss` applies the advertised sample weight to the joint visual/audio objective once; the experimental `lr` mode leaves loss terms unchanged and scales the whole successful optimizer update. Audio tuning is therefore also affected by dataset mix and `train.audio_loss_multiplier`; `num_repeats` changes sampling frequency and is not a loss multiplier.

### Rose optimizer

Stateless `rose` optimizer (`toolkit/optimizers/rose.py`), usable like any other optimizer in the config.

### MiniMax-H3 Turbo preview LoRA

`model.preview_lora_path` (default off) and `model.preview_lora_strength` (default `1.0`). Sampling-only; never trained or saved. Path is a local file, a filename under `models/loras`, or `user/repo/file.safetensors`. H3-only (`model.arch: minimax_h3`). Mutually exclusive with `model.inference_lora_path`.

### H3 standalone voice (audio-only) datasets

On MiniMax-H3, `datasets[].do_audio: true` enumerates standalone `.wav` / `.mp3` / `.flac` / `.m4a` (and the rest of the global audio extensions) as voice items, not ACE-Step. Requires `datasets[].buckets: true`. Durations must decode to the 17n+5 frame grid at 24 fps and the audio-VAE hop alignment; the voice gate is content/geometry based, not a visual product minimum-duration rule. Audio-only loss uses `train.audio_loss_multiplier` (default `1.0`) and does not add a video term.

### H3/Krea admission and saved-job migration

Before loading weights or preparing caches, Python resolves every process and
dataset override through the side-effect-free admission contract:

```bash
python -m toolkit.admission path/to/job.yaml
```

The JSON result is `{valid, diagnostics, deferred, resolved}`. Each diagnostic
contains `rule_id`, `severity`, `fields`, plain-language `reason`, actionable
`remedy`, and `phase`; `deferred` entries are runtime-only checks, not a
validation pass. The CLI does not construct models/optimizers/accelerators,
decode or collate media, write caches, or launch jobs (it may inspect known
dataset paths read-only to identify possible audio files). The same interface
is available to API/UI callers as
`toolkit.admission.collect_admission_diagnostics(config, source="api")`.
H3 rejects legacy accumulation, mean-flow, fused multi-backward windows,
target-builder collisions, image-only auxiliary overrides, unsupported clocks
or Fast/VSA controls, and self-D-OPSD voice targets. H3 audio itself is not
blanket-blocked: known files still receive the bucket, duration, silence, and
hop-alignment final checks.

Existing saved jobs remain editable, but imported, cloned, queued, direct-CLI,
and worker launches must revalidate the current configuration before saving or
changing queue/running state. Old jobs using
`gradient_accumulation_steps != 1`, `mean_flow`, unsupported H3 auxiliaries,
or fused multi-backward settings require an explicit config edit; admission
never silently translates, removes, or downgrades those fields. A scrubbed
accepted character recipe is maintained at
`testing/fixtures/admission/accepted_h3_character.yaml`.

### Release capability contract (Krea/H3 remediation)

This fork's Krea/H3 admission and training behavior is evidence-scoped. A CPU regression result does not certify a real checkpoint, GPU precision path, or visual/audio quality. The following labels are intentional:

- **Verified CPU contract** means the production consumer and an independent CPU oracle were exercised. It does not imply model-weight usefulness.
- **Temporary gate** means a known-broken or unproven path is rejected until its production acceptance evidence exists.
- **Experimental / uncertified** means the path is retained for controlled experiments, but this fork has no release-quality tolerance or multi-rank proof for it. Uncertified is not proof of corruption.

For joint H3 training, the fixed sample-weight contract is

```text
L = (1/B) * sum_i w_i * [visual_i + audio_loss_multiplier * audio_present_i * audio_i]
```

`visual_i` and `audio_i` are means over each item's own target elements. Standalone voice omits `visual_i`; an absent soundtrack contributes no audio term but remains in the overall `B` denominator. `audio_present` is a data fact, `do_audio` is supervision policy, and encoded silence is context. `num_repeats` changes sampling frequency and is not a replacement for `w_i` or `audio_loss_multiplier`. This is the F11 visual-only-to-joint/voice semantics cutover; prior visual-only adaptive-weight A/B results are not joint-objective evidence.

For the current flow wrappers, `sigma` is resolved in FP32 from the actual interpolation (`sigma = timestep / 1000` for the toolkit wrapper), `x_sigma = (1 - sigma) * x0 + sigma * epsilon`, and the noise-clean velocity target is `epsilon - x0`. H3 keeps its native time/sign conversion inside the H3 wrapper; the official scheduler/inference grid is a semantic reference, not a drop-in replacement for toolkit training targets. The unaugmented DDPM velocity conversion remains rejected for Krea/H3 until a model-specific contract is proven.

H3 latent and condition caches use versioned, provenance-bearing identities. The repaired cache namespace includes `latent_space_version=minimax_h3_v2` and latent provenance version `2`, while reference-video conditions use `recipe_namespace=h3_ref_video_condition_v3`; identities also include source-content fingerprint, model/VAE/checkpoint identity, transform/control/audio recipe, encoder identity, and target geometry where applicable. H3 condition keyframes use the dedicated CPU seed-42 posterior sample plus fp16-to-fp32 roundtrip before normalization; target posterior sampling remains a separate semantic and is intentionally fixed when it is cached. Cache fixes invalidate changed representations only: old entries are left in place and are not considered valid solely because shapes match; unrelated user caches are not deleted.

Krea guidance is **RAW training recommended / Turbo inference recommended**. Turbo training-adapter and edit roles are fork-specific warnings, not official equivalence claims. Krea full-tune preview/edit CFG, source-target, and CFG-Zero paths remain temporary-gated until real production acceptance evidence; adapter-only CPU state tests do not prove preview quality. H3 audio has precise 24-fps/32-kHz/stereo and 17n+5 frame/hop gates; it is not blanket-blocked, but non-grid durations, missing cached audio, and unsupported standalone-voice combinations remain scoped gates.

Quantized-base loading, FP32 islands, low-precision/AMP, offload, compile, sparse/VSA/TREAD, and distributed modes retain their current experimental labels unless the release report names an exercised tolerance. The current evidence is CPU-only for arithmetic/lifecycle and static/cache contracts; no blanket GPU claim is made. Distributed synchronization, rank-state resume, and exact replay outside controlled `num_workers=0` are **experimental / uncertified**.

`train.resume_mode` is `auto` by default. A fresh output directory starts normally; `auto` resumes only a complete matching raw `training_state.pt` within the admitted deterministic scope. Legacy or inference-only files require an explicit `weights_only` choice. `exact` restores matched raw parameters, optimizer, scheduler, EMA/count, Python/NumPy/Torch RNG, data order, and `completed_update_id`. Its current admitted scope is CPU, single-process, `num_workers: 0`, no buckets, accumulation 1, and compile off. GPU, bucketed, multiworker, accumulated and compiled exact replay are unavailable, not silently approximated. `weights_only` deliberately resets optimizer/scheduler/EMA/counters/RNG/order and is never called resume.

Remediation verification (2026-10-04): 321 scoped Python cases and 9 UI policy cases passed; UI/extension typechecks and Ruff's syntax/undefined-name checks passed. The CPU raw-checkpoint smoke resumed update 3 through update 6 with maximum parameter error `0.0`, preserving EMA while keeping raw training weights distinct from the inference export. Isolated production API handlers rejected invalid YAML, incompatible configs, spoofed job metadata, and invalid stored start/queue requests before database mutation.

Deployment evidence is separate: the already-running GUI still returned HTTP 404 for the new `/api/admission` route. It was inspected and closed without saving or launching a job; no service was restarted. Updated-GUI visual acceptance and real Krea/H3 checkpoint/GPU acceptance remain open. Do not treat these CPU/source-handler checks as release approval for those capabilities.

Pinned references used for this release ledger:

- Official [Krea repository revision `db3984fbc6e13b34c0064990fc2d95ac64d00058`](https://github.com/krea-ai/krea-2/commit/db3984fbc6e13b34c0064990fc2d95ac64d00058).
- Official [MiniMax H3 revision `d21241f0a4b3acbb34c97dae47fa417b7065e438`](https://github.com/MiniMax-AI/MiniMax-H3/commit/d21241f0a4b3acbb34c97dae47fa417b7065e438) and [checkpoint metadata revision `42ed227ee7df40d41602854ae760620d6eb651fe`](https://huggingface.co/MiniMaxAI/MiniMax-H3/commit/42ed227ee7df40d41602854ae760620d6eb651fe).
- [Diffusers H3 scheduler revision `8b33bfc04b6b5e8bb58a58e55f68746c1bbee4cd`](https://github.com/huggingface/diffusers/commit/8b33bfc04b6b5e8bb58a58e55f68746c1bbee4cd).


### MiniMax-H3 findings ported from musubi-tuner

Measured behaviour from [kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner)'s H3 docs (`minimax_h3*.md`), checked against this fork. These change default behaviour on every H3 run:

- **Base-sigma thresholds** — H3 sigma thresholds are in the pre-shift *base* space (1 = pure noise), as in musubi. `timesteps / 1000` is the post-shift-12 video sigma, so `train.guidance_loss_sigma_min: 0.15` used to gate base ~0.0145 (~1.5% of steps); it now gates base 0.15 (= video sigma ~0.68, the intended ~15%). The `guidance/base_sigma` and `teacher/base_sigma` logs are base sigma too. Other models keep `timesteps / 1000`.
- **Encoded silence** — clips without a soundtrack, `do_audio: false` datasets and image items ride with the audio VAE's encoding of silence instead of zero latents (a zero latent decodes to broadband noise at ~-26 dBFS). Its noise is shared across passes, so guidance/teacher probes see the student's audio rows. Video-only training that keeps real audio as context: `do_audio: true` + `train.audio_loss_multiplier: 0`.
- **fp32 VAE encoding** — training targets, keyframes and references encode with the video VAE encoder upcast to fp32 (the fp16 Comfy repack left outliers up to ~0.4 on a latent std of ~1). `latent_space_version` is now `minimax_h3_v2`, so H3 latent and reference-video caches rebuild once; `model.model_kwargs.vae_encode_fp32: false` keeps fp16 encodes and the existing v1 caches.
- **Samples keep the LoRA live on quantized bases** — training samples used to merge the LoRA into the base for speed. On a quantized base (H3's ConvRot int8, any ostris / torchao / quanto layer) that re-quantizes: measured on convrot8, a LoRA with delta rms 0.2% of the weights kept 0% of its effect in samples (1% kept 79%), and merge-out walked the frozen base 0.02–0.3% per sample cycle. Samples now run the LoRA as a live branch there (musubi's `--lora_runtime_attach`); plain bases still merge.

### Other additions

- `inference_lora_path` for Krea 2 — load a separate LoRA for turbo sampling during training samples.
- UI settings **offline mode** — starts jobs with `HF_HUB_OFFLINE=1`.
- Env-gated CUDA memory diagnostics (`KREA2_MEM_DIAG`) in the SD trainer.
- Fix: timer no longer divides by zero on empty buckets after OOM recovery.
- Fix: a sampling exception no longer leaves `assistant_lora` / `inference_lora` permanently disabled — restore runs in `finally`. Affects Flux, Krea2, z_image, wan22, and H3.
- Fix: prior predictions no longer replace the embeds they are given with the batch's cached caption embeds (a re-indent had pulled that block out of the ClipVision/embedding branch). From 2026-09-09 until this fix the H3 D-OPSD teacher ran on the student's plain caption embeds, and DOP / blank-prompt preservation priors used the training caption.
- Fix: a failure while building the sample pipeline now still restores the training RNG and device state and merges the network back out.
- Test infrastructure: unit tests in `testing/` plus a real-data integration harness in `testing/integration/` (perceptual noising QA, depth consistency, gradient-contract probes).

### Experimental

Default-off. Fail-closed at startup when the optimizer cannot honor the requested mode.

- **Per-image adaptive LR `lr` mode** — `train.per_image_adaptive_lr_mode: lr` (default `loss`). Requires `train.per_image_adaptive_lr: true`. In `loss` mode the effective objective is `(1/B) * sum_i w_i * [visual_i + audio_loss_multiplier * audio_present_i * audio_i]`; standalone voice omits `visual_i`, absent soundtrack rows remain in the overall `B` denominator, and zero-weight rows are excluded before sensitive arithmetic. In `lr` mode every loss component remains unscaled and the optimizer's applied update for the whole successful window is scaled instead. Requires an optimizer that `supports_step_scale`. Discrete watcher multipliers are not min/max LR bounds. On optimizers that normalise by a gradient-range statistic (Rose), a uniform loss scale cancels. Older visual-only adaptive-weight A/B results are pre-cutover evidence and must not be compared as joint/voice results.
- **H3 modality block routing** — `train.modality_block_routing` with optional `photo_blocks`, `clip_blocks`, `voice_blocks`. Default off; a blank/omitted key leaves that modality unrestricted. Spec is a range list such as `"3-12, 14-15, 22,27,31-33"`, parsed against `len(transformer.blocks)`. H3 LoRA/LoKr only. Requires an optimizer that `supports_active_param_mask` (Automagic3 is rejected). Mixed-modality windows are left unrestricted (every block trains); refiners and non-trunk adapters stay active. Measured A/B at rank-16 / 2000-step character-LoRA scale: no measurable quality benefit (interference protection only, no speed gain) — keep off for that recipe. That A/B measured the mask as built: the full backward still runs (masked gradients are computed, then dropped) and the token refiner keeps training. It is not a test of a block window with the refiner frozen; for that, see `network.train_blocks` below.
- **H3 block window (`network.train_blocks`)** — LoRA on a range of trunk blocks and nothing else, e.g. `"20-49"`. The token refiner, `final_layer`, and blocks outside the range get no adapter, so nothing before the first selected block is trainable and autograd stops there (Fizgig measured ~23% faster steps on int8 for its 20-49 recipe). Applies to every item type (photos, clips, voice). H3 LoRA/LoKr only; mutually exclusive with `network_kwargs.only_if_contains`; `ignore_if_contains` still applies. Verified at network build by module ownership (fails if any adapter sits outside the window or a selected block got none). Refused while the text encoder or an embedding trains (text rows pass through every block). Uncompiled runs also check on the first training forward that the block before the window needs no grad; compiled runs skip that runtime check. After Fizgig's "Default" training mode.
- **Low-noise share (`train.low_noise_share`)** — fraction of training draws below sigma 0.5, e.g. `0.6`. Solves the static shift that puts exactly that share of the training grid below 0.5 and uses it for the training draw only; the model's configured shift (12 on H3), sampling and previews are unchanged. H3's own draw puts ~6.6% of steps below 0.5 (and never goes below sigma ~0.126); `0.6` is Fizgig's "Likeness and Style". On H3 the share is the *video* share: audio rows keep the model's fixed video→audio pairing (`remap_sigma` from shift 12, the pairing inference uses at every step), so audio lands cleaner — `0.6` puts ~86% of audio draws below 0.5 (default ~24%). Remapping from the training shift instead would pair clean video with noisy audio, a combination inference never produces; Fizgig trains the same way. Both shares are logged at startup. Requires `noise_scheduler: flowmatch`, `timestep_type: shift`, a static-shift scheduler, `content_or_style: balanced`, full denoising range, and `first_timestep_chance: 0`; anything else fails at startup. The startup log prints the solved shift and sigma range.
- **Category stop (`train.category_stop`)** — `photo_step` / `clip_step` / `voice_step` retire that category once the training step reaches it; blank = never. `mode: anchor` (default) keeps training it at 0.1× the optimizer's applied update (requires `supports_step_scale`); `mode: stop` skips its batches before the forward pass. Requires `batch_size: 1` (train-level and any `datasets[].batch_size`, which overrides it when buckets are on) and no gradient accumulation (one category per update). `mode` must be a string: an unquoted YAML `off` loads as a boolean and is refused, not read as anchor. Regularisation items (`is_reg`) never count toward a category, so they are neither scaled nor skipped. Anchor mode needs a trainer that opens the optimizer runtime window (`diffusion_trainer` does) and is refused otherwise. Step-based, so it holds across resume. After Fizgig's per-category stop epoch.
- **EMA warmup (`train.ema_config.warmup`)** — ramps the EMA decay in as `min(ema_decay, (1+n)/(10+n))`, where `n` is the completed successful optimizer-update count. Scheduler, EMA and this count advance only after a real optimizer update; empty, OOM-discarded and AMP-skipped windows do not advance it. Raw resumable checkpoints must restore the matched `completed_update_id`, EMA shadow/count, optimizer/scheduler, RNG and data-order state atomically. An inference/export file is a weights-only warm start and does not resume that trajectory. Exact replay is only claimed for controlled deterministic input order (first acceptance scope: `num_workers=0`); multiworker, distributed and nondeterministic paths remain uncertified.
- **H3 TREAD token routing** — training-only clip-step token skip (Krause et al., [arXiv 2501.04765](https://arxiv.org/abs/2501.04765)). H3-only (`model.arch: minimax_h3*`; VSA / `gate_compress` is rejected at bind). Default off. Clip steps with batch size 1 and more than one latent video frame skip a random `tread_ratio` of *target* video tokens around blocks `[tread_start, tread_end)` and keep `1 - tread_ratio`; skipped rows rejoin in their start-block state. Text, condition, and audio rows stay. Stills, inference, and `batch_size != 1` never route. Bind rejects a post-rejoin tail shorter than 3 blocks (`tread_end` 47 on a 50-block trunk is the Fizgig prior). No UI until a config-file experiment passes the speed gate.
- **H3 D-OPSD other-photo / identity-first** — extends existing `model.model_kwargs.dopsd` (self-reference: one ref2va DiT, two forwards). Default-off. `dopsd_ref_mode: other` pairs each still with a different photo in the same folder (never itself, never its own flip, never another folder); clips and voice sit out. `dopsd_identity_first` is teacher-only at 1/3 LR for `dopsd_identity_first_steps` optimizer updates (`-1` → 650), then drops the teacher (one forward, full LR). Other-photo requires `train.batch_size: 1`. Identity-first requires an optimizer that `supports_step_scale`. No second teacher model. A/B tested and not adopted for the character-LoRA recipe; kept for further experimentation. No UI until a config-file experiment. (Those A/Bs predate the prior-embeds fix above: the teacher saw the plain caption, so they are worth re-running.)
- **Timestep focus (`train.timestep_focus_prob`)** — musubi's H3 timestep focus. With probability P a draw lands uniformly in base sigma `[timestep_focus_min, timestep_focus_max)` (default `0.4`–`0.8`, video sigma ~0.89–0.98 on H3, where content is decided); otherwise it stays as drawn. Band density becomes `P + (1-P)·band_share`; musubi measured ~2x faster convergence of that band at `0.5`. The band is on the model's own schedule, so it composes with `low_noise_share` (band points are weighted to stay uniform in base sigma on a re-bent grid) and is intersected with the denoising range. Fails closed with the cubic `content_or_style` bias and the fixed-step timestep types. Default off.
- **H3 D-OPSD teacher recipes (musubi `ref` / `subject_ref`)** — all `model.model_kwargs`, default off (D-OPSD unchanged):
  - `dopsd_loss_mag_weight` / `dopsd_loss_dc_weight` (default `1.0` = plain MSE): the teacher loss as musubi's exact magnitude/direction split of the MSE with the student-norm factor detached, so the student commits to the teacher's norm instead of the conditional mean's shrunken one (washed-out output). The DC weight scales the residual's per-channel mean, a global colour/tone cast (`0.3` keeps the dataset palette from being learned as style; keep `1.0` for style LoRAs). Needs `loss_type: mse`.
  - `dopsd_teacher_sigma_max` / `dopsd_teacher_sigma_min` (base sigma; default `1.0` / `0`): outside the band the teacher drops the reference and runs on the student's own text, a base-preservation anchor (full loss weights, no photo bleed, scaled by `dopsd_preservation_weight`). Logs `teacher/conditioned`, `loss/teaching`, `loss/anchor`. Needs `train.batch_size: 1`.
  - `dopsd_copy_declaration`: self-reference video teacher captions get the official copy declaration (`<Video 1>` fully_preserved, `<Audio 1>` fully_copy when the soundtrack rides along), which opens audio teaching.
  - `dopsd_subject_declaration`: other-photo teacher captions get the official subject-reference declaration; the reference token becomes `<Subject 1>` (defined as the subject of `<Picture 1>`), which doubles as the student's trigger.
  - `dopsd_musubi_recipe: true` fills musubi's validated values for unset keys — self: `sigma_max 0.75`, `dc 0.3`, copy declaration; other: `sigma_max 1.0`, `sigma_min 0.15`, `mag 0.5`, `dc 0.3`, subject declaration. The train-side parts are only logged: self → `timestep_focus_prob: 0.5`, keep checkpoints at or just after the ~300-step plateau; other → `lr: 3e-4`, 50 warmup steps, ~500 steps. Startup warns when a band contradicts the mode (other-photo with `sigma_max < 1` anchors the band where identity is decided).
  - `partition: fl2va` / `fl2va_pruned` is accepted with `dopsd: true` (musubi: the FL2VA weights copy a self-reference far more literally than Ref2VA); the student then trains on the FL2VA base and the LoRA records `minimax_h3_fl2va`.
  - musubi's caveat: a teacher that sees the target itself (self-reference) loses the distilled guidance amplification on image targets; for stills they use the other-photo teacher only.


```yaml
train:
  tread_ratio: 0.0   # 0.0 = off. (0, 1) enables. 0.5 skips half the target video tokens
  tread_start: 2     # first routed block (inclusive)
  tread_end: 47      # rejoin block (exclusive); 50-block trunk leaves blocks 47-49
```

```yaml
model:
  model_kwargs:
    dopsd: true
    dopsd_ref_mode: self          # self | other
    dopsd_ref_count: 1
    dopsd_identity_first: false
    dopsd_identity_first_steps: -1  # -1 = 650 optimizer updates
    dopsd_musubi_recipe: true       # musubi's per-mode teacher recipe for unset keys
    # dopsd_teacher_sigma_max: 0.75 # base sigma; above it the step is a base anchor
    # dopsd_loss_dc_weight: 0.3     # 1.0 = plain MSE
    # vae_encode_fp32: false        # keep fp16 encodes and the v1 latent caches
train:
  timestep_focus_prob: 0.5          # 0 = off; band [timestep_focus_min, timestep_focus_max)
```

```yaml
network:
  train_blocks: "20-49"     # H3 trunk blocks only; backward stops at block 20
train:
  timestep_type: shift
  low_noise_share: 0.6      # 60% of training draws below sigma 0.5
  category_stop:
    voice_step: 1500        # null = never; also photo_step, clip_step
    mode: anchor            # anchor (0.1x update) | stop (skip batches)
  ema_config:
    use_ema: true
    ema_decay: 0.98
    warmup: true
```



## Supported Models

### Image
- [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) (FLUX.1)
- [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) (FLUX.2)
- [black-forest-labs/FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) (FLUX.2-klein-base-4B)
- [black-forest-labs/FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) (FLUX.2-klein-base-9B)
- [ostris/Flex.1-alpha](https://huggingface.co/ostris/Flex.1-alpha) (Flex.1)
- [ostris/Flex.2-preview](https://huggingface.co/ostris/Flex.2-preview) (Flex.2)
- [lodestones/Chroma1-Base](https://huggingface.co/lodestones/Chroma1-Base) (Chroma)
- [Alpha-VLLM/Lumina-Image-2.0](https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0) (Lumina2)
- [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) (Qwen-Image)
- [Qwen/Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512) (Qwen-Image-2512)
- [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) (Qwen-Image-2.1)
- [HiDream-ai/HiDream-I1-Full](https://huggingface.co/HiDream-ai/HiDream-I1-Full) (HiDream I1)
- [OmniGen2/OmniGen2](https://huggingface.co/OmniGen2/OmniGen2) (OmniGen2)
- [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) (Z-Image Turbo)
- [Tongyi-MAI/Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) (Z-Image)
- [ostris/Z-Image-De-Turbo](https://huggingface.co/ostris/Z-Image-De-Turbo) (Z-Image De-Turbo)
- [zhen-nan/L2P](https://huggingface.co/zhen-nan/L2P) (Z-Image L2P)
- [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) (SDXL)
- [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) (SD 1.5)
- [baidu/ERNIE-Image](https://huggingface.co/baidu/ERNIE-Image) (ERNIE-Image)
- [NucleusAI/Nucleus-Image](https://huggingface.co/NucleusAI/Nucleus-Image) (Nucleus-Image)
- [Boogu/Boogu-Image-0.1-Base](https://huggingface.co/Boogu/Boogu-Image-0.1-Base) (Boogu Image 0.1)
- [HiDream-ai/HiDream-O1-Image](https://huggingface.co/HiDream-ai/HiDream-O1-Image) (HiDream O1)
- [ideogram-ai/ideogram-4-fp8](https://huggingface.co/ideogram-ai/ideogram-4-fp8) (Ideogram 4 FP8)
- [Photoroom/prxpixel-t2i](https://huggingface.co/Photoroom/prxpixel-t2i) (PRXPixel)
- [circlestone-labs/Anima-Base-v1.0-Diffusers](https://huggingface.co/circlestone-labs/Anima-Base-v1.0-Diffusers) (Anima)
- [krea/Krea-2-Raw](https://huggingface.co/krea/Krea-2-Raw) (Krea 2)
- [krea/Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo) (Krea 2 Turbo)
- [inclusionAI/Ming-Image-0.1-Design](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design) (Ming-Image 0.1 Design)
- [microsoft/Mage-Flow-Base](https://huggingface.co/microsoft/Mage-Flow-Base) (Mage-Flow)

### Instruction / Edit
- [black-forest-labs/FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) (FLUX.1-Kontext-dev)
- [Qwen/Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit) (Qwen-Image-Edit)
- [Qwen/Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) (Qwen-Image-Edit-2509)
- [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) (Qwen-Image-Edit-2511)
- [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) (Qwen-Image-2.1) - one model for both; it edits when your dataset has control images
- [HiDream-ai/HiDream-E1-1](https://huggingface.co/HiDream-ai/HiDream-E1-1) (HiDream E1)
- [Boogu/Boogu-Image-0.1-Edit](https://huggingface.co/Boogu/Boogu-Image-0.1-Edit) (Boogu Image Edit)
- [krea/Krea-2-Raw](https://huggingface.co/krea/Krea-2-Raw) (Krea 2 Edit Training)
- [krea/Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo) (Krea 2 Turbo Edit Training)
- [microsoft/Mage-Flow-Edit-Base](https://huggingface.co/microsoft/Mage-Flow-Edit-Base) (Mage-Flow Edit)

### Video
- [Wan-AI/Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) (Wan 2.1 1.3B)
- [Wan-AI/Wan2.1-I2V-14B-480P-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers) (Wan 2.1 I2V 14B-480P)
- [Wan-AI/Wan2.1-I2V-14B-720P-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers) (Wan 2.1 I2V 14B-720P)
- [Wan-AI/Wan2.1-T2V-14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers) (Wan 2.1 14B)
- [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) (Wan 2.2 14B)
- [Wan-AI/Wan2.2-I2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) (Wan 2.2 I2V 14B)
- [Wan-AI/Wan2.2-TI2V-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) (Wan 2.2 TI2V 5B)
- [Lightricks/LTX-2](https://huggingface.co/Lightricks/LTX-2) (LTX-2)
- [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) (LTX-2.3)
- [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) (LTX-2.5)
- [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) (MiniMaxAI/MiniMax-H3)
- [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) (MiniMax-H3 Ref2V) - reference image to video

### Audio
- [ACE-Step/Ace-Step1.5](https://huggingface.co/ACE-Step/Ace-Step1.5) (Ace Step 1.5)
- [ACE-Step/acestep-v15-xl-base](https://huggingface.co/ACE-Step/acestep-v15-xl-base) (Ace Step 1.5 XL)
- [m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) (YuE2) - the official audio-to-token encoder is unreleased; training uses the community tokenizer by Kytra ([@sin_ceriously](https://x.com/sin_ceriously)), [Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4](https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4).

### LLM
- [Qwen/Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) (Qwen2.5-Omni)

### Experimental
- [lodestones/Zeta-Chroma](https://huggingface.co/lodestones/Zeta-Chroma) (Zeta Chroma)

## Installation

### Install with the AI Toolkit Manager (experimental)

The recommended way to install and run AI Toolkit is with the **AI Toolkit
Manager**, built into this repo. The manager detects your hardware and sets up
the right PyTorch build, creates the python environment, and grabs local copies
of Node.js and FFmpeg — everything stays inside the ai-toolkit folder, nothing
is installed system-wide. On every launch the manager checks for updates and
applies them (your local changes are never overwritten — if you have modified
files, the update is skipped with a warning), then starts the UI at
`http://localhost:8675`.

The manager is still **experimental** — please let me know if you have any
issues with it. The manual instructions below still work if you prefer them
or run into problems.

The only requirement is **git** (on Windows the manager can even fetch a
portable git for updates, but you need one installed to clone the repo first).

```bash
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
```

Then start the manager with the script for your platform:

Linux (x86_64 and ARM64, including DGX Spark / DGX OS):
```bash
chmod +x run_linux.sh
./run_linux.sh
```

MacOS (Apple Silicon, experimental):
```bash
chmod +x run_mac.zsh
./run_mac.zsh
```

Windows: double-click `run_windows.bat` (or run it from a terminal).

You can also use the manager directly from a terminal (handy on headless
servers):

```bash
python3 -m manager install   # first-time setup
python3 -m manager update    # pull updates + sync dependencies
python3 -m manager launch    # start the UI
python3 -m manager doctor    # diagnose problems
```

### Manual installation

Requirements:
- python >=3.10 (3.12 recommended)
- Nvidia GPU with enough ram to do what you need
- python venv
- git


Linux:
```bash
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
python3 -m venv venv
source venv/bin/activate
# install torch first
pip3 install --no-cache-dir torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip3 install -r requirements.txt
```

These steps also work on ARM64 Linux, including DGX Spark / DGX OS.


Windows:

If you are having issues with Windows. I recommend using the easy install script at [https://github.com/Tavris1/AI-Toolkit-Easy-Install](https://github.com/Tavris1/AI-Toolkit-Easy-Install)

```bash
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
python -m venv venv
.\venv\Scripts\activate
pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```


# AI Toolkit UI

<img src="https://ostris.com/wp-content/uploads/2025/02/toolkit-ui.jpg" alt="AI Toolkit UI" width="100%">

The AI Toolkit UI is a web interface for the AI Toolkit. It allows you to easily start, stop, and monitor jobs. It also allows you to easily train models with a few clicks. It also allows you to set a token for the UI to prevent unauthorized access so it is mostly safe to run on an exposed server.

## Running the UI

Requirements:
- Node.js > 20

The UI does not need to be kept running for the jobs to run. It is only needed to start/stop/monitor jobs. The commands below
will install / update the UI and it's dependencies and start the UI. 

```bash
cd ui
npm run build_and_start
```

You can now access the UI at `http://localhost:8675` or `http://<your-ip>:8675` if you are running it on a server.

## Securing the UI

If you are hosting the UI on a cloud provider or any network that is not secure, I highly recommend securing it with an auth token. 
You can do this by setting the environment variable `AI_TOOLKIT_AUTH` to super secure password. This token will be required to access
the UI. You can set this when starting the UI like so:

```bash
# Linux
AI_TOOLKIT_AUTH=super_secure_password npm run build_and_start

# Windows
set AI_TOOLKIT_AUTH=super_secure_password && npm run build_and_start

# Windows Powershell
$env:AI_TOOLKIT_AUTH="super_secure_password"; npm run build_and_start
```

### Training
1. Copy the example config file located at `config/examples/train_lora_flux_24gb.yaml` (`config/examples/train_lora_flux_schnell_24gb.yaml` for schnell) to the `config` folder and rename it to `whatever_you_want.yml`
2. Edit the file following the comments in the file
3. Run the file like so `python run.py config/whatever_you_want.yml`

A folder with the name and the training folder from the config file will be created when you start. It will have all 
checkpoints and images in it. You can stop the training at any time using ctrl+c and when you resume, it will pick back up
from the last checkpoint.

IMPORTANT. If you press crtl+c while it is saving, it will likely corrupt that checkpoint. So wait until it is done saving

### Need help?

Please do not open a bug report unless it is a bug in the code. You are welcome to [Join my Discord](https://discord.gg/VXmU2f5WEU)
and ask for help there. However, please refrain from PMing me directly with general question or support. Ask in the discord
and I will answer when I can.

## Ostris Cloud

You can use many cloud providers to rent GPUs. If you want to help support this project in the largest way possible, please consider using [Ostris Cloud](https://cloud.ostris.com). Ostris Cloud is owned and operated by me, Ostris, and every dollar earned goes directly back into funding the development of this project.

<a href="https://cloud.ostris.com" target="_blank"><img src="https://cloud.ostris.com/api/og" alt="Ostris Cloud" style="max-width:100%;width:600px;height:auto;"></a>


## Training in RunPod
If you would like to use Runpod, but have not signed up yet, please consider using [my Runpod affiliate link](https://runpod.io?ref=h0y9jyr2) to help support this project.


I maintain an official Runpod Pod template here which can be accessed [here](https://console.runpod.io/deploy?template=0fqzfjy6f3&ref=h0y9jyr2).

I have also created a short video showing how to get started using AI Toolkit with Runpod [here](https://youtu.be/HBNeS-F6Zz8).

## Training in Modal

### 1. Setup
#### ai-toolkit:
```
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
git submodule update --init --recursive
python -m venv venv
source venv/bin/activate
pip install torch
pip install -r requirements.txt
pip install --upgrade accelerate transformers diffusers huggingface_hub #Optional, run it if you run into issues
```
#### Modal:
- Run `pip install modal` to install the modal Python package.
- Run `modal setup` to authenticate (if this doesn’t work, try `python -m modal setup`).

#### Hugging Face:
- Get a READ token from [here](https://huggingface.co/settings/tokens) and request access to Flux.1-dev model from [here](https://huggingface.co/black-forest-labs/FLUX.1-dev).
- Run `huggingface-cli login` and paste your token.

### 2. Upload your dataset
- Drag and drop your dataset folder containing the .jpg, .jpeg, or .png images and .txt files in `ai-toolkit`.

### 3. Configs
- Copy an example config file located at ```config/examples/modal``` to the `config` folder and rename it to ```whatever_you_want.yml```.
- Edit the config following the comments in the file, **<ins>be careful and follow the example `/root/ai-toolkit` paths</ins>**.

### 4. Edit run_modal.py
- Set your entire local `ai-toolkit` path at `code_mount = modal.Mount.from_local_dir` like:
  
   ```
   code_mount = modal.Mount.from_local_dir("/Users/username/ai-toolkit", remote_path="/root/ai-toolkit")
   ```
- Choose a `GPU` and `Timeout` in `@app.function` _(default is A100 40GB and 2 hour timeout)_.

### 5. Training
- Run the config file in your terminal: `modal run run_modal.py --config-file-list-str=/root/ai-toolkit/config/whatever_you_want.yml`.
- You can monitor your training in your local terminal, or on [modal.com](https://modal.com/).
- Models, samples and optimizer will be stored in `Storage > flux-lora-models`.

### 6. Saving the model
- Check contents of the volume by running `modal volume ls flux-lora-models`. 
- Download the content by running `modal volume get flux-lora-models your-model-name`.
- Example: `modal volume get flux-lora-models my_first_flux_lora_v1`.

### Screenshot from Modal

<img width="1728" alt="Modal Traning Screenshot" src="https://github.com/user-attachments/assets/7497eb38-0090-49d6-8ad9-9c8ea7b5388b">

---

## Dataset Preparation

Datasets generally need to be a folder containing images and associated text files. Currently, the only supported
formats are jpg, jpeg, and png. Webp currently has issues. The text files should be named the same as the images
but with a `.txt` extension. For example `image2.jpg` and `image2.txt`. The text file should contain only the caption.
You can add the word `[trigger]` in the caption file and if you have `trigger_word` in your config, it will be automatically
replaced. 

Images are never upscaled but they are downscaled and placed in buckets for batching. **You do not need to crop/resize your images**.
The loader will automatically resize them and can handle varying aspect ratios. 


## Training Specific Layers

To train specific layers with LoRA, you can use the `only_if_contains` network kwargs. For instance, if you want to train only the 2 layers
used by The Last Ben, [mentioned in this post](https://x.com/__TheBen/status/1829554120270987740), you can adjust your
network kwargs like so:

```yaml
      network:
        type: "lora"
        linear: 128
        linear_alpha: 128
        network_kwargs:
          only_if_contains:
            - "transformer.single_transformer_blocks.7.proj_out"
            - "transformer.single_transformer_blocks.20.proj_out"
```

The naming conventions of the layers are in diffusers format, so checking the state dict of a model will reveal 
the suffix of the name of the layers you want to train. You can also use this method to only train specific groups of weights.
For instance to only train the `single_transformer` for FLUX.1, you can use the following:

```yaml
      network:
        type: "lora"
        linear: 128
        linear_alpha: 128
        network_kwargs:
          only_if_contains:
            - "transformer.single_transformer_blocks."
```

You can also exclude layers by their names by using `ignore_if_contains` network kwarg. So to exclude all the single transformer blocks,


```yaml
      network:
        type: "lora"
        linear: 128
        linear_alpha: 128
        network_kwargs:
          ignore_if_contains:
            - "transformer.single_transformer_blocks."
```

`ignore_if_contains` takes priority over `only_if_contains`. So if a weight is covered by both,
if will be ignored.

## LoKr Training

To learn more about LoKr, read more about it at [KohakuBlueleaf/LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS/blob/main/docs/Guidelines.md). To train a LoKr model, you can adjust the network type in the config file like so:

```yaml
      network:
        type: "lokr"
        lokr_full_rank: true
        lokr_factor: 8
```

Everything else should work the same including layer targeting.


## Support My Work

If you enjoy my projects or use them commercially, please consider sponsoring me. Every bit helps! 💖

<a href="https://ostris.com/support" target="_blank"><img src="https://ostris.com/wp-content/uploads/2025/05/support-banner2.png" alt="Support my work" style="max-width:100%;height:auto;"></a>

### Current Sponsors

All of these people / organizations are the ones who selflessly make this project possible. Thank you!!

<a href="https://ostris.com/support"><img src="https://ostris.com/sponsors.svg" alt="Sponsors" style="width:100%;height:auto;"></a>
