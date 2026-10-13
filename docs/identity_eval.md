# Identity Evaluation (fork-only)

Scores generated sample faces against reference faces on the **Samples** screen
("Identity…" button). This is a fork-only feature of the MiddlePathLabs
AI-Toolkit; it is not intended for upstream.

Two independent recognition models score every face; their raw cosines are
shown side by side and are **never averaged or merged**:

| Model | Embedding | Source | License |
|---|---|---|---|
| ArcFace `w600k_r50` (+ SCRFD `det_10g` detection) | 512-d | `~/.insightface/models/buffalo_l/` | InsightFace non-commercial research terms |
| SFace `face_recognition_sface_2021dec` | 128-d | `models/face/` (user-supplied) | Apache-2.0 (opencv_zoo) |
| YuNet `face_detection_yunet_2023mar` (optional detector for SFace-only mode) | — | `models/face/` (user-supplied) | MIT (opencv_zoo) |

**SFace-only mode is the permissive option**: if `buffalo_l` is absent but
SFace + YuNet are installed, detection runs on YuNet and scoring on SFace only.
If one recognition model is missing the feature runs with the other; the UI
shows which models are active. If both are missing the worker fails with a
clear error naming the expected paths.

## Licensing rules

- No model weights are committed, bundled, or auto-downloaded by this code.
  `insightface`'s download path is never used (model files are pre-checked on
  disk before any insightface object is constructed).
- `buffalo_l` weights must be installed by the user into the insightface home
  (`~/.insightface/models/buffalo_l`). They are the same weights the trainer's
  face_id feature uses. Non-commercial research terms — fork-only.
- SFace and YuNet go in `AI-Toolkit/models/face/` (configurable via the
  `IDENTITY_MODELS_DIR` settings key). Keep the Apache-2.0/ MIT attribution if
  the files are ever redistributed.

## Method

- Detection: SCRFD `det_10g` at `det_size=(640,640)`, default threshold 0.5.
  Faces are ordered by bbox area (descending); that order defines the stable
  face index used by picks in `selection.json`. The legacy eval scripts'
  biggest-face-is-the-subject shortcut is deliberately **not** used.
- Images are fed to insightface exactly as `cv2.imread` returns them (BGR).
  insightface swaps to RGB internally (`blobFromImage(..., swapRB=True)`).
  The legacy `Scripts/*_eval.py` fed channel-swapped RGB and lost ~9%
  same/impostor separation — their historical numbers are **not comparable**
  to this feature's numbers. A regression test pins the BGR path.
- Per model, per reference source (training / external / combined):
  - **primary** = `mean(R @ g)`, the mean of per-reference raw cosines
    (same scale as the existing eval scripts);
  - **max** = `max(R @ g)` (detail view; a high max with a low mean can
    indicate a near-copy of one training image);
  - **reference LOO band** (leave-one-out) = mean over references of each
    reference's mean cosine to the others — how similar real photos of the
    person are to each other; a same-person baseline, not an upper bound (a
    generated face that captures the shared identity can score above the
    band, and strong checkpoints often do); shown as `0.42 (refs: 0.35)`.
- Raw cosines, 3 decimals, no percentages. The trainer's identity loss
  subtracts a noise-mean direction (bias-corrected); this UI shows **raw**
  cosine, so trainer-logged identity values are not comparable to these.
- ArcFace and SFace scales differ. Disagreement is surfaced as a rank-based
  flag (ranks differing by more than a third of the evaluated set), never as
  a cross-model threshold.

## References

- **Training refs** are discovered from the job config: datasets with
  `is_reg`/`prior_reg` false, `dataset_path` when set (trainer precedence)
  else `folder_path`; a divergent pair surfaces the same conflict the trainer
  fail-fasts on. Control/mask folders are excluded. The dataloader's image
  extensions are accepted (`.jpg .jpeg .png .webp`). json dataset files are
  reported as unsupported.
- **External refs** are user-uploaded images stored content-addressed
  (sha256 filenames) under `output/<job>/identity_eval/external_refs/` to
  dedupe. External-ref similarity tests generalization; training-ref
  similarity partly measures memorization.
- **Multi-face references are excluded by default** (marked "needs review").
  The user picks the subject's face from thumbnails; the pick is stored in
  `selection.json`. Single-face references are accepted automatically;
  no-face references are marked and never scored as 0.

## Samples

- One face → `evaluated`, primary score on the card overlay (`A 0.420 · S 0.381`).
- Several faces → `ambiguous` (per-face scores in the detail data; a subject
  face picker is Phase 2). Excluded from summary medians.
- No face / unreadable → `failed` with the reason; never scored as 0.

## Execution and GPU policy

The worker (`ui_scripts/identity_eval.py`, thin CLI over
`toolkit/identity_eval/`) is spawned by the API routes as its own process —
all memory is freed on exit. Progress is JSON-lines on stdout plus
`progress.json` for polling; cancel is a process kill and partial
`results.json` stays valid (it is written incrementally and atomically).

One active evaluation per job; a second start returns the running one.

Device selection (`auto` default, `gpu`/`cpu` override):

1. A job is running on the job's GPU (SQLite `Job.status` + `gpu_ids`) —
   **including the evaluated job itself** (watching a training job's samples
   mid-run is the normal case) → CPU.
2. Else > 4 GB free VRAM on that GPU (nvidia-smi) → CUDA, with the ORT arena
   capped (~2 GB, `arena_extend_strategy=kSameAsRequested`) **on that same
   GPU id** — the policy-checked device and the session device always agree.
3. Else CPU. Unknown GPU state → CPU (conservative).
4. Any CUDA init/OOM → CPU fallback recorded as a warning in the result.

Forcing GPU while training runs asks for confirmation in the UI.
`CUDA_DEVICE_ORDER=PCI_BUS_ID` is set so nvidia-smi indices match ORT device
ids. SFace runs on CPU unless the installed OpenCV build actually has the
CUDA DNN target (checked, not assumed — pip wheels usually don't); OpenCV's
CUDA target cannot select a device, so on multi-GPU hosts with a non-zero
selected GPU, SFace stays on CPU with a recorded note.

Selection keys for training references use one canonical form everywhere
(panel writes, UI reads, worker lookups): lowercase, forward-slash,
normalized paths (`normalizeRefPath` in the UI, `canonical_ref_key` in the
worker). Installing a recognition model after discovery is self-healing:
the run detects reference faces missing the new model's embeddings and
re-runs discovery (cache-first) before scoring.

## Storage

All under `output/<job>/identity_eval/` (never `samples/`, never Prisma):

```
refs.json          discovery output (per-image face status, models)
selection.json     user exclusions + face picks
external_refs/     user-added reference images (sha256-named)
emb_cache/         per-image cache: key = image sha + detector sig
                   (weights hash + det size + order version); each face keeps
                   per-model embeddings under that model's own file hash
results.json       per-sample scores (partial-valid, written incrementally)
progress.json      live progress for the status route
run_state.json     pid + start time of the active worker (route-managed)
run.log            worker stdout/stderr for debugging
checkpoint_report.json   Phase 2b: per-checkpoint stats + paired comparisons
prompt_categories.json   Phase 2b: user prompt-index category tags (per job)
```

## Checkpoint comparison (Phase 2b)

Compares checkpoints within a job **without fooling you with non-comparable
sample sets**. It is pure post-processing over `results.json` — no face
models load, so it is safe at any time, including during training.

- **Grouping:** by step from the filename; `RAW_` dual-save samples form
  separate raw/EMA groups at the same step and never mix.
- **Comparability:** direct comparison uses only the intersection of
  (prompt index, seed, weights-kind) pairs present at both steps. Everything
  else is listed per pair with a reason (`missing_at_a/b` = config or seed
  change, `not_evaluated` = no face / ambiguous / failed). Prompt texts are
  snapshotted from the job config at report time; mid-run sample-config
  changes cannot be recovered from filenames (stated in the report notes).
- **Per checkpoint, per model, per reference source:** median, p10, IQR, a
  distribution strip of all evaluated samples, status counts
  (evaluated / ambiguous / no-face / failed) and the training-vs-external
  median gap (a rising gap can indicate memorization).
- **Paired comparison:** per (prompt, seed) score differences; median
  difference with a deterministic percentile-bootstrap CI (10k resamples)
  and a win count. A checkpoint is only called "higher identity" when the
  CI excludes zero; otherwise the wording is "no detectable difference".
  **There is no overall best-checkpoint verdict, by design.**
- **Prompt categories:** tag prompt indices (e.g. solo / multi /
  gate-no-trigger) in the panel; medians split per category because these
  prompt kinds behave very differently. Tags persist per job in
  `prompt_categories.json`.
- **Convergence panel:** a loss-based "would more steps help?" readout
  (250-step bucket means, late-500 change vs bucket noise, LR floor — the
  core heuristic of the workspace-level `krea2_loss_progress.py`), shown as
  a clearly separate panel when `output/<job>/loss_log.db` exists. Loss is
  never mixed into identity numbers.

The report is recomputed by the `compare` worker command whenever
`results.json` or the category tags change; the API route serves the cached
`checkpoint_report.json` otherwise. ArcFace and SFace are never averaged or
share a scale anywhere in the report.

## Identity retention graph (Phase 2c)

Compact uPlot chart at the top of the checkpoint section: median primary
score per training step over the whole evaluation, so you can see where
likeness rises, plateaus or fades across the run.

- **Two series, two scales, always:** ArcFace on its own left axis, SFace on
  its own right axis. The models' cosine scales are incomparable, so a shared
  y-axis would imply an equivalence that does not exist; there is also **no
  composite series** (Phase 2a calibration has not shipped, so no combined
  score is drawn anywhere).
- **Band:** shaded 10th–90th percentile of the sample distribution at each
  step (`distribution_stats` computes p90 for this; report version 2). Hover
  anywhere to read the step, both models' medians, n per point, the band and
  the LOO value.
- **Sparse points:** steps with fewer than 5 evaluated samples draw as hollow
  rings — the median is noise-level there.
- **LOO guide:** a dashed horizontal line per model at the reference
  self-similarity (leave-one-out) value for the selected reference source,
  labeled in the legend.
- **Filters:** reference source follows the section's training/external
  selector; a prompt-category filter (from the tags below) and, on dual-save
  runs, an EMA/raw weights selector — raw and EMA are never mixed into one
  line.
- **Click a point** to filter the Samples grid to that step; click a sample
  in the grid or a strip dot to open it full-size (`SampleImageViewer`).
- Data comes entirely from the cached checkpoint report (pure post-processing,
  no face models); the compare route rebuilds any report older than the
  current `REPORT_VERSION` so v1 caches lacking p90 are refreshed, not served.

## Identity drift warnings (Phase 2d)

An amber block between "How to read this" and the retention graph surfaces
**statistical warning signs** from the checkpoint data. Warnings, never
verdicts: the strongest wording the tool ever produces is "possible drift",
and only for a statistical signal (a bootstrap CI excluding zero, or a robust
outlier rule). Anything stronger ("confirmed by review") is exclusively
user-marked — the tool never asserts it. Every warning carries the numbers
behind it plus numbered buttons that open the linked samples at full
resolution; never judge from downscaled sheets.

All signals are computed **per model and shown per model** (the section's
ArcFace/SFace and source selectors filter the block; a muted line counts the
warnings hidden by the filter). ArcFace and SFace are never averaged, merged
or given a shared threshold — same rule as everywhere else. The loss-based
convergence panel stays separate: loss is never an identity signal and never
a drift signal.

The six signals (`toolkit/identity_eval/drift.py`, pure post-processing, no
face models — pose comes from stored landmarks):

| Kind | Fires when | Method |
|---|---|---|
| `similarity_decline` | a later checkpoint scores below the highest-median earlier checkpoint | reuses the Phase 2b paired comparison; fires only when its bootstrap CI excludes 0 |
| `variance_increase` | score spread (IQR) grows from the peak checkpoint | paired bootstrap of the IQR difference, CI excludes 0 (≥ 5 pairs) |
| `train_external_divergence` | training-ref scores rise AND the training−external gap widens | training CI excludes 0 upward plus a paired bootstrap of the per-sample gap change excluding 0 — an inconclusive external trend (CI straddling 0) alone never fires it |
| `pose_weakness` | non-frontal samples score below frontal ones at a step | pose from the 5-point landmarks (eye–nose asymmetry ratio; no model), independent bootstrap of the median difference |
| `narrow_generalization` | one prompt category scores far above another | pooled per weights kind, independent bootstrap (≥ 2 tagged categories, ≥ 6 samples each; a single checkpoint qualifies — the signal is not temporal) |
| `near_copy` | a sample's closest-reference score far exceeds its mean-reference score | `max − mean` margin above a robust cutoff (median + 3·MAD, floor 0.05, refset ≥ 4), computed **per weights kind** — RAW and EMA margins never share a cutoff; informational |

Cut-points (pose buckets frontal < 0.18 / three-quarter < 0.35 / profile on
the asymmetry ratio, the near-copy floor, category minimums) are **provisional
pending the Phase 2 validation set** (11.7); everything statistical rides on
bootstrap CIs, which are deterministic (crc32-seeded like Phase 2b).

Requirements and limits:

- **Pose signal needs results v2**: sample faces must carry per-face `kps`
  (landmarks are persisted since results version 2; older documents predate
  them). Evaluations from before that skip the pose signal and get an
  availability note — re-run the evaluation to enable it. The landmarks come
  from the detector at scoring time; the drift pass never loads a model.
- RAW_/EMA weights never mix (each signal runs within one weights kind).
- Report version 3: the drift block ships inside `checkpoint_report.json`
  under `drift` (`version`, `signals`, `notes`); the compare route rebuilds
  older cached reports.

## Code map

- `toolkit/identity_eval/` — `detector.py` (SCRFD/YuNet), `arcface.py`,
  `sface.py` (incl. the SCRFD→YuNet landmark row conversion; landmark order
  verified by a real-model test), `scoring.py` (mean/max/LOO + filename
  parsing), `cache.py`, `refs.py` (discovery + selection), `device.py`
  (policy), `models.py` (discovery/status), `runner.py` (orchestration),
  `checkpoint_compare.py` (Phase 2b report builder + convergence heuristic),
  `drift.py` (Phase 2d signals + pose estimate + generic bootstrap).
- `ui_scripts/identity_eval.py` — CLI worker (`--command refs|run|compare`).
- `ui/src/app/api/identity-eval/` — `refs` (GET/POST), `run`, `status`,
  `cancel`, `clear`, `compare` (GET report / POST category tags) routes;
  shared helpers in `ui/src/server/identityEval.ts`.
- UI: `IdentityEvalPanel.tsx` (side panel), `IdentityScoreOverlay.tsx`
  (card overlay), `IdentityCheckpointSection.tsx` (Phase 2b section),
  `IdentityRetentionGraph.tsx` (Phase 2c chart), `useIdentityEval.ts`
  (state/actions), pure logic in `ui/src/utils/identityEval.ts` and
  `ui/src/utils/identityCheckpoint.ts` (retention-series builder included);
  sorting + overlays integrated in `SampleImages.tsx` / `SampleImageCard.tsx`.

## Tests

- Python: `testing/integration/test_identity_eval.py` (pure logic always
  runs; real-model tests skip with a reason when weights/fixtures are absent),
  `testing/integration/test_identity_checkpoint_compare.py` (pairing,
  bootstrap CIs, categories, convergence, CLI — always runs) and
  `testing/integration/test_identity_drift.py` (pose fixtures, each signal's
  fire/control cases, determinism, wording rules, per-model separation).
- UI: `ui/testing/identity-eval.test.ts` and
  `ui/testing/identity-checkpoint.test.ts` (`npx sucrase-node
  testing/<file>.test.ts`); the latter covers the retention-series builder
  (per-model separation, hollow flags for n < 5, band arrays, raw/EMA and
  category filters) and the drift helpers (filtering, hidden counts, kind
  labels that never invent verdicts).
- Manual checklist: run an evaluation during an active training run and
  confirm CPU was chosen and step time / VRAM are unaffected — verified
  live, user-confirmed 2026-10-13.
