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
  - **reference LOO band** = mean over references of each reference's mean
    cosine to the others — the natural ceiling, shown as `0.42 (refs: 0.35)`.
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
```

## Code map

- `toolkit/identity_eval/` — `detector.py` (SCRFD/YuNet), `arcface.py`,
  `sface.py` (incl. the SCRFD→YuNet landmark row conversion; landmark order
  verified by a real-model test), `scoring.py` (mean/max/LOO + filename
  parsing), `cache.py`, `refs.py` (discovery + selection), `device.py`
  (policy), `models.py` (discovery/status), `runner.py` (orchestration).
- `ui_scripts/identity_eval.py` — CLI worker (`--command refs|run`).
- `ui/src/app/api/identity-eval/` — `refs` (GET/POST), `run`, `status`,
  `cancel`, `clear` routes; shared helpers in `ui/src/server/identityEval.ts`.
- UI: `IdentityEvalPanel.tsx` (side panel), `IdentityScoreOverlay.tsx`
  (card overlay), `useIdentityEval.ts` (state/actions), pure logic in
  `ui/src/utils/identityEval.ts`; sorting + overlays integrated in
  `SampleImages.tsx` / `SampleImageCard.tsx`.

## Tests

- Python: `testing/integration/test_identity_eval.py` (pure logic always
  runs; real-model tests skip with a reason when weights/fixtures are absent).
- UI: `ui/testing/identity-eval.test.ts` (`npx sucrase-node
  testing/identity-eval.test.ts`).
- Manual checklist: run an evaluation during an active training run and
  confirm CPU was chosen and step time / VRAM are unaffected.
