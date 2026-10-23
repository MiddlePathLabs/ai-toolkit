# Training share: two-GPU LoRA delta-combine (slice 1)

A second GPU trains a fixed share of the dataset each round on its own copy of the
LoRA, and the lead merges the two weight changes every round — the Fizgig v8.9
"Helpers" training-share idea, ported for two processes on one machine sharing a
folder. No NCCL, no DDP: the merge is `lead_weights + helper_delta` in fp32, both
sides having started the round from an identical snapshot.

## When to use it

Two comparable GPUs and a run you want finished sooner. Measured on this box
(PRO 6000 + 5090, a character-LoRA recipe): a share round takes ~60-90s vs ~2x that
for the same images on one GPU. It is NOT bit-equivalent to a solo run — each
side trains its half without seeing the other's intermediate steps (federated
averaging), so treat share checkpoints as needing their own eval pass.

## How it works

- Both processes run the same recipe from the same dataset with `share` blocks.
  The **lead** keeps the output folder, checkpoints and samples; the **helper**
  (its own config/name, second GPU) trains its share and posts fp32 deltas.
- The split is drawn per round from `md5(seed:round:item_path)` — deterministic
  on both sides, zero communication. The helper owns the first `frac` fraction,
  the lead the rest; both sides always own at least one item.
- Both sides iterate the FULL dataloader and skip the other's batches at fetch
  time (the `category_stop` skip pattern). Ownership changes at round boundaries.
- A round closes when the lead's share of it is trained, when a save/sample step
  forces it (the round is held open one extra step so a save lands on the round's
  own close — saves fire at step indices `% save_every == 0`, round ends after L
  steps, inherently one step apart), or at the run's end. The helper closes its
  round after its own share; if the lead closes early the helper keeps its full
  share and the next round starts from the merged point.
- Protocol files under `share.dir`: `round_%06d/start.safetensors` (lead),
  `round_%06d/job.json` (lead: round/frac/seed/n_items), `round_%06d/delta.safetensors`
  (helper), `final.json` (lead, after the last merge). Round dirs are pruned after
  each merge. A stale share dir (round_000001 present at lead start) is refused —
  delete `share.dir` before both runs.

## Step budgets — the arithmetic that keeps it comparable to a solo run

With N dataset items, helper share `frac`, helper gets `n = clamp(round(N*frac), 1, N-1)`
items per round and the lead `L = N - n`. To match a solo run of `T` steps
(T images trained), pick the round count R and set:

- `lead steps   = R * L`
- `helper steps = R * n`
- combined images = `R * N` (pick R so this ≈ T; e.g. N=56, frac=31/56 → L=25, n=31;
  R=36 → lead 900, helper 1116, combined 2016 ≈ solo 2000)
- each side's LR scheduler runs over its OWN step budget — at combined image k,
  each side sits at `cos(pi*k/(R*N))`, exactly the solo trajectory
- save/sample cadences must be multiples of L; the trainer snaps them at startup
  and logs the adjustment

Checkpoints: the lead saves at lead-step `s`; the combined-image label is
`s / L * N` (e.g. lead 225 with L=25, N=56 = combined 504).

## Config

Process-level `share` block (both configs, same `dir`/`frac`/`seed`):

```json
"share": {"enabled": true, "role": "lead", "dir": "<workspace>/share_run",
           "frac": 0.5535714285714286, "seed": 42, "timeout_s": 900,
           "adaptive": false}
```

`adaptive: true` (lead and helper both): the lead measures both sides' per-step
time each round (the helper reports steps/seconds in its delta metadata) and
moves `frac` toward equal wall-clock — the slower card gets the smaller share —
smoothed 50% and banded to ±0.15 of the configured value. The helper then takes
its split AND its learning rate from the lead's round job (per-round LR handoff,
Fizgig-style) instead of its own scheduler, and its loop budget is auto-doubled
(it stops on `final.json`). Fixed-frac runs are unchanged.

Adaptive on this box (PRO 6000 lead + 5090 helper): round 1 measured 2.21 vs
2.77 s/it, frac settled at ~0.50 within two rounds, and the lead's per-round
idle wait fell from 10–25s to ~2s.

Required (fail-fast at startup): `train.batch_size 1`, `train.gradient_accumulation 1`,
EMA off, no `datasets_reg`, at least 4 dataset items, both processes on the same
fork commit and identical dataset paths (slice 1 assumes one machine).

## Runbook

1. `nvidia-smi` — both GPUs free, no in-flight trainer jobs (a new run.py kills them).
2. Delete the share dir and any old share output dirs.
3. Start the LEAD first: `CUDA_VISIBLE_DEVICES=0 python run.py <lead.json> -l lead.log`.
4. Wait for `<share.dir>/round_000001/job.json` (model loaded, cache ready, round 1 open),
   then start the helper: `CUDA_VISIBLE_DEVICES=1 python run.py <helper.json> -l helper.log`.
5. Watch `[share] round N: combined (... round Xs, waited Ys)` lines on the lead.
   `waited` is idle time on the faster side — persistent large waits mean `frac`
   is wrong for the GPU pair (recompute budgets with a different frac).
6. Done when the lead writes `final.json`, the no-suffix `<name>.safetensors`
   completion marker appears, and the share dir contains only `final.json`.

Failure behavior: a delta/start that misses its deadline (`max(300s, 3x last
round)` capped at `timeout_s`) is absorbed by catch-up (default on — see the
roadmap section). Set `share.catch_up: false` to restore fail-fast. Mid-round
resume is not supported — delete the share dir and restart both sides. Launch
long-running sides as processes that outlive their spawning shell (a shell's
orphan cleanup can silently kill a backgrounded trainer minutes in).

## Files

- `toolkit/training_share.py` — split, delta math, file protocol, `ShareState`
- `toolkit/config_modules.py` — `ShareConfig`
- `jobs/process/BaseSDTrainProcess.py` — hooks: pre-loop setup, fetch filter,
  round close between steps (before save/sample)
- `testing/test_training_share.py` — CPU unit tests
- Workspace configs: `<workspace>/KREA_EXP_SHARE_{SOLO,LEAD,HELPER,SMOKE_*}.json`

## Deferred roadmap

**Adaptive frac — SHIPPED** (see Config above).

**Helper catch-up / degrade-to-solo — SHIPPED.** `share.catch_up` (default
true): a round whose delta misses its deadline no longer kills the run. The
lead marks the round's helper items as *owed*, folds them into its own side of
the next round's filter, and publishes them as `excluded` in the job so the
helper stays off them (nothing trains twice; a round that leaves the helper
nothing posts a zero delta instead of tripping the degenerate-split guard).
After 3 consecutive misses the lead trains solo WITHOUT sitting out further
deadlines (round closes check for an already-present delta, so a slow-but-alive
helper rejoins by delivering on time once). A transient miss is nearly
lossless — validated live: the helper missed round 1 (still loading), the lead
absorbed its 31 items (round target 25→37), the helper rejoined and round 2
merged cleanly; a solo run with the helper dead from the start finished all
150 steps with streak-3+ closes costing 0s of waiting.

Two semantics to know: the lead's step budget is FIXED (the training loop's
range is built once, so absorption displaces later lead-share rounds rather
than extending the run — a dead-helper run finishes slightly under-trained
instead of crashing), and repayment is statistical (owed items are served by
the fetch filter over the next full pass — the same wrap-around looseness that
exists in healthy rounds).

**Cross-machine transport (slice 3, designed not built):** the file protocol is
already transport-agnostic — an SMB/NFS mount of `share.dir` works as-is. The
gaps, in order of importance:
1. *Model parity check* — add the DiT/VAE/TE file hashes to round-1 job.json;
   the helper refuses on mismatch (a different build trains a different loss
   landscape; merges would be meaningless). Small, high value.
2. *Split-key normalization* — splits key on absolute dataset paths, so both
   machines must mount the dataset at the same path (works today via drive
   mapping); a `share.key_mode: relpath` option (relative path + size) removes
   the constraint if it ever annoys.
3. *Cache parity* — cache files are content-addressed but their KEYS embed
   absolute model paths, so a second machine re-encodes the dataset once into
   its own cache (correct, just slower). Shipping the lead's cache needs a
   path-rekey tool; Fizgig instead trims Krea-2 text caches to each caption's
   real tokens (~1/8 size) for transfer.
4. *Bandwidth* — per round moves ~2× the LoRA in fp32; on slow links add a
   `transport_dtype: bf16` knob (halves bytes, rounding-level effect — Fizgig
   does this under ~160 Mb/s) and raise `timeout_s`.

**Previews on the helper / move-a-run:** not assessed in depth; both are
Fizgig-proven patterns that would build on the same round-file protocol.
