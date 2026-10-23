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
           "frac": 0.5535714285714286, "seed": 42, "timeout_s": 900}
```

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

Failure behavior (slice 1): a delta/start that misses its deadline
(`max(300s, 3x last round)` capped at `timeout_s`) raises and preserves the round
dir. Mid-round resume is not supported — delete the share dir and restart both sides.

## Files

- `toolkit/training_share.py` — split, delta math, file protocol, `ShareState`
- `toolkit/config_modules.py` — `ShareConfig`
- `jobs/process/BaseSDTrainProcess.py` — hooks: pre-loop setup, fetch filter,
  round close between steps (before save/sample)
- `testing/test_training_share.py` — CPU unit tests
- Workspace configs: `<workspace>/KREA_EXP_SHARE_{SOLO,LEAD,HELPER,SMOKE_*}.json`

## Deferred (slice 2+)

Helper-failure catch-up (lead re-trains the helper's items), degrade-to-solo,
adaptive frac from measured round times, cross-machine transport (cache keys
embed absolute model paths — a second machine re-encodes once; Fizgig's
token-trimmed text-cache transfer), previews on the helper, move-a-run.
