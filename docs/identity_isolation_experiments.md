# Identity Isolation: user-run real-data experiment plan

**Do not interpret synthetic test success as evidence of better identity isolation.** This protocol is for the user to execute after the manual training checks. No real-data training, inference comparisons, face scoring, or A/B experiments are part of the implementation validation.

## Data and controls

Use consented, licensed photos of two adults, identity S1 and identity S2, with disjoint training and held-out reference sets. Keep an immutable manifest of image/caption hashes, exclusions, crops, subject identifiers, preprocessing settings, stranger asset pin, composite manifests, and exact run configurations. Keep held-out references out of both training and composite construction. Do not use generated strangers as likeness references for trained people.

Hold fixed where applicable: Krea 2/Raw checkpoint and hash, text encoder/tokenizer, VAE, LoRA rank/alpha/targets, optimizer and parameters, scheduled base LR, scheduler and warmup, batch size, accumulation, precision, resolutions/buckets, caption dropout, masks, auxiliary loss settings, training seeds, preview seeds, inference sampler/steps/CFG/resolution. Save raw training-state checkpoints and adapter exports at the same update counts. Do not silently multiply training steps after adding data.

Use at least three independent training seeds if feasible. Within each checkpoint comparison, use the same inference seeds across arms. Record successful optimizer updates and processed windows separately; skipped updates can make configured input-step counts differ from actual compute.

## Arm progression

| Arm | Subjects | Conditioning | Composites | DOP | Composite LR |
|---|---|---|---|---|---|
| A | S1 | Existing raw captions | None | Existing every-step baseline, weight1 | Normal |
| B | S1 | Fused | None | Same as A | Normal |
| C | S1 | Fused | 16 pinned strangers/S1 | Same as A | 1× |
| D | S1+S2 | Fused | 16/subject | Same as A | 1×; balanced originals |
| E | S1+S2 | Fused | 16/subject | Interval2, weight2 | 2× composites; balanced originals |

A→B tests fused conditioning. B→C adds composites while holding LR and DOP cadence fixed. **C→D is not a clean balancing ablation:** adding S2 changes both the task and exposure allocation. Use a separate two-identity pair, D-unbalanced versus D-balanced, with identical data/composites and only balancing changed.

For E, run a two-by-two factorial on the same two-identity balanced dataset: DOP every-step versus stride2/weight2, crossed with composite LR1 versus LR2. This separates stride and LR effects; the full corner alone cannot establish which mechanism helped. Include a fused-without-composites two-identity arm if testing whether subject count interacts with composites. Keep any legacy DOP baseline's actual transformation rules explicit; a multi-trigger mapping-aware baseline is not identical to historical single-trigger replacement.

## Compute and exposure accounting

**Primary comparison: fixed compute.** Choose a common wall-clock/GPU-time budget before examining results. Record updates reached, total forward/backward passes, preservation passes, samples processed, time per update, peak allocated/reserved VRAM, and total runtime. Because stride reduces extra passes, equal updates are not necessarily equal compute. Also report checkpoints at common successful update counts as a secondary view.

**Exposure-matched comparison:** where practical, set run length to match expected S1 original-photo exposures, then report observed exposures and compute. For two-identity comparisons, match both identities' original exposure allocation. Composite exposures remain a separate variable unless explicitly matched. If bucket rounding, repeated items, or category probabilities prevent exact matching, report estimates and observed counts, not an assertion of equality.

For every arm record: original images per subject, repeats, resolution variants, effective loader slots/batch-size distribution, original/composite probability, per-subject probabilities, expected and observed original/composite exposures. State whether bucket-tail samples are padded/repeated. Balancing rounds repeats; it does not imply exactly equal exposures. Adding subjects or composites at fixed updates generally reduces the original-photo exposure of an individual subject. Never conflate an exposure change with an isolation mechanism.

## Fixed prompt suite

Replace symbolic S1_FUSED/S2_FUSED with the actual configured fused strings; use raw trigger/class syntax for baseline A. Use seeds **104729, 130363, 155921, 196613** for every prompt and checkpoint. Evaluate both subject orders and both left/right placements. Keep ordinary people described before trained identities in the primary suite, matching the compositor; add reversed-order stress prompts as a separate stratum.

1. Alone: `a natural daylight close-up portrait of S1_FUSED in a park, face visible, looking at the camera`.
2. Stranger woman: `a woman with short silver hair on the left, and S1_FUSED on the right, two people side by side, two close-up portraits, both faces visible`.
3. Stranger man: `a man in his 50s with round glasses on the right, and S1_FUSED on the left, two people side by side, both faces visible`.
4. Two trained: `S1_FUSED on the left and S2_FUSED on the right, two friends at a cafe, both faces visible`.
5. No identities: `a woman with short silver hair and a man in his 50s with round glasses at a cafe, both faces visible`.
6. Unseen scene: `three friends at a garden party, an older man in the middle, S1_FUSED on the left and S2_FUSED on the right, all faces visible`.

Repeat alone/stranger prompts for S2 in two-identity arms. For classes other than woman/man, adapt ordinary-person wording consistently before starting. Separate primary two-person scores from three-person stress scores. Include prompt-adherence checks for clothing, position, requested count, and scene, so identity retention cannot hide prompt collapse.

## Face detection and assignment

Pin detector implementation/version/checkpoint, face-size threshold, confidence threshold, NMS, crop margin, alignment, and preprocessing. Use one fixed detector across all arms. Run ArcFace and SFace independently on the same accepted aligned crops, following each model's required normalization; record implementation and model hashes. Never average raw scores across embedding families as though calibrated identically.

Assign generated faces to prompt roles using requested left/right positions and scene metadata, **not the highest similarity to the expected identity**. For unconstrained layouts, use blinded human role annotation or predeclared spatial rules. A maximum-similarity assignment alone would bias likeness upward and hide leakage. Mark ambiguous assignments and report them separately. For group faces overlapping or unassignable to roles, count assignment failure rather than silently discarding the image.

For each held-out identity build a fixed reference template (normalized mean embedding) plus a per-reference similarity distribution. Report face-level cosine similarity to each template and the margin against the other identity. For an unrelated face, report similarity to **every** trained template, especially the maximum, and false-match rate against thresholds fixed using a held-out nontraining calibration set. Do not tune thresholds on the evaluation generations. Check whether a stranger crop resembles S1/S2 even when spatially assigned correctly.

A missed expected face, extra face, failed alignment, or too-small crop is a separate failure outcome. Report detection yield and assignment yield per prompt/arm; do not replace failure with zero similarity and then hide it inside an average. Report likeness on valid assigned faces alongside unconditional success rates including all requested images.

## Metrics and review

- **Likeness:** ArcFace and SFace similarity and calibrated match rate for the correct trained role, with the other identity's similarity and margin.
- **Leakage:** unrelated-role similarity to all trained subjects, calibrated false-match rate, and blinded visual leakage labels.
- **Separation:** both trained faces correctly assigned and matched, wrong-identity matches, minimum correct-versus-other margin, duplicate/same-face rate. Do not compare only the two faces' mutual embedding distance.
- **Adherence:** requested person count, left/right roles, attributes, setting, no-trigger behavior.
- **Quality:** blinded paired ratings for anatomy, face artifacts, lighting, realism, diversity, and overall image quality; report rubric and rater agreement.
- **Cost:** runtime, successful updates, main/preservation passes, peak VRAM, sample exposures, checkpoint/resume failures.

Aggregate per prompt category, identity, seed, checkpoint, then report macro means (equal weight to identities/prompt categories), medians, failure rates, and uncertainty intervals. Bootstrap at training-seed/run level when repeated outputs share a trained checkpoint; do not treat many images from one run as independent training replicates. Publish per-image results and exclusions. Use blinded randomized A/B order and prespecified checkpoint selection; if selecting a best checkpoint, use a development prompt set and evaluate once on a held-out prompt set.

**Face-embedding similarity alone is not proof of identity separation.** High trained-role similarity can coexist with copied faces, missing strangers, incorrect spatial roles, poor quality, or failed prompt adherence. Conclusions require concordant quantitative and visual evidence, with costs and exposure differences disclosed.

## Decision and next handoff

Complete the manual asset/configuration/checkpoint checks first. Then execute the prespecified arms yourself, retain manifests/configs/checkpoints/logs, and share the result tables plus representative blinded successes/failures. Until those results exist, effectiveness remains unproven; local implementation validation establishes mechanics only.
