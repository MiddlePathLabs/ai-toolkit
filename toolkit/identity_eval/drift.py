"""Identity drift detection (Phase 2d, fork-only).

Pure post-processing over ``results.json`` plus the Phase 2b checkpoint
report -- no face models are loaded, so this runs anywhere (CPU, during
training, offline). Hard rules from the spec:

- **Warnings, never verdicts.** The strongest thing this module says is
  "possible drift", and only for statistical signals (a bootstrap CI
  excluding zero, or a robust outlier rule). Anything stronger comes
  exclusively from the user's own review; the tool never asserts it.
- **Per model.** Every signal is computed and reported per model
  (ArcFace / SFace). The two are never averaged, merged or shared a
  scale -- same rule as the scoring stack and the retention graph.
- **Loss is never a drift signal.** The convergence panel reads the loss
  log and stays separate; drift here means identity-score behaviour only.

Each warning carries the numbers behind it (``evidence``) and filenames
(``samples``) the UI links to full-resolution samples -- never judge from
downscaled sheets.

Cut-points and margins in this module are provisional: the spec fixes
them only after the Phase 2 validation set (11.7) is labelled. Until then
everything statistical rides on bootstrap CIs, and the few absolute
constants are documented at each definition.
"""

import zlib
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .checkpoint_compare import (
    MIN_PAIRS_FOR_VERDICT,
    STATUS_EVALUATED,
    checkpoint_key,
    pair_up,
    score_of,
)

DRIFT_VERSION = 1
BOOTSTRAP_RESAMPLES = 10_000

DRIFT_SIGNAL_KINDS = (
    'similarity_decline',
    'variance_increase',
    'train_external_divergence',
    'pose_weakness',
    'narrow_generalization',
    'near_copy',
)

# Provisional pose cut-points on the eye-nose asymmetry ratio (0 = frontal,
# ~1 = full profile). Refined only via the 11.7 validation set.
POSE_FRONTAL_MAX = 0.18
POSE_PROFILE_MIN = 0.35
MIN_POSE_GROUP_N = 3
# IQR comparisons on fewer than 5 paired points are noise.
MIN_VARIANCE_PAIRS = 5
# Pooled prompt-category comparison needs at least this many samples per side.
MIN_CATEGORY_N = 6
# Near-copy margins need both a reference set big enough for mean != max to
# mean anything, and enough samples for a robust cutoff.
NEAR_COPY_REFSET_MIN = 4
NEAR_COPY_MIN_SAMPLES = 10
NEAR_COPY_MAD_SIGMA = 3.0
NEAR_COPY_MARGIN_FLOOR = 0.05
MAX_SAMPLE_LINKS = 6


def _model_display(model: str) -> str:
    return {'arcface': 'ArcFace', 'sface': 'SFace'}.get(model, model)


def _source_display(source: Optional[str]) -> str:
    return f'{source} refs' if source else 'refs'


def _fmt(value: Optional[float], digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return 'n/a'
    return f'{value:+.{digits}f}' if value < 0 else f'{value:.{digits}f}'


def _fmt_gap(value: Optional[float]) -> str:
    """Signed format for training-external gaps (negative = external above)."""
    if value is None or not np.isfinite(value):
        return 'n/a'
    return f'{value:+.3f}'


# ------------------------------------------------------------------- pose


def pose_asymmetry(kps: object) -> Optional[float]:
    """Eye-nose asymmetry ratio as a yaw proxy; ``None`` when unusable.

    The 5-point landmarks (YuNet / SCRFD order) are right eye, left eye,
    nose tip, right mouth corner, left mouth corner. Frontal faces have
    the nose horizontally centred between the eyes (ratio ~0); as the head
    turns the nose approaches one eye and the ratio grows toward 1. Only
    horizontal distances are used -- pitch moves landmarks vertically and
    must not read as turn.
    """
    if not isinstance(kps, (list, tuple)) or len(kps) < 3:
        return None
    try:
        right_eye, left_eye, nose = kps[0], kps[1], kps[2]
        d_right = abs(float(nose[0]) - float(right_eye[0]))
        d_left = abs(float(left_eye[0]) - float(nose[0]))
    except (TypeError, ValueError, IndexError):
        return None
    span = d_right + d_left
    if span <= 0:
        return None
    return abs(d_right - d_left) / span


def pose_bucket(ratio: float) -> str:
    if ratio < POSE_FRONTAL_MAX:
        return 'frontal'
    if ratio < POSE_PROFILE_MIN:
        return 'three_quarter'
    return 'profile'


def _iqr_stat(values: np.ndarray, axis: int = -1) -> np.ndarray:
    p25, p75 = np.percentile(values, [25.0, 75.0], axis=axis)
    return p75 - p25


# -------------------------------------------------------------- bootstrap


def bootstrap_diff_ci(
    values_a: Sequence[float],
    values_b: Sequence[float],
    stat_fn: Callable[[np.ndarray, int], np.ndarray],
    seed_key: str,
    *,
    paired: bool = False,
    n_boot: int = BOOTSTRAP_RESAMPLES,
    ci: float = 95.0,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for ``stat(b) - stat(a)``.

    ``paired=True`` resamples shared indices (both sides must be the same
    length; the caller pairs them by (prompt, seed)); ``paired=False``
    resamples each side independently (groups hold different samples).
    Deterministic: the RNG is seeded from ``seed_key`` (crc32) exactly like
    :func:`checkpoint_compare.bootstrap_median_ci`.
    """
    a = np.asarray(list(values_a), dtype=float)
    b = np.asarray(list(values_b), dtype=float)
    if a.size == 0 or b.size == 0 or (paired and a.size != b.size):
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(zlib.crc32(seed_key.encode('utf-8')) & 0xFFFFFFFF)
    if paired:
        indices = rng.integers(0, a.size, size=(n_boot, a.size))
        stats = stat_fn(b[indices], 1) - stat_fn(a[indices], 1)
    else:
        indices_a = rng.integers(0, a.size, size=(n_boot, a.size))
        indices_b = rng.integers(0, b.size, size=(n_boot, b.size))
        stats = stat_fn(b[indices_b], 1) - stat_fn(a[indices_a], 1)
    alpha = (100.0 - ci) / 2.0
    low, high = np.percentile(stats, [alpha, 100.0 - alpha])
    return (round(float(low), 4), round(float(high), 4))


def _median(values: Sequence[float]) -> Optional[float]:
    array = np.asarray(list(values), dtype=float)
    return round(float(np.median(array)), 3) if array.size else None


def _sample_kps(sample: dict) -> Optional[list]:
    """Landmarks of the primary face of an evaluated sample, if persisted.

    Sample faces gained ``kps`` in results v2; older documents carry only
    bbox/det_score/scores and the pose signal skips them.
    """
    if sample.get('status') != STATUS_EVALUATED:
        return None
    faces = sample.get('faces') or []
    if not faces:
        return None
    return faces[0].get('kps')


# ---------------------------------------------------------------- signals


def _signal_similarity_decline(
    groups: Dict[Tuple[int, bool], dict],
    seq: List[Tuple[int, bool]],
    step_entries: Dict[Tuple[int, bool], dict],
    pair_entries: Dict[Tuple[Tuple[int, bool], Tuple[int, bool]], dict],
    models: Sequence[str],
    sources: Sequence[str],
    signals: List[dict],
) -> None:
    """Paired median vs the highest-median earlier checkpoint, CI-gated.

    Reuses the Phase 2b paired comparisons verbatim (same CIs the UI
    shows); the only new logic is choosing the anchor step.
    """
    for model in models:
        for source in sources:
            for j, key_b in enumerate(seq[1:], start=1):
                anchor_idx = None
                anchor_median = -np.inf
                for i, key_a in enumerate(seq[:j]):
                    stats = (step_entries.get(key_a, {}).get('stats', {}).get(model, {}).get(source, {}))
                    median = stats.get('median')
                    if median is not None and median > anchor_median:
                        anchor_median, anchor_idx = median, i
                if anchor_idx is None:
                    continue
                key_a = seq[anchor_idx]
                pair = pair_entries.get((key_a, key_b))
                if not pair:
                    continue
                cmp = (pair.get('models') or {}).get(f'{model}:{source}')
                if not cmp or cmp.get('verdict') != 'lower':
                    continue
                pairs, _ = pair_up(groups[key_a], groups[key_b], model, source)
                decliners = sorted(
                    pairs, key=lambda ab: (score_of(ab[1], model, source) or 0) - (score_of(ab[0], model, source) or 0)
                )[:MAX_SAMPLE_LINKS]
                signals.append(
                    {
                        'kind': 'similarity_decline',
                        'model': model,
                        'source': source,
                        'scope': {'step': key_b[0], 'is_raw': key_b[1]},
                        'message': (
                            f"possible drift: {_model_display(model)} identity ({_source_display(source)}) at "
                            f"{step_entries[key_b]['label']} fell below the highest-median earlier checkpoint "
                            f"{step_entries[key_a]['label']} on comparable prompts "
                            f"(median difference {_fmt(cmp.get('diff_median'))}, 95% CI "
                            f"[{_fmt(cmp.get('ci_low'), 4)}, {_fmt(cmp.get('ci_high'), 4)}] excludes 0)"
                        ),
                        'evidence': {
                            'anchor_step': key_a[0],
                            'anchor_label': step_entries[key_a]['label'],
                            'diff_median': cmp.get('diff_median'),
                            'ci_low': cmp.get('ci_low'),
                            'ci_high': cmp.get('ci_high'),
                            'wins_later': cmp.get('wins_b'),
                            'wins_anchor': cmp.get('wins_a'),
                            'n_pairs': cmp.get('n_pairs'),
                        },
                        'samples': [b['filename'] for _, b in decliners if b.get('filename')],
                    }
                )


def _signal_variance_increase(
    groups: Dict[Tuple[int, bool], dict],
    seq: List[Tuple[int, bool]],
    step_entries: Dict[Tuple[int, bool], dict],
    models: Sequence[str],
    sources: Sequence[str],
    signals: List[dict],
) -> None:
    """Paired IQR growth vs the highest-median earlier checkpoint.

    Anchor matches the similarity-decline signal so the two warnings read
    as one story: "identity peaked at X; since then the median fell and
    the spread grew".
    """
    for model in models:
        for source in sources:
            for j, key_b in enumerate(seq[1:], start=1):
                anchor_idx = None
                anchor_median = -np.inf
                for i, key_a in enumerate(seq[:j]):
                    stats = (step_entries.get(key_a, {}).get('stats', {}).get(model, {}).get(source, {}))
                    median = stats.get('median')
                    if median is not None and median > anchor_median:
                        anchor_median, anchor_idx = median, i
                if anchor_idx is None:
                    continue
                key_a = seq[anchor_idx]
                pairs, _ = pair_up(groups[key_a], groups[key_b], model, source)
                if len(pairs) < MIN_VARIANCE_PAIRS:
                    continue
                values_a = [score_of(a, model, source) for a, _ in pairs]
                values_b = [score_of(b, model, source) for _, b in pairs]
                iqr_a = float(_iqr_stat(np.asarray(values_a)))
                iqr_b = float(_iqr_stat(np.asarray(values_b)))
                ci_low, ci_high = bootstrap_diff_ci(
                    values_a,
                    values_b,
                    _iqr_stat,
                    seed_key=f'variance_increase|{model}|{source}|{key_a}|{key_b}',
                    paired=True,
                )
                if not (ci_low > 0):
                    continue
                # spread outliers at the later step: farthest from its own median
                median_b = float(np.median(values_b))
                outliers = sorted(
                    pairs, key=lambda ab: -abs((score_of(ab[1], model, source) or 0) - median_b)
                )[:MAX_SAMPLE_LINKS]
                signals.append(
                    {
                        'kind': 'variance_increase',
                        'model': model,
                        'source': source,
                        'scope': {'step': key_b[0], 'is_raw': key_b[1]},
                        'message': (
                            f"possible drift: {_model_display(model)} score spread ({_source_display(source)}) grew from "
                            f"{step_entries[key_a]['label']} to {step_entries[key_b]['label']} "
                            f"(IQR {iqr_a:.3f} -> {iqr_b:.3f}; 95% CI of the paired IQR difference "
                            f"[{_fmt(ci_low, 4)}, {_fmt(ci_high, 4)}] excludes 0)"
                        ),
                        'evidence': {
                            'anchor_step': key_a[0],
                            'anchor_label': step_entries[key_a]['label'],
                            'iqr_anchor': round(iqr_a, 3),
                            'iqr_later': round(iqr_b, 3),
                            'ci_low': ci_low,
                            'ci_high': ci_high,
                            'n_pairs': len(pairs),
                        },
                        'samples': [b['filename'] for _, b in outliers if b.get('filename')],
                    }
                )


def _signal_train_external_divergence(
    groups: Dict[Tuple[int, bool], dict],
    seq: List[Tuple[int, bool]],
    step_entries: Dict[Tuple[int, bool], dict],
    pair_entries: dict,
    models: Sequence[str],
    sources: Sequence[str],
    signals: List[dict],
) -> None:
    """Training-ref scores rising while external-ref scores stay flat or fall.

    The memorization signature from the spec: the model increasingly
    matches the images it trained on without a matching gain on held-out
    references. CI-gated on both sides via the existing earliest->latest
    paired comparison.
    """
    if 'training' not in sources or 'external' not in sources:
        return
    key_first, key_last = seq[0], seq[-1]
    pair = pair_entries.get((key_first, key_last))
    if not pair:
        return
    for model in models:
        training = (pair.get('models') or {}).get(f'{model}:training')
        external = (pair.get('models') or {}).get(f'{model}:external')
        if not training or not external:
            continue
        # "Training rose" alone is not memorization: an external trend whose
        # verdict is "none" only means its CI includes zero, which also
        # covers an external median that rose MORE than training. The
        # evidence that matters is the training-external gap GROWING -- gate
        # on a paired bootstrap of the per-sample gap change so an
        # inconclusive-but-rising external trend can never fire this.
        if training.get('verdict') != 'higher':
            continue
        rows: List[Tuple[float, float]] = []
        for a, b in pair_up(groups[key_first], groups[key_last], model, 'training')[0]:
            ea = score_of(a, model, 'external')
            eb = score_of(b, model, 'external')
            if ea is None or eb is None:
                continue
            rows.append((score_of(a, model, 'training') - ea, score_of(b, model, 'training') - eb))
        if len(rows) < MIN_PAIRS_FOR_VERDICT:
            continue
        growth_median = round(float(np.median([b - a for a, b in rows])), 3)
        growth_low, growth_high = bootstrap_diff_ci(
            [a for a, _ in rows],
            [b for _, b in rows],
            np.median,
            seed_key=f'train_external_divergence|{model}|{key_first}|{key_last}',
            paired=True,
        )
        if not (growth_low > 0):
            continue
        gap_first = ((step_entries.get(key_first, {}).get('train_vs_external') or {}).get(model) or {}).get('gap')
        gap_last = ((step_entries.get(key_last, {}).get('train_vs_external') or {}).get(model) or {}).get('gap')
        # evidence samples: at the later step, the widest per-sample training-minus-external gap
        later_samples = sorted(
            groups[key_last].values(),
            key=lambda s: -(
                ((score_of(s, model, 'training') or 0) - (score_of(s, model, 'external') or 0))
            ),
        )[:MAX_SAMPLE_LINKS]
        external_word = 'fell' if external.get('verdict') == 'lower' else 'did not keep pace'
        signals.append(
            {
                'kind': 'train_external_divergence',
                'model': model,
                'source': None,
                'scope': {'step': key_last[0], 'is_raw': key_last[1]},
                'message': (
                    f"possible drift: {_model_display(model)} training-ref scores rose from "
                    f"{step_entries[key_first]['label']} to {step_entries[key_last]['label']} "
                    f"({_fmt(training.get('diff_median'))}, 95% CI [{_fmt(training.get('ci_low'), 4)}, "
                    f"{_fmt(training.get('ci_high'), 4)}]) while external-ref scores {external_word} "
                    f"({_fmt(external.get('diff_median'))}, 95% CI [{_fmt(external.get('ci_low'), 4)}, "
                    f"{_fmt(external.get('ci_high'), 4)}]) and the training-external gap grew "
                    f"({_fmt_gap(gap_first)} -> {_fmt_gap(gap_last)}; paired gap change {_fmt(growth_median)}, "
                    f"95% CI [{_fmt(growth_low, 4)}, {_fmt(growth_high, 4)}] excludes 0) -- a memorization "
                    f"signature: the model may be matching the training images rather than the identity"
                ),
                'evidence': {
                    'first_step': key_first[0],
                    'last_step': key_last[0],
                    'training_diff': training.get('diff_median'),
                    'training_ci': [training.get('ci_low'), training.get('ci_high')],
                    'external_diff': external.get('diff_median'),
                    'external_ci': [external.get('ci_low'), external.get('ci_high')],
                    'gap_first': gap_first,
                    'gap_last': gap_last,
                    'gap_growth_median': growth_median,
                    'gap_growth_ci': [growth_low, growth_high],
                    'n_pairs': len(rows),
                },
                'samples': [s['filename'] for s in later_samples if s.get('filename')],
            }
        )


def _signal_pose_weakness(
    groups: Dict[Tuple[int, bool], dict],
    step_entries: Dict[Tuple[int, bool], dict],
    models: Sequence[str],
    sources: Sequence[str],
    signals: List[dict],
    availability: Dict[str, int],
) -> None:
    """Non-frontal samples scoring below frontal ones, CI-gated per step.

    Pose comes from the stored 5-point landmarks (``pose_asymmetry``) --
    no model involved. Requires results v2 (per-face ``kps``); samples
    without landmarks are counted and reported as an availability note,
    never as a warning.
    """
    for key, group in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        buckets: Dict[str, list] = {}
        for sample in group.values():
            if sample.get('status') != STATUS_EVALUATED:
                continue
            kps = _sample_kps(sample)
            ratio = pose_asymmetry(kps)
            if ratio is None:
                if kps is None:
                    availability['missing_kps'] = availability.get('missing_kps', 0) + 1
                else:
                    availability['unusable_kps'] = availability.get('unusable_kps', 0) + 1
                continue
            availability['with_kps'] = availability.get('with_kps', 0) + 1
            buckets.setdefault(pose_bucket(ratio), []).append(sample)
        frontal = buckets.get('frontal', [])
        non_frontal = buckets.get('three_quarter', []) + buckets.get('profile', [])
        if len(frontal) < MIN_POSE_GROUP_N or len(non_frontal) < MIN_POSE_GROUP_N:
            continue
        for model in models:
            for source in sources:
                frontal_scores = [v for v in (score_of(s, model, source) for s in frontal) if v is not None]
                non_frontal_scores = [v for v in (score_of(s, model, source) for s in non_frontal) if v is not None]
                if len(frontal_scores) < MIN_POSE_GROUP_N or len(non_frontal_scores) < MIN_POSE_GROUP_N:
                    continue
                ci_low, ci_high = bootstrap_diff_ci(
                    frontal_scores,
                    non_frontal_scores,
                    np.median,
                    seed_key=f'pose_weakness|{model}|{source}|{key}',
                )
                if not (ci_high < 0):
                    continue
                weakest = sorted(non_frontal, key=lambda s: score_of(s, model, source) or 0)[:MAX_SAMPLE_LINKS]
                counts = {name: len(buckets.get(name, [])) for name in ('frontal', 'three_quarter', 'profile')}
                signals.append(
                    {
                        'kind': 'pose_weakness',
                        'model': model,
                        'source': source,
                        'scope': {'step': key[0], 'is_raw': key[1]},
                        'message': (
                            f"possible drift: {_model_display(model)} identity ({_source_display(source)}) at "
                            f"{step_entries[key]['label']} is pose-dependent -- non-frontal samples score lower than "
                            f"frontal ones (frontal median {_fmt(_median(frontal_scores))} vs non-frontal "
                            f"{_fmt(_median(non_frontal_scores))}, 95% CI [{_fmt(ci_low, 4)}, {_fmt(ci_high, 4)}])"
                        ),
                        'evidence': {
                            'pose_counts': counts,
                            'frontal_median': _median(frontal_scores),
                            'non_frontal_median': _median(non_frontal_scores),
                            'ci_low': ci_low,
                            'ci_high': ci_high,
                            'n_frontal': len(frontal_scores),
                            'n_non_frontal': len(non_frontal_scores),
                        },
                        'samples': [s['filename'] for s in weakest if s.get('filename')],
                    }
                )


def _signal_narrow_generalization(
    groups: Dict[Tuple[int, bool], dict],
    seq: List[Tuple[int, bool]],
    categories: Dict[str, str],
    models: Sequence[str],
    sources: Sequence[str],
    signals: List[dict],
) -> None:
    """One prompt category scoring far above another, pooled over checkpoints.

    Pooled per weights kind because per-step category cells are usually a
    few samples strong; pooling is also why this reads as a run-level
    property ("the model generalizes narrowly") rather than a temporal
    change -- a single checkpoint with enough tagged samples qualifies.
    Untagged prompts never form their own comparison group. ``seq`` holds
    the group keys of ONE weights kind (the caller filters).
    """

    def category_of_sample(sample: dict) -> str:
        key = checkpoint_key(sample)
        if key is None:
            return 'untagged'
        return categories.get(str(key[1]), '').strip() or 'untagged'

    weights_word = 'raw' if seq[0][1] else 'EMA'
    for model in models:
        for source in sources:
            pooled: Dict[str, List[float]] = {}
            pooled_samples: Dict[str, List[dict]] = {}
            for key in seq:
                for sample in groups[key].values():
                    cat = category_of_sample(sample)
                    if cat == 'untagged':
                        continue
                    value = score_of(sample, model, source)
                    if value is None:
                        continue
                    pooled.setdefault(cat, []).append(value)
                    pooled_samples.setdefault(cat, []).append(sample)
            qualifying = {cat: values for cat, values in pooled.items() if len(values) >= MIN_CATEGORY_N}
            if len(qualifying) < 2:
                continue
            hi_cat = max(qualifying, key=lambda c: float(np.median(qualifying[c])))
            lo_cat = min(qualifying, key=lambda c: float(np.median(qualifying[c])))
            # a=high, b=low -> the difference is low-minus-high, negative
            # when the spread is real; the CI must exclude 0 from below
            ci_low, ci_high = bootstrap_diff_ci(
                qualifying[hi_cat],
                qualifying[lo_cat],
                np.median,
                seed_key=f'narrow_generalization|{model}|{source}|{seq[0][1]}|{lo_cat}|{hi_cat}',
            )
            if not (ci_high < 0):
                continue
            weakest = sorted(pooled_samples[lo_cat], key=lambda s: score_of(s, model, source) or 0)[:MAX_SAMPLE_LINKS]
            signals.append(
                {
                    'kind': 'narrow_generalization',
                    'model': model,
                    'source': source,
                    'scope': None,
                    'message': (
                        f"possible drift: narrow generalization -- {_model_display(model)} identity "
                        f"({_source_display(source)}, {weights_word} checkpoints pooled) is high in prompt "
                        f"category '{hi_cat}' (median {_fmt(_median(qualifying[hi_cat]))}) but low in "
                        f"'{lo_cat}' (median {_fmt(_median(qualifying[lo_cat]))}); 95% CI of the median "
                        f"difference [{_fmt(ci_low, 4)}, {_fmt(ci_high, 4)}] excludes 0"
                    ),
                    'evidence': {
                        'high_category': hi_cat,
                        'high_median': _median(qualifying[hi_cat]),
                        'low_category': lo_cat,
                        'low_median': _median(qualifying[lo_cat]),
                        'ci_low': ci_low,
                        'ci_high': ci_high,
                        'n_high': len(qualifying[hi_cat]),
                        'n_low': len(qualifying[lo_cat]),
                        'weights': weights_word,
                    },
                    'samples': [s['filename'] for s in weakest if s.get('filename')],
                }
            )


def _signal_near_copy(
    groups: Dict[Tuple[int, bool], dict],
    refsets: dict,
    models: Sequence[str],
    signals: List[dict],
) -> None:
    """Samples whose closest-reference score far exceeds their mean score.

    ``max`` is the cosine against the single nearest reference, ``mean``
    the average over the set; a sample that nearly copies one training
    image separates the two. Distribution-relative cutoff (median + MAD
    robust z) instead of an absolute margin, so it self-calibrates per
    job; informational -- flagged samples are for review, not accusations.

    RAW and EMA margins are pooled into one distribution by nothing here:
    the cutoff is computed per weights kind (like every other signal --
    the two populations can sit at very different margin levels and must
    not shift each other's cutoff).
    """
    training = (refsets or {}).get('training') or {}
    if not training.get('n') or training['n'] < NEAR_COPY_REFSET_MIN:
        return
    for is_raw in (False, True):
        weights_word = 'raw' if is_raw else 'EMA'
        for model in models:
            margins: List[Tuple[dict, float]] = []
            for key, group in groups.items():
                if key[1] != is_raw:
                    continue
                for sample in group.values():
                    if sample.get('status') != STATUS_EVALUATED:
                        continue
                    entry = ((sample.get('primary') or {}).get(model) or {}).get('training') or {}
                    mean, mx = entry.get('mean'), entry.get('max')
                    if mean is None or mx is None:
                        continue
                    margins.append((sample, float(mx) - float(mean)))
            if len(margins) < NEAR_COPY_MIN_SAMPLES:
                continue
            values = np.asarray([m for _, m in margins], dtype=float)
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            cutoff = median + max(NEAR_COPY_MAD_SIGMA * 1.4826 * mad, NEAR_COPY_MARGIN_FLOOR)
            flagged = sorted([item for item in margins if item[1] > cutoff], key=lambda item: -item[1])
            if not flagged:
                continue
            signals.append(
                {
                    'kind': 'near_copy',
                    'model': model,
                    'source': 'training',
                    'scope': None,
                    'message': (
                        f"possible drift: {len(flagged)} sample(s) score far higher against their single closest "
                        f"training reference than against the reference-set average ({_model_display(model)}, "
                        f"{weights_word} checkpoints; max-minus-mean above {cutoff:.3f} vs median {median:.3f}) "
                        f"-- possible near-copies; review the linked samples at full resolution"
                    ),
                    'evidence': {
                        'cutoff': round(cutoff, 4),
                        'median_margin': round(median, 4),
                        'mad': round(mad, 4),
                        'n_flagged': len(flagged),
                        'n_samples': len(margins),
                        'training_refs': training.get('n'),
                        'weights': weights_word,
                    },
                    'samples': [s['filename'] for s, _ in flagged[:MAX_SAMPLE_LINKS] if s.get('filename')],
                }
            )


# ------------------------------------------------------------------ entry


def detect_drift(results: dict, report: dict) -> dict:
    """Scan a results document plus its Phase 2b report for drift warnings.

    Returns ``{'version', 'signals', 'notes'}``. Signals are per model --
    never averaged across ArcFace/SFace; each carries evidence numbers and
    sample filenames for the UI's links.
    """
    samples = results.get('samples') or []
    models = report.get('models') or []
    sources = report.get('sources') or []
    categories = {str(k): v for k, v in (report.get('categories') or {}).items()}

    groups: Dict[Tuple[int, bool], Dict[Tuple[int, int, int, bool], dict]] = {}
    for sample in samples:
        key = checkpoint_key(sample)
        if key is None:
            continue
        step, _, _, is_raw = key
        groups.setdefault((step, is_raw), {})[key] = sample
    ordered_keys = sorted(groups, key=lambda g: (g[0], g[1]))
    step_entries = {(e.get('step'), e.get('is_raw')): e for e in (report.get('steps') or [])}
    pair_entries = {
        ((p.get('a', {}).get('step'), p.get('a', {}).get('is_raw')), (p.get('b', {}).get('step'), p.get('b', {}).get('is_raw'))): p
        for p in (report.get('pairs') or [])
    }

    signals: List[dict] = []
    notes: List[str] = []
    availability: Dict[str, int] = {}

    for is_raw in (False, True):
        seq = [k for k in ordered_keys if k[1] == is_raw]
        if not seq:
            continue
        # narrow generalization compares prompt categories, not steps -- a
        # single checkpoint with enough tagged samples already qualifies
        _signal_narrow_generalization(groups, seq, categories, models, sources, signals)
        if len(seq) < 2:
            continue
        _signal_similarity_decline(groups, seq, step_entries, pair_entries, models, sources, signals)
        _signal_variance_increase(groups, seq, step_entries, models, sources, signals)
        _signal_train_external_divergence(groups, seq, step_entries, pair_entries, models, sources, signals)

    if not any(v.strip() and v.strip() != 'untagged' for v in categories.values()):
        notes.append(
            'Narrow-generalization signal idle: no prompt categories tagged (tag prompts below to enable it).'
        )

    _signal_pose_weakness(groups, step_entries, models, sources, signals, availability)
    _signal_near_copy(groups, report.get('refsets') or {}, models, signals)

    with_kps = availability.get('with_kps', 0)
    missing_kps = availability.get('missing_kps', 0)
    if with_kps == 0 and missing_kps > 0:
        notes.append(
            'Pose-dependent scoring signal unavailable: results.json predates per-face landmark persistence '
            '(re-run the evaluation to enable it).'
        )
    elif missing_kps > 0:
        notes.append(
            f'Pose-dependent scoring signal covered {with_kps} of {with_kps + missing_kps} evaluated samples; '
            f'the rest predate landmark persistence.'
        )
    if availability.get('unusable_kps'):
        notes.append(f"{availability['unusable_kps']} sample face(s) had unreadable landmarks and were skipped by the pose signal.")
    notes.append(
        "Drift warnings are statistical suggestions ('possible drift'); anything stronger comes only from your "
        'own review, never from this tool. Loss-based convergence is reported separately and is never an '
        'identity signal.'
    )
    notes.append('Signals are computed and shown per model; ArcFace and SFace are never averaged or merged.')

    return {'version': DRIFT_VERSION, 'signals': signals, 'notes': notes}
