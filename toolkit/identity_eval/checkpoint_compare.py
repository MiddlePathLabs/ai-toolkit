"""Checkpoint comparison for identity evaluation (Phase 2b, fork-only).

Pure post-processing over ``results.json`` -- no face models are loaded, so
this runs anywhere (CPU, during training, offline). The report it builds
follows two hard rules from the spec:

- **Comparability:** checkpoints are compared directly only on the
  intersection of (prompt index, seed, weights-kind) pairs present at both
  steps. Pairs outside the intersection are listed with a reason, never
  silently dropped.
- **No overall "best checkpoint" verdict.** The report states per-pair
  evidence (median difference, bootstrap CI, win count). The only per-pair
  word used is "higher"/"lower" identity *when the CI excludes zero*, and
  "no detectable difference" otherwise.

Grouping key is ``(step, prompt_idx, seed, is_raw)``: RAW_ prefix samples
(dual-save raw/EMA) are separate checkpoints from the un-prefixed ones at the
same step and are never mixed into one comparison.

Prompt texts are snapshotted from the job config *at report time*. If the
sample config changed mid-run the historical mapping cannot be recovered
from filenames alone; that limitation is stated in the report notes. Seed
changes ARE detectable (the pair simply stops matching) and surface as
excluded pairs.
"""

import datetime
import json
import os
import zlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# v2: distribution_stats gained p90 (retention graph band, Phase 2c). The
# compare route pins this version so a cached v1 report is rebuilt, never
# served with missing p90 fields.
# v3: the report gained the Phase 2d drift block (warnings, never verdicts)
# computed by ``drift.detect_drift`` over the same results document.
# v4: Phase 2e leakage -- sample subject-face picks resolve ambiguous
# samples into the stats/comparisons, steps gain per-model leakage rates,
# and the report carries the ``leakage`` block (``leakage.detect_leakage``).
REPORT_VERSION = 4
BOOTSTRAP_RESAMPLES = 10_000
MIN_PAIRS_FOR_VERDICT = 3

STATUS_EVALUATED = 'evaluated'
STATUS_AMBIGUOUS = 'ambiguous'
STATUS_FAILED = 'failed'
NO_FACE_WARNING = 'no face detected'


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec='seconds')


# ------------------------------------------------------------------ grouping


def checkpoint_key(sample: dict) -> Optional[Tuple[int, int, int, bool]]:
    """``(step, prompt_idx, seed, is_raw)`` for a results.json sample record.

    None when the filename could not be parsed (step/prompt/seed unknown) --
    those samples count toward statuses but never join comparisons.
    """
    step = sample.get('step')
    prompt_idx = sample.get('prompt_idx')
    seed = sample.get('seed')
    if step is None or prompt_idx is None or seed is None:
        return None
    return (int(step), int(prompt_idx), int(seed), bool(sample.get('is_raw')))


def pair_id_str(key: Tuple[int, int, int, bool]) -> str:
    step, prompt_idx, seed, is_raw = key
    return f'step {step} · prompt {prompt_idx} · seed {seed}' + (' · raw' if is_raw else '')


def group_label(step: int, is_raw: bool, has_raw: bool) -> str:
    """Human label; the un-prefixed series is only called out when both exist."""
    if not has_raw:
        return str(step)
    return f'{step} (raw)' if is_raw else f'{step} (EMA)'


def count_statuses(samples: Sequence[dict]) -> Dict[str, int]:
    """evaluated / ambiguous / no_face / failed.

    Phase 1 folds no-face detections into ``failed`` with a warning; the
    checkpoint table splits them back out for readable counts.
    """
    counts = {STATUS_EVALUATED: 0, STATUS_AMBIGUOUS: 0, 'no_face': 0, STATUS_FAILED: 0}
    for sample in samples:
        status = sample.get('status')
        if status == STATUS_FAILED and any(NO_FACE_WARNING in w for w in sample.get('warnings') or []):
            counts['no_face'] += 1
        elif status in counts:
            counts[status] += 1
        else:
            counts[STATUS_FAILED] += 1
    return counts


def score_of(sample: dict, model: str, source: str) -> Optional[float]:
    if sample.get('status') != STATUS_EVALUATED:
        return None
    scores = (sample.get('primary') or {}).get(model) or {}
    entry = scores.get(source) or {}
    value = entry.get('mean')
    return float(value) if value is not None else None


def distribution_stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    """median / p10 / p90 / IQR for a score distribution; empty input -> all None.

    p10/p90 feed the retention graph's shaded band (Phase 2c); the paired
    comparisons never read them.
    """
    if not values:
        return {'n': 0, 'median': None, 'p10': None, 'p25': None, 'p75': None, 'p90': None, 'iqr': None}
    array = np.asarray(values, dtype=float)
    p10, p25, p50, p75, p90 = np.percentile(array, [10, 25, 50, 75, 90])
    return {
        'n': int(array.size),
        'median': round(float(p50), 3),
        'p10': round(float(p10), 3),
        'p25': round(float(p25), 3),
        'p75': round(float(p75), 3),
        'p90': round(float(p90), 3),
        'iqr': round(float(p75 - p25), 3),
    }


def available_sources(samples: Sequence[dict]) -> List[str]:
    """Sources that actually carry scores in the results."""
    sources = []
    for source in ('training', 'external', 'combined'):
        if any(score_of(s, model, source) is not None for s in samples for model in ('arcface', 'sface')):
            sources.append(source)
    return sources


def available_models(samples: Sequence[dict]) -> List[str]:
    return [
        m
        for m in ('arcface', 'sface')
        if any(score_of(s, m, source) is not None for s in samples for source in ('training', 'external', 'combined'))
    ]


# ------------------------------------------------------------------ pairing


def bootstrap_median_ci(
    diffs: np.ndarray,
    seed_key: str,
    n_boot: int = BOOTSTRAP_RESAMPLES,
    ci: float = 95.0,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for the median of paired differences.

    Deterministic: the RNG is seeded from ``seed_key`` (crc32) so the same
    pair always produces the same interval on re-runs.
    """
    if diffs.size == 0:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(zlib.crc32(seed_key.encode('utf-8')) & 0xFFFFFFFF)
    n = diffs.size
    indices = rng.integers(0, n, size=(n_boot, n))
    medians = np.median(diffs[indices], axis=1)
    alpha = (100.0 - ci) / 2.0
    low, high = np.percentile(medians, [alpha, 100.0 - alpha])
    return (round(float(low), 4), round(float(high), 4))


def pair_up(
    group_a: Dict[Tuple[int, int, int, bool], dict],
    group_b: Dict[Tuple[int, int, int, bool], dict],
    model: str,
    source: str,
) -> Tuple[List[Tuple[dict, dict]], List[dict]]:
    """Match samples of two checkpoints on (prompt, seed, is_raw).

    The step is dropped from the matching key -- it is what differs between
    the two groups. Returns ``(pairs, excluded)``. ``pairs`` holds samples
    that are evaluated with a score for ``model``/``source`` on both sides.
    ``excluded`` lists every other shared-or-unshared key with a machine
    reason:

    - ``missing_at_a`` / ``missing_at_b``: the (prompt, seed) combo was not
      sampled at that step (config or seed change)
    - ``not_evaluated``: sampled at both steps but not evaluable on both
      (ambiguous / no face / failed / missing model score)
    """
    pairs: List[Tuple[dict, dict]] = []
    excluded: List[dict] = []
    b_by_pair = {key[1:]: sample for key, sample in group_b.items()}
    a_pairs = {key[1:] for key in group_a}
    for key, sample_a in group_a.items():
        sample_b = b_by_pair.get(key[1:])
        if sample_b is None:
            excluded.append({'id': pair_id_str(key), 'reason': 'missing_at_b'})
            continue
        score_a = score_of(sample_a, model, source)
        score_b = score_of(sample_b, model, source)
        if score_a is None or score_b is None:
            excluded.append(
                {
                    'id': pair_id_str(key),
                    'reason': 'not_evaluated',
                    'status_a': sample_a.get('status'),
                    'status_b': sample_b.get('status'),
                }
            )
            continue
        pairs.append((sample_a, sample_b))
    for key, sample_b in group_b.items():
        if key[1:] not in a_pairs:
            excluded.append({'id': pair_id_str(key), 'reason': 'missing_at_a'})
    return pairs, excluded


def compare_pair(
    group_a: Dict[Tuple[int, int, int, bool], dict],
    group_b: Dict[Tuple[int, int, int, bool], dict],
    models: Sequence[str],
    sources: Sequence[str],
) -> Dict[str, dict]:
    """Per (model, source) paired comparison of checkpoint B minus A.

    Verdict words: ``higher`` / ``lower`` only when the bootstrap CI of the
    median difference excludes zero (and enough pairs exist); ``none``
    otherwise; ``insufficient`` below :data:`MIN_PAIRS_FOR_VERDICT` pairs.
    """
    out: Dict[str, dict] = {}
    for model in models:
        for source in sources:
            pairs, excluded = pair_up(group_a, group_b, model, source)
            diffs = np.asarray(
                [score_of(b, model, source) - score_of(a, model, source) for a, b in pairs],
                dtype=float,
            )
            wins_b = int(np.sum(diffs > 0))
            wins_a = int(np.sum(diffs < 0))
            if diffs.size:
                median = round(float(np.median(diffs)), 3)
                ci_low, ci_high = bootstrap_median_ci(diffs, seed_key=f'{model}|{source}|{sorted(group_a)}|{sorted(group_b)}')
            else:
                median, ci_low, ci_high = None, float('nan'), float('nan')
            if len(pairs) < MIN_PAIRS_FOR_VERDICT:
                verdict = 'insufficient'
            elif ci_low > 0:
                verdict = 'higher'
            elif ci_high < 0:
                verdict = 'lower'
            else:
                verdict = 'none'
            out[f'{model}:{source}'] = {
                'n_pairs': len(pairs),
                'diff_median': median,
                'ci_low': None if np.isnan(ci_low) else ci_low,
                'ci_high': None if np.isnan(ci_high) else ci_high,
                'wins_b': wins_b,
                'wins_a': wins_a,
                'ties': int(np.sum(diffs == 0)),
                'verdict': verdict,
                'excluded': excluded,
            }
    return out


# ------------------------------------------------------------------ report


def extract_prompt_texts(job_config: Optional[dict]) -> Dict[int, str]:
    """prompt index -> text from a job config (DB envelope or inner config).

    Accepts both the legacy ``sample.prompts`` list and the ``sample.samples``
    list of {prompt, ...} items. Best-effort: anything unreadable yields {}.
    """
    if not isinstance(job_config, dict):
        return {}
    config = job_config.get('config') if isinstance(job_config.get('config'), dict) else job_config
    sample = config.get('sample') if isinstance(config, dict) else None
    if not isinstance(sample, dict):
        return {}
    texts: Dict[int, str] = {}
    prompts = sample.get('prompts')
    if isinstance(prompts, list):
        for idx, text in enumerate(prompts):
            if isinstance(text, str) and text.strip():
                texts[idx] = text.strip()
        return texts
    items = sample.get('samples')
    if isinstance(items, list):
        for idx, item in enumerate(items):
            if isinstance(item, dict) and isinstance(item.get('prompt'), str) and item['prompt'].strip():
                texts[idx] = item['prompt'].strip()
    return texts


def category_of(prompt_idx: int, categories: Dict[str, str]) -> str:
    return categories.get(str(prompt_idx), '').strip() or 'untagged'


def build_checkpoint_report(
    results: dict,
    categories: Optional[Dict[str, str]],
    prompt_texts: Optional[Dict[int, str]],
    convergence: Optional[dict],
    picks: Optional[dict] = None,
) -> dict:
    """Assemble the full checkpoint report from a results.json document.

    ``picks`` is the ``sample_picks`` section of selection.json (Phase 2e):
    subject-face picks that resolve ambiguous multi-face samples into
    evaluated ones *before* any grouping, so stats, strips, paired
    comparisons and drift all see the same resolved view. The leakage
    block is computed from the same inputs.
    """
    # Phase 2e lazy imports (leakage, like drift, reads helpers from this
    # module; a top-level import would be circular)
    from .leakage import detect_leakage, resolve_picks

    resolved_samples, picks_used = resolve_picks(results.get('samples') or [], picks)
    # detect_leakage re-resolves picks itself and must see the ORIGINAL
    # statuses (resolution flips ambiguous -> evaluated, which would hide
    # the picks from its own valid_pick lookup)
    raw_results = results
    results = {**results, 'samples': resolved_samples}
    samples = resolved_samples
    sources = available_sources(samples)
    models = available_models(samples)
    categories = {str(k): v for k, v in (categories or {}).items()}

    # group samples by (step, is_raw); keyed maps keep per-pair lookup fast
    groups: Dict[Tuple[int, bool], Dict[Tuple[int, int, int, bool], dict]] = {}
    unparsed = 0
    for sample in samples:
        key = checkpoint_key(sample)
        if key is None:
            unparsed += 1
            continue
        step, _, _, is_raw = key
        groups.setdefault((step, is_raw), {})[key] = sample
    ordered_keys = sorted(groups, key=lambda g: (g[0], g[1]))
    has_raw = any(is_raw for _, is_raw in ordered_keys)

    step_entries = []
    leakage_block = detect_leakage(raw_results, picks, models, sources)
    step_leakage = {(item['step'], item['is_raw']): item['rates'] for item in leakage_block['per_step']}
    for group_key in ordered_keys:
        step, is_raw = group_key
        bucket = list(groups[group_key].values())
        counts = count_statuses(bucket)
        picked_here = sum(1 for s in bucket if s.get('picked_subject'))
        if picked_here:
            counts['picked'] = picked_here
        entry = {
            'step': step,
            'is_raw': is_raw,
            'label': group_label(step, is_raw, has_raw),
            'counts': counts,
            'n_prompts': len({key[1] for key in groups[group_key]}),
            'stats': {
                model: {source: distribution_stats([v for v in (score_of(s, model, source) for s in bucket) if v is not None])
                        for source in sources}
                for model in models
            },
            'strip': {
                model: {
                    source: [
                        {'f': s['filename'], 's': score_of(s, model, source)}
                        for s in sorted(bucket, key=lambda s: s.get('filename') or '')
                        if score_of(s, model, source) is not None
                    ]
                    for source in sources
                }
                for model in models
            },
        }
        # training vs external medians side by side + their gap
        if 'training' in sources and 'external' in sources:
            gaps = {}
            for model in models:
                training = entry['stats'][model]['training']['median']
                external = entry['stats'][model]['external']['median']
                gaps[model] = {
                    'training': training,
                    'external': external,
                    'gap': round(training - external, 3) if training is not None and external is not None else None,
                }
            entry['train_vs_external'] = gaps
        # per prompt-category medians (categories come from the user's tags)
        by_category: Dict[str, dict] = {}
        for sample in bucket:
            key = checkpoint_key(sample)
            if key is None:
                continue
            cat = category_of(key[1], categories)
            by_category.setdefault(cat, []).append(sample)
        entry['by_category'] = {
            cat: {
                model: {source: distribution_stats([v for v in (score_of(s, model, source) for s in bucket_cat) if v is not None])
                        for source in sources}
                for model in models
            }
            for cat, bucket_cat in sorted(by_category.items())
        }
        # Phase 2e: per-checkpoint leakage rates (flagged / evaluable
        # multi-person samples), per model per source; only steps with
        # evaluable multi-person samples carry the block
        if group_key in step_leakage:
            entry['leakage'] = step_leakage[group_key]
        step_entries.append(entry)

    pairs = []
    for i, key_a in enumerate(ordered_keys):
        for key_b in ordered_keys[i + 1:]:
            comparisons = compare_pair(groups[key_a], groups[key_b], models, sources)
            pairs.append(
                {
                    'a': {'step': key_a[0], 'is_raw': key_a[1], 'label': group_label(key_a[0], key_a[1], has_raw)},
                    'b': {'step': key_b[0], 'is_raw': key_b[1], 'label': group_label(key_b[0], key_b[1], has_raw)},
                    'models': comparisons,
                }
            )

    notes = [
        'Checkpoints are compared only on (prompt, seed) pairs present at both steps; others are listed per pair with a reason.',
        'No overall best-checkpoint verdict: per-pair "higher"/"lower" only when the bootstrap CI of the median difference excludes zero.',
    ]
    if unparsed:
        notes.append(f'{unparsed} sample filenames did not parse to step/prompt/seed and are not grouped (custom naming scheme?).')
    if 'training' in sources and 'external' in sources:
        notes.append('Training-ref vs external-ref gap is shown separately; a rising gap can indicate memorization (see drift detection, Phase 2d).')
    if prompt_texts:
        notes.append('Prompt texts were captured from the job config at report time; mid-run sample-config changes cannot be recovered from filenames.')
    if picks_used:
        notes.append(f"{picks_used} ambiguous multi-person sample(s) resolved by user subject-face picks (Phase 2e); their picked face's scores join the stats and comparisons.")

    report = {
        'version': REPORT_VERSION,
        'job': results.get('job'),
        'generated_at': _now(),
        'results_completed': bool(results.get('completed')),
        'results_updated_at': results.get('updated_at'),
        'sources': sources,
        'models': models,
        'unparsed_samples': unparsed,
        'refsets': {
            name: ({'n': ref.get('n'), 'loo': ref.get('loo')} if ref else None)
            for name, ref in (results.get('refsets') or {}).items()
        },
        'prompt_texts': {str(k): v for k, v in (prompt_texts or {}).items()},
        'categories': categories,
        'steps': step_entries,
        'pairs': pairs,
        'convergence': convergence,
        'notes': notes,
    }
    # Phase 2d drift warnings. Imported here, not at module top: drift reads
    # helpers from this module, so a top-level import would be circular.
    # Drift sees the pick-resolved results document (a picked subject face
    # is the sample's identity from here on).
    from .drift import detect_drift

    report['drift'] = detect_drift(results, report)
    report['leakage'] = leakage_block
    return report


# ------------------------------------------------------- convergence (loss)


def convergence_summary(job_dir: str) -> Optional[dict]:
    """Loss-based "would more steps help?" gate, separate from identity.

    A minimal fork-side port of the core heuristic in the workspace-level
    ``Scripts/krea2_loss_progress.py`` (250-step bucket means, late-500
    change vs bucket noise, LR-floor check). Reads ``loss_log.db``
    read-only; returns None when the job has no loss log (e.g. never
    trained locally).
    """
    import sqlite3

    db_path = os.path.join(job_dir, 'loss_log.db')
    if not os.path.isfile(db_path):
        return None
    try:
        connection = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True, timeout=5.0)
        try:
            rows = connection.execute('SELECT step, value_real FROM metrics WHERE key="loss/img" ORDER BY step').fetchall()
            lr_start = connection.execute('SELECT value_real FROM metrics WHERE key="learning_rate" ORDER BY step LIMIT 1').fetchone()
            lr_end = connection.execute('SELECT AVG(value_real) FROM metrics WHERE key="learning_rate" AND step >= (SELECT MAX(step) FROM metrics WHERE key="learning_rate") - 50').fetchone()
        finally:
            connection.close()
    except Exception:  # noqa: BLE001 -- unreadable loss log is not fatal here
        return None
    if len(rows) < 2 or not lr_start or lr_end is None or lr_end[0] is None:
        return None
    data = np.asarray(rows, dtype=float)
    steps, values = data[:, 0], data[:, 1]
    hi = int(steps.max())

    def window(lo: float, hi_bound: float) -> Optional[float]:
        mask = (steps >= lo) & (steps < hi_bound)
        return float(values[mask].mean()) if mask.any() else None

    buckets = []
    for lo in range(0, hi + 1, 250):
        mean = window(lo, lo + 250)
        if mean is not None:
            buckets.append(mean)
    w2, w3 = window(hi - 1000, hi - 500), window(hi - 500, hi)
    late_change = (1.0 - w3 / w2) * 100.0 if w2 and w3 and w2 != 0 else None
    if len(buckets) >= 4:
        noise = float(np.std(buckets[-4:])) / float(np.mean(buckets[-4:])) * 100.0 if np.mean(buckets[-4:]) else 99.0
    else:
        noise = 99.0
    lr_start_value = float(lr_start[0])
    lr_end_value = float(lr_end[0])
    lr_floored = lr_end_value <= 2.5 * (lr_start_value * 0.2) or lr_end_value / lr_start_value <= 0.25 if lr_start_value else False
    converged = late_change is not None and abs(late_change) < max(5.0, 2 * noise) and lr_floored
    return {
        'last_step': hi,
        'loss_buckets_250': [round(b, 4) for b in buckets],
        'late_500_change_pct': round(late_change, 2) if late_change is not None else None,
        'lr_start': lr_start_value,
        'lr_end': lr_end_value,
        'lr_floored': bool(lr_floored),
        'bucket_noise_pct': round(noise, 2),
        'converged': bool(converged),
        'source': 'loss_log.db heuristic (core of Scripts/krea2_loss_progress.py); answers "would more steps help", never an identity measure',
    }


# ------------------------------------------------------------------ io


def categories_path(identity_dir: str) -> str:
    return os.path.join(identity_dir, 'prompt_categories.json')


def load_categories(identity_dir: str) -> Dict[str, str]:
    try:
        with open(categories_path(identity_dir), 'r', encoding='utf-8') as handle:
            document = json.load(handle)
        raw = document.get('categories') if isinstance(document, dict) else document
        if not isinstance(raw, dict):
            return {}
        return {str(k): str(v) for k, v in raw.items()}
    except (OSError, ValueError):
        return {}
