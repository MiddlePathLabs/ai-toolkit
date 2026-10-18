"""Multi-person identity leakage (Phase 2e, fork-only).

Pure post-processing over ``results.json`` plus the user's sample face
picks from ``selection.json`` -- no face models are loaded, so this runs
anywhere (CPU, during training, offline). Hard rules from the spec:

- **Per model.** Every level is computed and reported per model per
  reference source (ArcFace / SFace, training / external / combined);
  the two models are never averaged, merged or shared a scale. This is
  the same rule as the scoring stack, the retention graph and drift.
- **Three levels, weakest first.** ``resembles`` (informational: a
  non-target face scores within a small margin of the same-person band),
  ``likely_duplicate`` (at or above the same-person band -- the same
  operating point the reference photos themselves reach), and the
  user's own twin mark. A moderately high score alone is never called a
  duplicate; the tool's strongest statistical wording is "possible
  identity leakage".
- **The tool never asserts a twin.** A face is only ever *shown* as a
  user-marked twin; that level comes exclusively from the user's marks
  and outranks the statistical levels for display only.
- **No re-run needed.** Phase 1 already scores every face in every
  sample; a pick only re-selects which face is the subject, so picking
  (and un-picking) is instant post-processing.

Thresholds are LOO-anchored per model: the reference leave-one-out band
(how similar the reference photos are to each other) is the provisional
same-person operating point, and ``RESEMBLES_MARGIN`` below it defines
the informational band. Both are provisional pending Phase 2a
calibration and the 11.7 validation set -- the same convention the 2d
drift cut-points use.

Leakage rates respect the weights-kind rule: RAW_ and un-prefixed (EMA)
samples are grouped by ``(step, is_raw)`` and never pooled into one
rate.
"""

from typing import Dict, List, Optional, Sequence, Tuple

# v1: initial Phase 2e block -- resolved picks, three flag levels, per-step
# rates, closest-reference passthrough.
# v2: Phase 2f subject-absent marks -- ``subject_absent: true`` picks flip a
# face-bearing sample out of the stats entirely and feed the adherence block
# (does the subject appear anyway on a no-trigger prompt?).
LEAKAGE_VERSION = 2
# Provisional width of the informational "resembles" band, applied to each
# model's own LOO band (per model, so never a shared scale). Pending 2a/11.7.
RESEMBLES_MARGIN = 0.10
# Resembles-level faces are informational; a large run can produce many, so
# the report lists at most this many of them (stronger levels are all kept).
MAX_RESEMBLES_LISTED = 40

STATUS_AMBIGUOUS = 'ambiguous'
STATUS_EVALUATED = 'evaluated'
STATUS_ABSENT = 'subject_absent'
LEVEL_DUPLICATE = 'likely_duplicate'
LEVEL_RESEMBLES = 'resembles'


def classify_level(mean: Optional[float], loo: Optional[float]) -> Optional[str]:
    """Level of one non-target face score against a same-person band.

    ``>= loo`` -> ``likely_duplicate``; ``>= loo - RESEMBLES_MARGIN`` ->
    ``resembles`` (informational); below or unavailable -> ``None``.
    """
    if mean is None or loo is None:
        return None
    if mean >= loo:
        return LEVEL_DUPLICATE
    if mean >= loo - RESEMBLES_MARGIN:
        return LEVEL_RESEMBLES
    return None


def valid_pick(sample: dict, picks: Optional[dict]) -> Optional[dict]:
    """The sample's pick entry when it resolves a subject face, else None.

    Only multi-face ("ambiguous") samples need -- and honour -- a pick; the
    face index must point at a scored face. Everything else (no entry, null
    index, out-of-range index, non-ambiguous sample) resolves to None.
    """
    if sample.get('status') != STATUS_AMBIGUOUS:
        return None
    pick = (picks or {}).get(sample.get('filename') or '')
    if not isinstance(pick, dict):
        return None
    index = pick.get('face_index')
    faces = sample.get('faces') or []
    if not isinstance(index, int) or isinstance(index, bool) or not (0 <= index < len(faces)):
        return None
    return pick


def absent_pick(sample: dict, picks: Optional[dict]) -> Optional[dict]:
    """The sample's pick entry when you marked the subject absent, else None.

    Applies to any face-bearing sample (single- or multi-face); failed
    samples have no faces, so they can never be marked. Mirrors the UI's
    absentPickFor.
    """
    faces = sample.get('faces') or []
    if not faces or sample.get('status') not in (STATUS_EVALUATED, STATUS_AMBIGUOUS):
        return None
    pick = (picks or {}).get(sample.get('filename') or '')
    if isinstance(pick, dict) and pick.get('subject_absent') is True:
        return pick
    return None


def _resolved_copy(sample: dict, pick: dict) -> dict:
    picked = dict(sample)
    picked['status'] = STATUS_EVALUATED
    picked['primary'] = picked['faces'][pick['face_index']].get('scores')
    picked['picked_subject'] = True
    picked['subject_face_index'] = pick['face_index']
    return picked


def _absent_copy(sample: dict) -> dict:
    """A subject-absent sample leaves likeness stats entirely: no primary,
    its own status, and a marker. Its faces still carry scores, so the
    adherence block can ask whether the subject appeared anyway."""
    marked = dict(sample)
    marked['status'] = STATUS_ABSENT
    marked['primary'] = None
    marked['subject_absent'] = True
    marked.pop('picked_subject', None)
    marked.pop('subject_face_index', None)
    return marked


def resolve_picks(samples: Sequence[dict], picks: Optional[dict]) -> Tuple[List[dict], int]:
    """Apply subject-face picks and subject-absent marks to a sample list.

    A picked ambiguous sample becomes ``evaluated`` with ``primary`` set to
    the picked face's already-persisted per-model scores (Phase 1 scores
    every face), carrying ``picked_subject`` / ``subject_face_index``
    markers. An absent-marked face-bearing sample becomes
    ``subject_absent`` with ``primary`` cleared, so medians, strips and
    paired comparisons skip it. Unmarked samples pass through unchanged;
    the input list is never mutated. Returns ``(resolved_samples,
    picks_used)`` where picks_used counts subject-face picks only.
    """
    resolved: List[dict] = []
    used = 0
    for sample in samples:
        if absent_pick(sample, picks) is not None:
            resolved.append(_absent_copy(sample))
            continue
        pick = valid_pick(sample, picks)
        if pick is None:
            resolved.append(sample)
            continue
        resolved.append(_resolved_copy(sample, pick))
        used += 1
    return resolved, used


def _twin_marked(pick: Optional[dict], face_index: int) -> bool:
    marks = (pick or {}).get('twin_confirmed')
    return isinstance(marks, list) and face_index in marks


def _loo_for(refsets: dict, source: str, model: str) -> Optional[float]:
    refset = refsets.get(source) if isinstance(refsets, dict) else None
    if not isinstance(refset, dict):
        return None
    value = (refset.get('loo') or {}).get(model)
    return float(value) if isinstance(value, (int, float)) else None


def detect_leakage(
    results: dict,
    picks: Optional[dict],
    models: Sequence[str],
    sources: Sequence[str],
) -> dict:
    """Scan a results document plus sample picks for identity leakage.

    Returns ``{'version', 'margin', 'thresholds', 'picks_used', 'counts',
    'faces', 'per_step', 'notes'}``. ``faces`` lists every flagged non-target
    face (any statistical level, or user-marked) with both models' scores for
    the inspection view; ``per_step`` carries the per-checkpoint rate
    ``flagged / evaluable`` per model per source, grouped by ``(step,
    is_raw)`` so RAW and EMA never share a rate.
    """
    raw_samples = results.get('samples') or []
    resolved, picks_used = resolve_picks(raw_samples, picks)
    # (resolved view, pick, absent) triples -- the pick/absent lookups run on
    # the RAW record (resolution flips status away from ambiguous/evaluated)
    # while the resolved copy supplies the scored-subject view downstream
    samples: List[Tuple[dict, Optional[dict], bool]] = [
        (res, valid_pick(raw, picks), absent_pick(raw, picks) is not None)
        for raw, res in zip(raw_samples, resolved)
    ]
    refsets = results.get('refsets') or {}

    thresholds: Dict[str, Dict[str, dict]] = {}
    for model in models:
        for source in sources:
            loo = _loo_for(refsets, source, model)
            if loo is None:
                continue
            thresholds.setdefault(model, {})[source] = {
                'loo': round(loo, 3),
                'resembles_min': round(loo - RESEMBLES_MARGIN, 3),
            }

    faces: List[dict] = []
    resembles_only: List[dict] = []
    # (step, is_raw) -> model -> source -> {flagged, evaluable}
    per_step: Dict[Tuple[int, bool], Dict[str, Dict[str, dict]]] = {}
    unpicked_multi = 0
    no_loo_pairs = set()

    for sample, pick, absent in samples:
        if len(sample.get('faces') or []) < 2:
            continue
        filename = sample.get('filename') or ''
        if absent:
            # resolved-absent, not awaiting a pick; its faces are judged by
            # the adherence block instead of the leakage rate
            continue
        if pick is None:
            unpicked_multi += 1
            continue
        subject_index = pick['face_index']
        step, is_raw = sample.get('step'), bool(sample.get('is_raw'))
        step_key: Optional[Tuple[int, bool]] = (
            (int(step), is_raw) if isinstance(step, int) else None
        )
        # a picked multi-face sample is evaluable for the rate regardless of
        # model (per-model gaps would hide samples a model failed to score;
        # flagged counts stay per model)
        if step_key is not None:
            for model in models:
                rates = per_step.setdefault(step_key, {}).setdefault(model, {})
                for source in sources:
                    entry = rates.setdefault(source, {'flagged': 0, 'evaluable': 0})
                    entry['evaluable'] += 1

        sample_flagged = set()
        for face in sample.get('faces') or []:
            index = face.get('index')
            if index == subject_index:
                continue
            scores = face.get('scores') or {}
            levels: Dict[str, Dict[str, str]] = {}
            for model in models:
                for source in sources:
                    mean = ((scores.get(model) or {}).get(source) or {}).get('mean')
                    loo = _loo_for(refsets, source, model)
                    level = classify_level(float(mean) if mean is not None else None, loo)
                    if loo is None:
                        no_loo_pairs.add(f'{model}:{source}')
                    if level is not None:
                        levels.setdefault(model, {})[source] = level
                    if level == LEVEL_DUPLICATE:
                        sample_flagged.add(f'{model}:{source}')
            user_twin = _twin_marked(pick, index if isinstance(index, int) else -1)
            if user_twin:
                # the user's mark outranks every statistical level and counts
                # toward the rate for both models (it is not a per-model claim)
                sample_flagged.update(f'{model}:{source}' for model in models for source in sources)
            if not levels and not user_twin:
                continue
            record = {
                'filename': filename,
                'sample_path': sample.get('path'),
                'step': step,
                'is_raw': is_raw,
                'subject_face': subject_index,
                'face_index': index,
                'det_score': face.get('det_score'),
                'user_twin': user_twin,
                'levels': levels,
                'scores': scores,
                'closest_ref': face.get('closest_ref'),
            }
            if user_twin or any(l == LEVEL_DUPLICATE for per_model in levels.values() for l in per_model.values()):
                faces.append(record)
            else:
                resembles_only.append(record)
        if step_key is not None:
            for key in sample_flagged:
                model, source = key.split(':', 1)
                per_step[step_key][model][source]['flagged'] += 1

    faces.sort(key=lambda f: (not f['user_twin'], f['filename'], f['face_index'] if isinstance(f['face_index'], int) else 0))
    resembles_only.sort(key=lambda f: (f['filename'], f['face_index'] if isinstance(f['face_index'], int) else 0))
    truncated = max(0, len(resembles_only) - MAX_RESEMBLES_LISTED)
    faces += resembles_only[:MAX_RESEMBLES_LISTED]

    # Trigger adherence: for every subject-absent sample (any face count),
    # does any face reach the LOO band anyway? leaked = the subject likely
    # appeared despite the prompt not asking for them; borderline = in the
    # resembles band; clean = all scored faces clearly strangers. Per model
    # per source, never averaged; a model that scored nothing is 'unscored',
    # never clean.
    absent_count = sum(1 for _, _, absent in samples if absent)
    adherence_models: Dict[str, Dict[str, dict]] = {}
    for sample, _, absent in samples:
        if not absent:
            continue
        for model in models:
            for source in sources:
                loo = _loo_for(refsets, source, model)
                entry = adherence_models.setdefault(model, {}).setdefault(
                    source, {'scored': 0, 'leaked': 0, 'borderline': 0, 'clean': 0, 'unscored': 0}
                )
                verdict: Optional[str] = None
                saw_score = False
                for face in sample.get('faces') or []:
                    mean = (((face.get('scores') or {}).get(model) or {}).get(source) or {}).get('mean')
                    if mean is None:
                        continue
                    saw_score = True
                    level = classify_level(float(mean), loo)
                    if level == LEVEL_DUPLICATE:
                        verdict = 'leaked'
                        break
                    if level == LEVEL_RESEMBLES and verdict is None:
                        verdict = 'borderline'
                if saw_score:
                    entry['scored'] += 1
                    entry[verdict or 'clean'] += 1
                else:
                    entry['unscored'] += 1
    adherence = {'marked': absent_count, 'models': adherence_models} if absent_count else None

    per_step_list = []
    for (step, is_raw), by_model in sorted(per_step.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        rates: Dict[str, Dict[str, dict]] = {}
        for model, by_source in by_model.items():
            for source, entry in by_source.items():
                if entry['evaluable'] <= 0:
                    continue
                flagged = min(entry['flagged'], entry['evaluable'])
                rates.setdefault(model, {})[source] = {
                    'flagged': flagged,
                    'evaluable': entry['evaluable'],
                    'rate': round(flagged / entry['evaluable'], 3),
                }
        if rates:
            per_step_list.append({'step': step, 'is_raw': is_raw, 'rates': rates})

    notes = [
        'Leakage levels are computed and shown per model and per reference source; ArcFace and SFace are never averaged or merged.',
        "'resembles subject' is informational and excluded from the per-checkpoint rate; only likely-duplicate faces and your own twin marks count.",
        'Levels are LOO-anchored and provisional pending calibration (Phase 2a) and the validation set (11.7).',
    ]
    if picks_used:
        notes.append(f'{picks_used} multi-person sample(s) scored via a user-picked subject face; per-face scores were already stored, no re-run needed.')
    if absent_count:
        notes.append(
            f'{absent_count} sample(s) marked subject-absent (no-trigger prompts): excluded from likeness stats; '
            "checked for trigger adherence instead -- 'leaked' means a face still reached the LOO band."
        )
    if unpicked_multi:
        notes.append(f'{unpicked_multi} multi-person sample(s) have no subject-face pick yet; pick the subject in the panel to include them in the leakage rate.')
    if no_loo_pairs:
        notes.append(
            'No leave-one-out band (fewer than two references) for ' + ', '.join(sorted(no_loo_pairs)) +
            ' -- statistical levels unavailable for those model/source pairs.'
        )
    if truncated:
        notes.append(f'{truncated} resembles-level face(s) not listed (informational band; stronger levels are always kept).')
    notes.append("A face is only shown as a twin when you marked it; the tool's strongest statistical wording is 'possible identity leakage'.")

    return {
        'version': LEAKAGE_VERSION,
        'margin': RESEMBLES_MARGIN,
        'thresholds': thresholds,
        'picks_used': picks_used,
        'counts': {
            'multi_face_samples': sum(1 for s, _, _ in samples if len(s.get('faces') or []) >= 2),
            'evaluable': sum(1 for _, p, _ in samples if p is not None),
            'unpicked_multi': unpicked_multi,
            'subject_absent': absent_count,
        },
        'adherence': adherence,
        'faces': faces,
        'per_step': per_step_list,
        'notes': notes,
    }
