"""Drift detection (Phase 2d) tests.

Pure post-processing like Phase 2b -- no face models, so every test runs.
Covers the spec's rules: warnings only ("possible drift", never anything
stronger), per-model computation, CI-gated statistics, and graceful skips
(old results without landmarks, sparse categories, small reference sets).
"""

import json

import pytest

from toolkit.identity_eval import checkpoint_compare as cc
from toolkit.identity_eval import drift

FRONTAL_KPS = [[30.0, 40.0], [70.0, 40.0], [50.0, 55.0], [40.0, 70.0], [60.0, 70.0]]
THREE_QUARTER_KPS = [[30.0, 40.0], [70.0, 40.0], [56.0, 55.0], [40.0, 70.0], [60.0, 70.0]]
PROFILE_KPS = [[30.0, 40.0], [70.0, 40.0], [62.0, 55.0], [40.0, 70.0], [60.0, 70.0]]


def make_sample(
    step,
    prompt_idx,
    seed,
    arcface=None,
    sface=None,
    external_delta=-0.02,
    max_delta=0.01,
    kps=None,
    status='evaluated',
    is_raw=False,
):
    primary = None
    if status == 'evaluated':

        def block(value):
            return None if value is None else {'mean': round(value, 4), 'max': round(value + max_delta, 4)}

        external = None if arcface is None else arcface + external_delta
        primary = {
            'arcface': {'training': block(arcface), 'external': block(external)},
            'sface': {'training': block(sface), 'external': block(None if sface is None else sface + external_delta)},
        }
    faces = []
    if status == 'evaluated':
        faces = [{'index': 0, 'bbox': [0.0, 0.0, 10.0, 10.0], 'kps': kps, 'det_score': 0.99, 'scores': primary or {}}]
    return {
        'path': f'out/{step}_{prompt_idx}_{seed}.jpg',
        'filename': f'1791073894128__{step:09d}_{prompt_idx:02d}_seed-{seed}_steps-20.jpg',
        'file_sha': None,
        'step': step,
        'prompt_idx': prompt_idx,
        'seed': seed,
        'is_raw': is_raw,
        'status': status,
        'face_count': 1 if status == 'evaluated' else 0,
        'error': None,
        'primary': primary,
        'faces': faces,
        'warnings': [],
    }


def make_results(samples, training_refs=5):
    return {
        'version': 2,
        'job': 'job_drift',
        'completed': True,
        'updated_at': '2026-10-13T00:00:00',
        'sources': ['training', 'external'],
        'refsets': {
            'training': {'n': training_refs, 'hash': 'x', 'skipped': {}, 'loo': {'arcface': 0.58, 'sface': 0.6}},
            'external': {'n': 4, 'hash': 'y', 'skipped': {}, 'loo': {'arcface': 0.57, 'sface': 0.59}},
        },
        'samples': samples,
        'warnings': [],
    }


def build(samples, categories=None, training_refs=5):
    return cc.build_checkpoint_report(make_results(samples, training_refs), categories, None, None)


def kinds(report):
    return [s['kind'] for s in report['drift']['signals']]


def of_kind(report, kind):
    return [s for s in report['drift']['signals'] if s['kind'] == kind]


class TestPose:
    def test_frontal_symmetric_landmarks(self):
        ratio = drift.pose_asymmetry(FRONTAL_KPS)
        assert ratio == pytest.approx(0.0, abs=1e-9)
        assert drift.pose_bucket(ratio) == 'frontal'

    def test_profile_nose_near_one_eye(self):
        ratio = drift.pose_asymmetry(PROFILE_KPS)
        assert ratio == pytest.approx((32.0 - 8.0) / 40.0, abs=1e-9)
        assert drift.pose_bucket(ratio) == 'profile'

    def test_three_quarter_between_cut_points(self):
        ratio = drift.pose_asymmetry(THREE_QUARTER_KPS)
        assert drift.POSE_FRONTAL_MAX <= ratio < drift.POSE_PROFILE_MIN
        assert drift.pose_bucket(ratio) == 'three_quarter'

    def test_malformed_landmarks_return_none(self):
        assert drift.pose_asymmetry(None) is None
        assert drift.pose_asymmetry([]) is None
        assert drift.pose_asymmetry([[1, 2]]) is None
        assert drift.pose_asymmetry([['a', 'b'], ['c', 'd'], ['e', 'f']]) is None
        assert drift.pose_asymmetry([[10, 10], [10, 10], [10, 10]]) is None  # zero span


class TestSimilarityDecline:
    def test_decline_fires_against_highest_median_anchor(self):
        samples = [make_sample(1000, p, 42, 0.60, sface=0.61) for p in range(6)]
        samples += [make_sample(2000, p, 42, 0.52, sface=0.53) for p in range(6)]
        report = build(samples)
        fired = of_kind(report, 'similarity_decline')
        assert {s['model'] for s in fired} == {'arcface', 'sface'}
        for signal in fired:
            assert signal['evidence']['anchor_step'] == 1000
            assert signal['scope'] == {'step': 2000, 'is_raw': False}
            assert signal['evidence']['ci_high'] < 0
            assert len(signal['samples']) == 6  # all decliners, under the link cap

    def test_stable_scores_do_not_fire(self):
        samples = [make_sample(1000, p, 42, 0.60 + 0.001 * p) for p in range(6)]
        samples += [make_sample(2000, p, 42, 0.60 + 0.001 * p) for p in range(6)]
        report = build(samples)
        assert of_kind(report, 'similarity_decline') == []

    def test_anchor_is_the_peak_not_the_first_step(self):
        # 1000 low, 1500 peak, 2000 low: the 2000 warning must anchor on 1500,
        # and 1500 itself is a rise, not a decline
        samples = [make_sample(1000, p, 42, 0.50) for p in range(6)]
        samples += [make_sample(1500, p, 42, 0.60) for p in range(6)]
        samples += [make_sample(2000, p, 42, 0.52) for p in range(6)]
        report = build(samples)
        fired = of_kind(report, 'similarity_decline')
        assert {s['scope']['step'] for s in fired} == {2000}
        assert all(s['evidence']['anchor_step'] == 1500 for s in fired)


class TestVarianceIncrease:
    def test_spread_growth_fires(self):
        # ten pairs with equal medians: small paired-IQR bootstrap cells
        # degenerate at n=6 and the CI correctly touches zero
        tight = [0.597, 0.598, 0.599, 0.600, 0.600, 0.601, 0.602, 0.603, 0.598, 0.602]
        wide = [0.48, 0.50, 0.52, 0.54, 0.56, 0.64, 0.66, 0.68, 0.70, 0.72]
        samples = [make_sample(1000, p, 42, tight[p]) for p in range(10)]
        samples += [make_sample(2000, p, 42, wide[p]) for p in range(10)]
        report = build(samples)
        fired = of_kind(report, 'variance_increase')
        assert fired, 'expected variance warnings for a clearly wider later spread'
        for signal in fired:
            assert signal['evidence']['ci_low'] > 0
            assert signal['evidence']['iqr_later'] > signal['evidence']['iqr_anchor']
        # same medians: the decline signal must stay silent here
        assert of_kind(report, 'similarity_decline') == []

    def test_equal_spread_does_not_fire(self):
        values = [0.57, 0.585, 0.60, 0.60, 0.615, 0.63]
        samples = [make_sample(1000, p, 42, values[p]) for p in range(6)]
        samples += [make_sample(2000, p, 42, values[p]) for p in range(6)]
        report = build(samples)
        assert of_kind(report, 'variance_increase') == []


class TestTrainExternalDivergence:
    def test_training_rises_external_flat_fires(self):
        samples = [make_sample(1000, p, 42, 0.55, sface=0.56, external_delta=-0.02) for p in range(6)]
        samples += [make_sample(2000, p, 42, 0.58, sface=0.59, external_delta=-0.05) for p in range(6)]
        report = build(samples)
        fired = of_kind(report, 'train_external_divergence')
        assert {s['model'] for s in fired} == {'arcface', 'sface'}
        for signal in fired:
            assert signal['source'] is None  # cross-source by construction
            assert signal['evidence']['training_ci'][0] > 0
            assert signal['evidence']['external_diff'] == pytest.approx(0.0, abs=1e-6)
            assert len(signal['samples']) == 6

    def test_inconclusive_external_rise_is_not_memorization(self):
        # training +0.05 (CI excludes 0) but external rose +0.15 median with
        # a CI straddling 0 (verdict 'none') and the gap SHRANK -- an
        # inconclusive external trend must not be read as "stayed flat"
        external_deltas = [0.33, 0.33, -0.17, -0.17, 0.33, -0.17]  # external diffs +0.4 x3 / -0.1 x3
        samples = [make_sample(1000, p, 42, 0.55, external_delta=-0.02) for p in range(6)]
        samples += [make_sample(2000, p, 42, 0.60, external_delta=external_deltas[p]) for p in range(6)]
        report = build(samples)
        assert of_kind(report, 'train_external_divergence') == []

    def test_gap_growth_ci_in_evidence_when_firing(self):
        samples = [make_sample(1000, p, 42, 0.55, external_delta=-0.02) for p in range(6)]
        samples += [make_sample(2000, p, 42, 0.58, external_delta=-0.05) for p in range(6)]
        report = build(samples)
        fired = of_kind(report, 'train_external_divergence')
        assert fired
        for signal in fired:
            growth_low, growth_high = signal['evidence']['gap_growth_ci']
            assert growth_low > 0
            # per-sample gap 0.02 -> 0.05 on every pair
            assert signal['evidence']['gap_growth_median'] == pytest.approx(0.03, abs=1e-6)
            assert 'gap grew' in signal['message']

    def test_both_rising_does_not_fire(self):
        samples = [make_sample(1000, p, 42, 0.55, external_delta=-0.02) for p in range(6)]
        samples += [make_sample(2000, p, 42, 0.58, external_delta=-0.02) for p in range(6)]
        report = build(samples)
        assert of_kind(report, 'train_external_divergence') == []

    def test_needs_both_sources(self):
        samples = [make_sample(1000, p, 42, 0.55, external_delta=-0.02) for p in range(6)]
        samples += [make_sample(2000, p, 42, 0.58, external_delta=-0.05) for p in range(6)]
        for sample in samples:  # remove every external score -> source disappears entirely
            sample['primary']['arcface'].pop('external')
            sample['primary']['sface'].pop('external')
        report = build(samples)
        assert 'external' not in report['sources']
        assert of_kind(report, 'train_external_divergence') == []


class TestPoseWeakness:
    def test_non_frontal_scoring_lower_fires(self):
        samples = [make_sample(1000, p, 42, 0.62 if p < 4 else 0.54, kps=FRONTAL_KPS if p < 4 else PROFILE_KPS) for p in range(8)]
        report = build(samples)
        fired = of_kind(report, 'pose_weakness')
        assert fired
        for signal in fired:
            assert signal['evidence']['pose_counts'] == {'frontal': 4, 'three_quarter': 0, 'profile': 4}
            assert signal['evidence']['ci_high'] < 0
            assert len(signal['samples']) == 4  # the weakest non-frontal samples
            for filename in signal['samples']:
                assert '_04_' in filename or '_05_' in filename or '_06_' in filename or '_07_' in filename

    def test_pose_independent_scores_do_not_fire(self):
        samples = [make_sample(1000, p, 42, 0.60, kps=FRONTAL_KPS if p < 4 else PROFILE_KPS) for p in range(8)]
        report = build(samples)
        assert of_kind(report, 'pose_weakness') == []

    def test_missing_kps_note_and_no_signal(self):
        samples = [make_sample(1000, p, 42, 0.62 if p < 4 else 0.54) for p in range(8)]  # no kps anywhere
        report = build(samples)
        assert of_kind(report, 'pose_weakness') == []
        assert any('predates per-face landmark persistence' in note for note in report['drift']['notes'])


class TestNarrowGeneralization:
    def test_category_spread_fires_pooled(self):
        samples = []
        for step in (1000, 2000):
            for p in range(4):
                samples.append(make_sample(step, p, 42, 0.64))  # prompts 0-3 -> 'closeup'
            for p in range(4, 8):
                samples.append(make_sample(step, p, 42, 0.50))  # prompts 4-7 -> 'wide'
        categories = {str(p): 'closeup' for p in range(4)}
        categories.update({str(p): 'wide' for p in range(4, 8)})
        report = build(samples, categories)
        fired = of_kind(report, 'narrow_generalization')
        assert fired
        for signal in fired:
            assert signal['evidence']['high_category'] == 'closeup'
            assert signal['evidence']['low_category'] == 'wide'
            assert signal['evidence']['ci_high'] < 0
            assert signal['evidence']['n_high'] == 8 and signal['evidence']['n_low'] == 8

    def test_untagged_only_idles_with_note(self):
        samples = [make_sample(1000, p, 42, 0.64 if p < 4 else 0.50) for p in range(8)]
        report = build(samples)
        assert of_kind(report, 'narrow_generalization') == []
        assert any('no prompt categories tagged' in note for note in report['drift']['notes'])

    def test_single_checkpoint_fires(self):
        # category spread is not temporal: one checkpoint with 6+ samples per
        # tagged category is enough -- the old two-step guard skipped it
        samples = [make_sample(1000, p, 42, 0.64 if p < 6 else 0.50) for p in range(12)]
        categories = {str(p): ('closeup' if p < 6 else 'wide') for p in range(12)}
        report = build(samples, categories)
        fired = of_kind(report, 'narrow_generalization')
        assert fired
        for signal in fired:
            assert signal['evidence']['high_category'] == 'closeup'
            assert signal['evidence']['low_category'] == 'wide'
            assert signal['evidence']['n_high'] == 6 and signal['evidence']['n_low'] == 6

    def test_single_qualified_category_does_not_fire(self):
        samples = []
        for step in (1000, 2000):
            for p in range(6):
                samples.append(make_sample(step, p, 42, 0.64))  # six in 'closeup'
            samples.append(make_sample(step, 6, 42, 0.50))  # one in 'wide' (below min n)
        categories = {str(p): ('closeup' if p < 6 else 'wide') for p in range(7)}
        report = build(samples, categories)
        assert of_kind(report, 'narrow_generalization') == []


class TestNearCopy:
    def test_outlier_margin_flagged(self):
        samples = [make_sample(1000, p, 42, 0.60, sface=0.61) for p in range(6)]
        samples += [make_sample(2000, p, 42, 0.60, sface=0.61, max_delta=0.20 if p == 3 else 0.01) for p in range(6)]
        report = build(samples)
        fired = of_kind(report, 'near_copy')
        assert {s['model'] for s in fired} == {'arcface', 'sface'}
        for signal in fired:
            assert signal['evidence']['n_flagged'] == 1
            assert signal['evidence']['cutoff'] == pytest.approx(0.06)  # median margin + floor
            assert len(signal['samples']) == 1
            assert '_03_' in signal['samples'][0]

    def test_uniform_margins_do_not_fire(self):
        samples = [make_sample(1000, p, 42, 0.60) for p in range(6)]
        samples += [make_sample(2000, p, 42, 0.60) for p in range(6)]
        report = build(samples)
        assert of_kind(report, 'near_copy') == []

    def test_raw_pool_cannot_shift_the_ema_cutoff(self):
        # one EMA near-copy outlier; 20 RAW samples with uniform 0.30 margins
        # must not lift the pooled cutoff above it -- cutoffs are per weights kind
        samples = [make_sample(1000, p, 42, 0.60, sface=0.61) for p in range(6)]
        samples += [make_sample(2000, p, 42, 0.60, sface=0.61, max_delta=0.20 if p == 3 else 0.01) for p in range(6)]
        for step in (1000, 2000):
            for p in range(6, 16):  # RAW_ population at the same steps, uniform high margins
                samples.append(make_sample(step, p, 42, 0.60, sface=0.61, max_delta=0.30, is_raw=True))
        report = build(samples)
        fired = of_kind(report, 'near_copy')
        ema = [s for s in fired if s['evidence']['weights'] == 'EMA']
        assert ema, 'RAW margins must not suppress the EMA near-copy warning'
        for signal in ema:
            assert signal['evidence']['n_flagged'] == 1
            assert '_03_' in signal['samples'][0]
        # the uniform RAW margins are their own population: cutoff 0.30+floor, no outliers
        assert [s for s in fired if s['evidence']['weights'] == 'raw'] == []

    def test_per_weights_kind_outliers_flagged_independently(self):
        samples = [make_sample(1000, p, 42, 0.60, sface=0.61) for p in range(6)]
        samples += [make_sample(2000, p, 42, 0.60, sface=0.61, max_delta=0.20 if p == 3 else 0.01) for p in range(6)]
        # RAW population with its own outlier (uniform 0.30 margins + one at 0.60)
        for step in (1000, 2000):
            for p in range(6, 16):
                samples.append(
                    make_sample(step, p, 42, 0.60, sface=0.61, max_delta=0.60 if p == 8 else 0.30, is_raw=True)
                )
        report = build(samples)
        fired = of_kind(report, 'near_copy')
        by_weights = {s['evidence']['weights']: s for s in fired}
        assert set(by_weights) == {'EMA', 'raw'}
        assert by_weights['EMA']['evidence']['cutoff'] == pytest.approx(0.06)
        assert by_weights['raw']['evidence']['cutoff'] == pytest.approx(0.35)  # 0.30 median + floor

    def test_small_refset_skips_signal(self):
        samples = [make_sample(1000, p, 42, 0.60) for p in range(6)]
        samples += [make_sample(2000, p, 42, 0.60, max_delta=0.20 if p == 3 else 0.01) for p in range(6)]
        report = build(samples, training_refs=3)
        assert of_kind(report, 'near_copy') == []


class TestIntegrity:
    def test_detection_is_deterministic(self):
        samples = [make_sample(1000, p, 42, 0.60, kps=FRONTAL_KPS if p < 4 else PROFILE_KPS) for p in range(8)]
        samples += [make_sample(2000, p, 42, 0.52, kps=FRONTAL_KPS if p < 4 else PROFILE_KPS) for p in range(8)]
        first = build(samples, {'0': 'solo', '1': 'solo', '2': 'multi', '3': 'multi', '4': 'solo', '5': 'multi', '6': 'solo', '7': 'multi'})
        second = build(samples, {'0': 'solo', '1': 'solo', '2': 'multi', '3': 'multi', '4': 'solo', '5': 'multi', '6': 'solo', '7': 'multi'})
        assert first['drift'] == second['drift']

    def test_wording_and_schema_rules(self):
        samples = [make_sample(1000, p, 42, 0.55, external_delta=-0.02) for p in range(6)]
        samples += [make_sample(2000, p, 42, 0.58, external_delta=-0.05) for p in range(6)]
        report = build(samples)
        blob = json.dumps(report['drift']).lower()
        assert 'confirmed' not in blob
        assert '"best"' not in json.dumps(report).lower()
        for signal in report['drift']['signals']:
            assert signal['kind'] in drift.DRIFT_SIGNAL_KINDS
            assert signal['model'] in ('arcface', 'sface')  # exactly one model, never a blend
            assert signal['message'].startswith('possible drift:')
            assert isinstance(signal['samples'], list)
            assert set(signal['evidence'].keys()) <= {
                'anchor_step', 'anchor_label', 'diff_median', 'ci_low', 'ci_high', 'wins_later', 'wins_anchor', 'n_pairs',
                'iqr_anchor', 'iqr_later', 'first_step', 'last_step', 'training_diff', 'training_ci', 'external_diff',
                'external_ci', 'gap_first', 'gap_last', 'gap_growth_median', 'gap_growth_ci', 'pose_counts',
                'frontal_median', 'non_frontal_median', 'n_frontal', 'n_non_frontal', 'high_category', 'high_median',
                'low_category', 'low_median', 'n_high', 'n_low', 'weights', 'cutoff', 'median_margin', 'mad',
                'n_flagged', 'n_samples', 'training_refs',
            }

    def test_arcface_only_input_yields_arcface_only_warnings(self):
        samples = [make_sample(1000, p, 42, 0.60, sface=None) for p in range(6)]
        samples += [make_sample(2000, p, 42, 0.52, sface=None) for p in range(6)]
        report = build(samples)
        assert report['models'] == ['arcface']
        assert report['drift']['signals']
        assert all(s['model'] == 'arcface' for s in report['drift']['signals'])

    def test_report_carries_drift_block(self):
        samples = [make_sample(1000, p, 42, 0.60) for p in range(3)]
        report = build(samples)
        assert report['drift']['version'] == drift.DRIFT_VERSION
        assert report['drift']['signals'] == []
        assert report['drift']['notes']
        assert cc.REPORT_VERSION == 4
