"""Multi-person identity leakage (Phase 2e) tests.

Pure post-processing like Phase 2b/2d -- no face models, so every test
runs. Covers the spec's rules: three levels (resembles informational /
likely duplicate LOO-anchored / user-marked twin -- never asserted by the
tool), per-model computation (never averaged, never merged), pick
resolution of ambiguous samples, per-checkpoint rates that never pool RAW
with EMA, and graceful skips (no pick, no LOO band).
"""

import json

import pytest

from toolkit.identity_eval import checkpoint_compare as cc
from toolkit.identity_eval import leakage

KPS = [[30.0, 40.0], [70.0, 40.0], [50.0, 55.0], [40.0, 70.0], [60.0, 70.0]]

LOO = {'arcface': 0.58, 'sface': 0.60}


def _scores(arcface, sface):
    def block(value):
        return None if value is None else {'mean': round(value, 4), 'max': round(value + 0.02, 4)}

    return {
        'arcface': {'training': block(arcface), 'external': block(None if arcface is None else arcface - 0.02)},
        'sface': {'training': block(sface), 'external': block(None if sface is None else sface - 0.02)},
    }


def make_face(index, arcface, sface, closest=None):
    return {
        'index': index,
        'bbox': [10.0 * index, 0.0, 10.0 * index + 8.0, 8.0],
        'kps': KPS,
        'det_score': 0.9,
        'scores': _scores(arcface, sface),
        'closest_ref': closest or {},
    }


def make_multi_sample(step, prompt_idx, seed, faces, is_raw=False):
    return {
        'path': f'out/{step}_{prompt_idx}_{seed}.jpg',
        'filename': f'1791073894128__{step:09d}_{prompt_idx:02d}_seed-{seed}_steps-20.jpg',
        'file_sha': None,
        'step': step,
        'prompt_idx': prompt_idx,
        'seed': seed,
        'is_raw': is_raw,
        'status': 'ambiguous',
        'face_count': len(faces),
        'error': None,
        'primary': None,
        'faces': faces,
        'warnings': ['multiple faces; scores shown per face (Phase 2 adds a subject face picker)'],
    }


def make_solo_sample(step, prompt_idx, seed, arcface=0.6, sface=0.62):
    return {
        'path': f'out/{step}s_{prompt_idx}_{seed}.jpg',
        'filename': f'1791073894128__{step:09d}_{prompt_idx:02d}_seed-{seed}_steps-20.jpg',
        'file_sha': None,
        'step': step,
        'prompt_idx': prompt_idx,
        'seed': seed,
        'is_raw': False,
        'status': 'evaluated',
        'face_count': 1,
        'error': None,
        'primary': _scores(arcface, sface),
        'faces': [make_face(0, arcface, sface)],
        'warnings': [],
    }


def make_results(samples):
    return {
        'version': 3,
        'job': 'job_leak',
        'completed': True,
        'updated_at': '2026-10-13T00:00:00',
        'sources': ['training', 'external'],
        'refsets': {
            'training': {'n': 5, 'hash': 'x', 'skipped': {}, 'loo': dict(LOO)},
            'external': {'n': 4, 'hash': 'y', 'skipped': {}, 'loo': {'arcface': 0.57, 'sface': 0.59}},
        },
        'samples': samples,
        'warnings': [],
    }


def detect(samples, picks=None):
    return leakage.detect_leakage(make_results(samples), picks or {}, ['arcface', 'sface'], ['training', 'external'])


def build(samples, picks=None):
    return cc.build_checkpoint_report(make_results(samples), None, None, None, picks=picks or {})


class TestClassifyLevel:
    def test_at_loo_is_likely_duplicate(self):
        assert leakage.classify_level(0.58, 0.58) == 'likely_duplicate'
        assert leakage.classify_level(0.65, 0.58) == 'likely_duplicate'

    def test_within_margin_is_resembles(self):
        assert leakage.classify_level(0.58 - 0.05, 0.58) == 'resembles'
        assert leakage.classify_level(0.58 - leakage.RESEMBLES_MARGIN, 0.58) == 'resembles'

    def test_below_margin_is_none(self):
        assert leakage.classify_level(0.58 - leakage.RESEMBLES_MARGIN - 0.001, 0.58) is None

    def test_missing_inputs_are_none(self):
        assert leakage.classify_level(None, 0.58) is None
        assert leakage.classify_level(0.6, None) is None

    def test_margin_is_applied_to_each_models_own_band(self):
        # same score, different bands -> different levels (never a shared scale)
        assert leakage.classify_level(0.585, 0.60) == 'resembles'
        assert leakage.classify_level(0.585, 0.58) == 'likely_duplicate'


class TestResolvePicks:
    def test_pick_resolves_ambiguous_to_evaluated(self):
        faces = [make_face(0, 0.6, 0.62), make_face(1, 0.4, 0.41)]
        sample = make_multi_sample(1000, 0, 42, faces)
        resolved, used = leakage.resolve_picks([sample], {'1791073894128__000001000_00_seed-42_steps-20.jpg': {'face_index': 1}})
        assert used == 1
        assert resolved[0]['status'] == 'evaluated'
        assert resolved[0]['primary'] is faces[1]['scores']
        assert resolved[0]['picked_subject'] is True
        assert resolved[0]['subject_face_index'] == 1
        # the original record is untouched (results.json is immutable)
        assert sample['status'] == 'ambiguous' and sample['primary'] is None

    def test_invalid_and_absent_picks_pass_through(self):
        sample = make_multi_sample(1000, 0, 42, [make_face(0, 0.6, 0.62), make_face(1, 0.4, 0.41)])
        name = sample['filename']
        for picks in ({}, {name: {}}, {name: {'face_index': None}}, {name: {'face_index': 9}}, {name: {'face_index': '0'}}, {'other.jpg': {'face_index': 0}}):
            resolved, used = leakage.resolve_picks([sample], picks)
            assert used == 0 and resolved[0]['status'] == 'ambiguous'

    def test_solo_samples_ignore_picks(self):
        sample = make_solo_sample(1000, 0, 42)
        resolved, used = leakage.resolve_picks([sample], {sample['filename']: {'face_index': 0}})
        assert used == 0 and resolved[0] is sample


class TestSubjectAbsent:
    """Phase 2f: subject_absent marks ("no one in the picture is the subject")."""

    def test_absent_mark_flips_sample_out_of_stats(self):
        multi = make_multi_sample(1000, 0, 42, [make_face(0, 0.6, 0.62), make_face(1, 0.4, 0.41)])
        solo = make_solo_sample(1000, 1, 42, arcface=0.6, sface=0.62)
        picks = {multi['filename']: {'subject_absent': True}, solo['filename']: {'subject_absent': True}}
        resolved, used = leakage.resolve_picks([multi, solo], picks)
        # an absent mark is not a subject pick: picks_used stays 0
        assert used == 0
        for original, marked in zip((multi, solo), resolved):
            assert marked['status'] == 'subject_absent'
            assert marked['primary'] is None
            assert marked['subject_absent'] is True
            assert 'picked_subject' not in marked
            assert original['status'] in ('ambiguous', 'evaluated')  # originals untouched

    def test_absent_mark_outranks_a_stale_face_pick(self):
        multi = make_multi_sample(1000, 0, 42, [make_face(0, 0.6, 0.62), make_face(1, 0.4, 0.41)])
        picks = {multi['filename']: {'subject_absent': True, 'face_index': 1, 'twin_confirmed': [0]}}
        resolved, used = leakage.resolve_picks([multi], picks)
        assert used == 0
        assert resolved[0]['status'] == 'subject_absent'

    def test_failed_and_faceless_samples_never_absent(self):
        no_face = make_solo_sample(1000, 0, 42)
        no_face['status'] = 'failed'
        no_face['faces'] = []
        no_face['primary'] = None
        resolved, used = leakage.resolve_picks([no_face], {no_face['filename']: {'subject_absent': True}})
        assert used == 0
        assert resolved[0] is no_face  # passes through unchanged

    def test_absent_multi_is_not_unpicked_and_not_evaluable(self):
        sample = make_multi_sample(1000, 0, 42, [make_face(0, 0.6, 0.62), make_face(1, 0.59, 0.61)])
        block = detect([sample], {sample['filename']: {'subject_absent': True}})
        assert block['counts']['multi_face_samples'] == 1
        assert block['counts']['evaluable'] == 0
        assert block['counts']['unpicked_multi'] == 0  # resolved, not awaiting a pick
        assert block['counts']['subject_absent'] == 1
        assert block['per_step'] == []
        assert any('subject-absent' in note for note in block['notes'])

    def test_adherence_block_judges_faces_against_the_band_per_model(self):
        leaked = make_solo_sample(1000, 0, 42, arcface=0.59, sface=0.2)   # arcface >= band
        clean = make_solo_sample(1000, 1, 42, arcface=0.2, sface=0.2)     # both far below
        picks = {leaked['filename']: {'subject_absent': True}, clean['filename']: {'subject_absent': True}}
        block = detect([leaked, clean], picks)
        assert block['adherence']['marked'] == 2
        training = block['adherence']['models']
        assert training['arcface']['training'] == {'scored': 2, 'leaked': 1, 'borderline': 0, 'clean': 1, 'unscored': 0}
        assert training['sface']['training']['leaked'] == 0
        assert training['sface']['training']['clean'] == 2
        # no marks -> no adherence block at all (old reports stay shape-stable)
        assert detect([leaked], {})['adherence'] is None

    def test_absent_solo_leaves_report_stats_and_gets_its_own_bucket(self):
        good = make_solo_sample(1000, 0, 42, arcface=0.6, sface=0.62)
        stranger = make_solo_sample(1000, 1, 42, arcface=0.2, sface=0.21)
        picks = {stranger['filename']: {'subject_absent': True}}
        report = build([good, stranger], picks)
        step = report['steps'][0]
        # the stranger's 0.2 must not drag the median: only the subject sample counts
        assert step['counts']['evaluated'] == 1
        assert step['counts']['subject_absent'] == 1
        assert step['stats']['arcface']['training']['n'] == 1
        assert step['stats']['arcface']['training']['median'] == pytest.approx(0.6)
        assert report['leakage']['adherence']['marked'] == 1
        assert report['leakage']['counts']['subject_absent'] == 1


class TestDetectLeakage:
    def test_equal_embedding_face_is_likely_duplicate(self):
        # non-target face as close to the references as they are to each other
        faces = [make_face(0, 0.6, 0.62), make_face(1, 0.59, 0.61)]
        sample = make_multi_sample(1000, 0, 42, faces)
        block = detect([sample], {sample['filename']: {'face_index': 0}})
        entry = next(f for f in block['faces'] if f['face_index'] == 1)
        # the fixture scores training AND external (external = value - 0.02,
        # external LOO 0.57/0.59): 0.57 >= 0.57 -> duplicate there as well
        assert entry['levels'] == {
            'arcface': {'training': 'likely_duplicate', 'external': 'likely_duplicate'},
            'sface': {'training': 'likely_duplicate', 'external': 'likely_duplicate'},
        }
        rates = next(s for s in block['per_step'] if s['step'] == 1000)['rates']
        assert rates['arcface']['training'] == {'flagged': 1, 'evaluable': 1, 'rate': 1.0}
        assert rates['sface']['training']['flagged'] == 1

    def test_moderate_similarity_is_resembles_only(self):
        # inside the margin band but below the LOO band
        faces = [make_face(0, 0.6, 0.62), make_face(1, LOO['arcface'] - 0.05, LOO['sface'] - 0.05)]
        sample = make_multi_sample(1000, 0, 42, faces)
        block = detect([sample], {sample['filename']: {'face_index': 0}})
        entry = next(f for f in block['faces'] if f['face_index'] == 1)
        assert entry['levels']['arcface']['training'] == 'resembles'
        # resembles is informational: never counted in the rate
        rates = next(s for s in block['per_step'] if s['step'] == 1000)['rates']
        assert rates['arcface']['training'] == {'flagged': 0, 'evaluable': 1, 'rate': 0.0}

    def test_low_similarity_face_not_listed(self):
        faces = [make_face(0, 0.6, 0.62), make_face(1, 0.2, 0.21)]
        sample = make_multi_sample(1000, 0, 42, faces)
        block = detect([sample], {sample['filename']: {'face_index': 0}})
        assert block['faces'] == []
        assert block['per_step'][0]['rates']['arcface']['training']['flagged'] == 0

    def test_levels_are_per_model(self):
        # arcface at the band, sface far below: duplicate under one model only
        faces = [make_face(0, 0.6, 0.62), make_face(1, LOO['arcface'], 0.3)]
        sample = make_multi_sample(1000, 0, 42, faces)
        block = detect([sample], {sample['filename']: {'face_index': 0}})
        entry = next(f for f in block['faces'] if f['face_index'] == 1)
        assert entry['levels']['arcface']['training'] == 'likely_duplicate'
        assert 'sface' not in entry['levels']
        rates = next(s for s in block['per_step'] if s['step'] == 1000)['rates']
        assert rates['arcface']['training']['flagged'] == 1
        assert rates['sface']['training']['flagged'] == 0

    def test_user_twin_mark_counts_and_outranks_scores(self):
        # far below every band, but the user marked it
        faces = [make_face(0, 0.6, 0.62), make_face(1, 0.2, 0.21)]
        sample = make_multi_sample(1000, 0, 42, faces)
        picks = {sample['filename']: {'face_index': 0, 'twin_confirmed': [1]}}
        block = detect([sample], picks)
        entry = next(f for f in block['faces'] if f['face_index'] == 1)
        assert entry['user_twin'] is True and entry['levels'] == {}
        rates = next(s for s in block['per_step'] if s['step'] == 1000)['rates']
        for model in ('arcface', 'sface'):
            assert rates[model]['training']['flagged'] == 1

    def test_twin_mark_without_subject_pick_is_ignored(self):
        faces = [make_face(0, 0.6, 0.62), make_face(1, 0.6, 0.62)]
        sample = make_multi_sample(1000, 0, 42, faces)
        block = detect([sample], {sample['filename']: {'twin_confirmed': [1]}})  # no face_index
        assert block['faces'] == []
        assert block['counts']['unpicked_multi'] == 1
        assert block['per_step'] == []
        assert any('no subject-face pick' in note for note in block['notes'])

    def test_unpicked_multi_samples_not_evaluable(self):
        sample = make_multi_sample(1000, 0, 42, [make_face(0, 0.6, 0.62), make_face(1, 0.59, 0.61)])
        block = detect([sample], {})
        assert block['counts']['multi_face_samples'] == 1
        assert block['counts']['evaluable'] == 0
        assert block['per_step'] == []

    def test_rates_never_pool_raw_and_ema(self):
        ema = make_multi_sample(1000, 0, 42, [make_face(0, 0.6, 0.62), make_face(1, 0.2, 0.21)])
        raw = make_multi_sample(1000, 1, 42, [make_face(0, 0.6, 0.62), make_face(1, 0.59, 0.61)], is_raw=True)
        picks = {ema['filename']: {'face_index': 0}, raw['filename']: {'face_index': 0}}
        block = detect([ema, raw], picks)
        by_kind = {(item['step'], item['is_raw']): item['rates'] for item in block['per_step']}
        assert by_kind[(1000, False)]['arcface']['training']['flagged'] == 0
        assert by_kind[(1000, True)]['arcface']['training']['flagged'] == 1

    def test_thresholds_block_exposes_bands_per_model(self):
        sample = make_multi_sample(1000, 0, 42, [make_face(0, 0.6, 0.62), make_face(1, 0.2, 0.21)])
        block = detect([sample], {sample['filename']: {'face_index': 0}})
        assert block['thresholds']['arcface']['training'] == {'loo': 0.58, 'resembles_min': round(0.58 - leakage.RESEMBLES_MARGIN, 3)}
        assert block['thresholds']['sface']['training']['loo'] == 0.6

    def test_closest_ref_passthrough(self):
        closest = {'arcface': {'training': {'key': 'e:/refs/a.jpg', 'face_index': 0, 'sim': 0.61}}}
        faces = [make_face(0, 0.6, 0.62), make_face(1, 0.59, 0.61, closest=closest)]
        sample = make_multi_sample(1000, 0, 42, faces)
        block = detect([sample], {sample['filename']: {'face_index': 0}})
        entry = next(f for f in block['faces'] if f['face_index'] == 1)
        assert entry['closest_ref'] == closest

    def test_solo_samples_do_not_participate(self):
        block = detect([make_solo_sample(1000, 0, 42)], {})
        assert block['faces'] == [] and block['per_step'] == []
        assert block['counts']['multi_face_samples'] == 0

    def test_resembles_list_is_capped_but_duplicates_are_not(self):
        samples = []
        picks = {}
        for p in range(leakage.MAX_RESEMBLES_LISTED + 5):
            deep = make_multi_sample(1000, p, 42, [make_face(0, 0.6, 0.62), make_face(1, 0.59, 0.61)])  # duplicates
            mild = make_multi_sample(2000, p, 42, [make_face(0, 0.6, 0.62), make_face(1, 0.56, 0.57)])  # resembles
            samples += [deep, mild]
            picks[deep['filename']] = {'face_index': 0}
            picks[mild['filename']] = {'face_index': 0}
        block = detect(samples, picks)
        listed_levels = [f for f in block['faces'] if 'levels' in f and f['levels']]
        duplicates = [f for f in listed_levels if any(l == 'likely_duplicate' for m in f['levels'].values() for l in m.values())]
        resembles = [f for f in listed_levels if all(l == 'resembles' for m in f['levels'].values() for l in m.values())]
        assert len(duplicates) == leakage.MAX_RESEMBLES_LISTED + 5
        assert len(resembles) == leakage.MAX_RESEMBLES_LISTED
        assert any('not listed' in note for note in block['notes'])


class TestReportIntegration:
    def test_picked_samples_join_stats_and_counts(self):
        # 0.50 solo + ambiguous sample whose face 1 scores 0.60: picking it
        # moves the training median from 0.50 to 0.60
        solo = make_solo_sample(1000, 0, 42, arcface=0.5, sface=0.52)
        multi = make_multi_sample(1000, 1, 42, [make_face(0, 0.2, 0.21), make_face(1, 0.6, 0.62)])
        picks = {multi['filename']: {'face_index': 1}}
        without = build([solo, multi])
        with_pick = build([solo, multi], picks)
        assert without['steps'][0]['counts']['ambiguous'] == 1
        assert without['steps'][0]['stats']['arcface']['training']['median'] == pytest.approx(0.5)
        assert with_pick['steps'][0]['counts']['ambiguous'] == 0
        assert with_pick['steps'][0]['counts']['evaluated'] == 2
        assert with_pick['steps'][0]['counts']['picked'] == 1
        assert with_pick['steps'][0]['stats']['arcface']['training']['median'] == pytest.approx(0.55)
        assert with_pick['steps'][0]['strip']['arcface']['training'][1]['f'] == multi['filename']

    def test_step_entries_carry_leakage_rates(self):
        flagged = make_multi_sample(1000, 0, 42, [make_face(0, 0.6, 0.62), make_face(1, 0.59, 0.61)])
        picks = {flagged['filename']: {'face_index': 0}}
        report = build([flagged], picks)
        assert report['steps'][0]['leakage']['arcface']['training'] == {'flagged': 1, 'evaluable': 1, 'rate': 1.0}
        assert report['leakage']['version'] == leakage.LEAKAGE_VERSION
        assert report['leakage']['picks_used'] == 1
        assert any('subject-face picks' in note for note in report['notes'])

    def test_solo_only_report_has_no_leakage_keys(self):
        report = build([make_solo_sample(1000, 0, 42)])
        assert 'leakage' not in report['steps'][0]
        assert report['leakage']['faces'] == []
        assert report['leakage']['per_step'] == []

    def test_picks_missing_loo_band_skip_levels_with_note(self):
        sample = make_multi_sample(1000, 0, 42, [make_face(0, 0.6, 0.62), make_face(1, 0.59, 0.61)])
        results = make_results([sample])
        results['refsets'] = {'training': {'n': 1, 'hash': 'x', 'skipped': {}, 'loo': {'arcface': None, 'sface': None}}, 'external': None}
        block = leakage.detect_leakage(results, {sample['filename']: {'face_index': 0}}, ['arcface', 'sface'], ['training', 'external'])
        assert block['thresholds'] == {}
        assert block['faces'] == []  # no band -> no statistical level
        assert any('leave-one-out band' in note for note in block['notes'])


class TestIntegrity:
    def test_detection_is_deterministic(self):
        samples = [make_multi_sample(1000, 0, 42, [make_face(0, 0.6, 0.62), make_face(1, 0.59, 0.61)])]
        picks = {samples[0]['filename']: {'face_index': 0}}
        assert detect(samples, picks) == detect(samples, picks)

    def test_wording_and_schema_rules(self):
        # the tool never asserts a twin, never calls anything a clone, and
        # the strongest statistical wording stays hedged
        samples = [
            make_multi_sample(1000, 0, 42, [make_face(0, 0.6, 0.62), make_face(1, 0.59, 0.61)]),
            make_multi_sample(1000, 1, 42, [make_face(0, 0.6, 0.62), make_face(1, 0.2, 0.21)]),
        ]
        picks = {samples[0]['filename']: {'face_index': 0}, samples[1]['filename']: {'face_index': 0, 'twin_confirmed': [1]}}
        report = build(samples, picks)
        blob = json.dumps(report['leakage']).lower()
        assert 'confirmed' not in blob
        assert 'clone' not in blob
        assert '"best"' not in json.dumps(report).lower()
        for face in report['leakage']['faces']:
            assert set(face.keys()) == {
                'filename', 'sample_path', 'step', 'is_raw', 'subject_face', 'face_index', 'det_score',
                'user_twin', 'levels', 'scores', 'closest_ref',
            }
            for per_model in face['levels'].values():
                assert set(per_model.values()) <= {'resembles', 'likely_duplicate'}
            assert isinstance(face['user_twin'], bool)
        assert report['leakage']['version'] == leakage.LEAKAGE_VERSION

    def test_arcface_only_input_yields_arcface_only_levels(self):
        samples = [make_multi_sample(1000, 0, 42, [make_face(0, 0.6, None), make_face(1, 0.59, None)])]
        picks = {samples[0]['filename']: {'face_index': 0}}
        block = leakage.detect_leakage(make_results(samples), picks, ['arcface'], ['training'])
        entry = next(f for f in block['faces'] if f['face_index'] == 1)
        assert set(entry['levels']) == {'arcface'}
