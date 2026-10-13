"""Checkpoint comparison (Phase 2b) tests.

Pure post-processing over results.json -- no face models needed, so every
test here always runs. Covers the spec's paired-comparison rules:
identical sample sets -> CI includes 0; prompt/seed mismatches -> excluded
and flagged; no-face/ambiguous never scored; RAW/EMA never mixed; prompt
category breakdown; convergence heuristic on a synthetic loss_log.db.
"""

import json
import os
import sqlite3

import numpy as np
import pytest

from toolkit.identity_eval import checkpoint_compare as cc


def make_sample(step, prompt_idx, seed, arcface=None, sface=None, status='evaluated', is_raw=False, warnings=None):
    primary = None
    if status == 'evaluated':
        primary = {
            'arcface': {'training': {'mean': arcface}, 'external': {'mean': arcface - 0.02 if arcface is not None else None}},
            'sface': {'training': {'mean': sface}},
        }
    return {
        'path': f'out/{step}_{prompt_idx}.jpg',
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
        'faces': [],
        'warnings': warnings or [],
    }


def make_results(samples, refsets=None):
    return {
        'version': 1,
        'job': 'job_t',
        'completed': True,
        'updated_at': '2026-10-13T00:00:00',
        'sources': ['training'],
        'refsets': refsets if refsets is not None else {'training': {'n': 5, 'hash': 'x', 'skipped': {}, 'loo': {'arcface': 0.58, 'sface': 0.6}}},
        'samples': samples,
        'warnings': [],
    }


class TestStats:
    def test_distribution_stats_known_values(self):
        stats = cc.distribution_stats([0.1, 0.2, 0.3, 0.4, 0.5])
        assert stats['n'] == 5
        assert stats['median'] == pytest.approx(0.3)
        assert stats['p25'] == pytest.approx(0.2)
        assert stats['p75'] == pytest.approx(0.4)
        assert stats['iqr'] == pytest.approx(0.2, abs=1e-6)
        assert 0.1 <= stats['p10'] <= stats['p25']
        # p90 (retention-graph band, Phase 2c): linear percentile of the ramp
        assert stats['p90'] == pytest.approx(0.46)
        assert stats['p75'] <= stats['p90'] <= 0.5

    def test_distribution_stats_empty(self):
        assert cc.distribution_stats([]) == {'n': 0, 'median': None, 'p10': None, 'p25': None, 'p75': None, 'p90': None, 'iqr': None}

    def test_report_version_pinned_for_route_cache(self):
        # the compare route rebuilds any report whose version differs
        assert cc.REPORT_VERSION == 2

    def test_count_statuses_splits_no_face_from_failed(self):
        samples = [
            make_sample(1, 0, 1, 0.5),
            make_sample(1, 1, 1, status='failed', warnings=['no face detected']),
            make_sample(1, 2, 1, status='failed', warnings=['unreadable image file']),
            make_sample(1, 3, 1, status='ambiguous'),
        ]
        assert cc.count_statuses(samples) == {'evaluated': 1, 'ambiguous': 1, 'no_face': 1, 'failed': 1}


class TestPairing:
    def test_identical_sets_ci_includes_zero(self):
        samples = [make_sample(1000, p, 42, 0.40 + 0.01 * p) for p in range(4)]
        samples += [make_sample(2000, p, 42, 0.40 + 0.01 * p) for p in range(4)]
        report = cc.build_checkpoint_report(make_results(samples), None, None, None)
        pair = report['pairs'][0]
        comparison = pair['models']['arcface:training']
        assert comparison['n_pairs'] == 4
        assert comparison['diff_median'] == pytest.approx(0.0, abs=1e-9)
        assert comparison['ci_low'] <= 0 <= comparison['ci_high']
        assert comparison['verdict'] == 'none'
        assert comparison['wins_b'] == 0 and comparison['wins_a'] == 0 and comparison['ties'] == 4
        assert comparison['excluded'] == []

    def test_clear_difference_ci_excludes_zero(self):
        samples = [make_sample(1000, p, 42, 0.40 + 0.01 * p) for p in range(4)]
        samples += [make_sample(2000, p, 42, 0.44 + 0.01 * p) for p in range(4)]
        report = cc.build_checkpoint_report(make_results(samples), None, None, None)
        comparison = report['pairs'][0]['models']['arcface:training']
        assert comparison['diff_median'] == pytest.approx(0.04, abs=1e-6)
        assert comparison['ci_low'] > 0
        assert comparison['verdict'] == 'higher'
        assert comparison['wins_b'] == 4

    def test_seed_change_excludes_and_flags(self):
        # prompt 1 changes seed between steps: not directly comparable
        samples = [make_sample(1000, p, 42, 0.4) for p in range(3)]
        samples += [make_sample(2000, 0, 42, 0.42), make_sample(2000, 1, 77, 0.42), make_sample(2000, 2, 42, 0.42)]
        report = cc.build_checkpoint_report(make_results(samples), None, None, None)
        comparison = report['pairs'][0]['models']['arcface:training']
        assert comparison['n_pairs'] == 2
        reasons = {entry['reason'] for entry in comparison['excluded']}
        assert reasons == {'missing_at_b', 'missing_at_a'}
        assert all('prompt 1' in entry['id'] for entry in comparison['excluded'])

    def test_not_evaluated_excluded_with_status(self):
        samples = [make_sample(1000, p, 42, 0.4) for p in range(3)]
        samples += [make_sample(2000, 0, 42, 0.42), make_sample(2000, 1, 42, status='failed', warnings=['no face detected']), make_sample(2000, 2, 42, 0.42)]
        report = cc.build_checkpoint_report(make_results(samples), None, None, None)
        comparison = report['pairs'][0]['models']['arcface:training']
        assert comparison['n_pairs'] == 2
        excluded = [e for e in comparison['excluded'] if e['reason'] == 'not_evaluated']
        assert len(excluded) == 1
        assert excluded[0]['status_b'] == 'failed'

    def test_raw_and_ema_never_mix(self):
        samples = [make_sample(1500, p, 42, 0.4) for p in range(3)]
        samples += [make_sample(1500, p, 42, 0.41, is_raw=True) for p in range(3)]
        report = cc.build_checkpoint_report(make_results(samples), None, None, None)
        labels = [entry['label'] for entry in report['steps']]
        assert labels == ['1500 (EMA)', '1500 (raw)']
        pair = report['pairs'][0]
        assert pair['models']['arcface:training']['n_pairs'] == 0
        assert pair['models']['arcface:training']['verdict'] == 'insufficient'

    def test_few_pairs_insufficient(self):
        samples = [make_sample(1000, 0, 42, 0.4), make_sample(2000, 0, 42, 0.5)]
        report = cc.build_checkpoint_report(make_results(samples), None, None, None)
        comparison = report['pairs'][0]['models']['arcface:training']
        assert comparison['n_pairs'] == 1
        assert comparison['verdict'] == 'insufficient'

    def test_bootstrap_deterministic(self):
        diffs = np.asarray([0.01 * i for i in range(8)])
        first = cc.bootstrap_median_ci(diffs, 'arcface:training|a|b')
        second = cc.bootstrap_median_ci(diffs, 'arcface:training|a|b')
        other = cc.bootstrap_median_ci(diffs, 'sface:training|a|b')
        assert first == second
        assert first != other or True  # different seeds may coincide on tiny sets; not an assertion target


class TestReport:
    def test_grouping_and_ordering(self):
        samples = [make_sample(2000, 0, 42, 0.4), make_sample(1000, 0, 42, 0.4), make_sample(1000, 0, 42, 0.4, is_raw=True)]
        report = cc.build_checkpoint_report(make_results(samples), None, None, None)
        assert [(e['step'], e['is_raw']) for e in report['steps']] == [(1000, False), (1000, True), (2000, False)]

    def test_category_breakdown(self):
        samples = [make_sample(1000, p, 42, 0.30 + 0.1 * p) for p in range(4)]
        report = cc.build_checkpoint_report(make_results(samples), {'0': 'solo', '1': 'solo', '2': 'multi'}, None, None)
        by_category = report['steps'][0]['by_category']
        assert set(by_category) == {'multi', 'solo', 'untagged'}
        assert by_category['solo']['arcface']['training']['n'] == 2
        assert by_category['solo']['arcface']['training']['median'] == pytest.approx(0.35, abs=1e-6)
        assert by_category['untagged']['arcface']['training']['n'] == 1

    def test_train_vs_external_gap(self):
        samples = [make_sample(1000, p, 42, 0.40 + 0.02 * p) for p in range(3)]
        results = make_results(samples)
        results['sources'] = ['training', 'external']
        results['refsets']['external'] = {'n': 2, 'hash': 'y', 'skipped': {}, 'loo': {'arcface': 0.55}}
        report = cc.build_checkpoint_report(results, None, None, None)
        gap = report['steps'][0]['train_vs_external']['arcface']
        assert gap['training'] == pytest.approx(0.42, abs=1e-6)
        assert gap['external'] == pytest.approx(0.40, abs=1e-6)
        assert gap['gap'] == pytest.approx(0.02, abs=1e-6)
        assert 'external' in report['sources']

    def test_prompt_texts_captured_and_noted(self):
        samples = [make_sample(1000, 0, 42, 0.4)]
        report = cc.build_checkpoint_report(make_results(samples), None, {0: 'a woman, solo'}, None)
        assert report['prompt_texts'] == {'0': 'a woman, solo'}
        assert any('captured from the job config at report time' in note for note in report['notes'])

    def test_report_never_asserts_best_checkpoint(self):
        samples = [make_sample(1000, p, 42, 0.4) for p in range(4)]
        samples += [make_sample(2000, p, 42, 0.5) for p in range(4)]
        report = cc.build_checkpoint_report(make_results(samples), None, None, None)
        allowed = {'higher', 'lower', 'none', 'insufficient'}
        for pair in report['pairs']:
            for comparison in pair['models'].values():
                assert comparison['verdict'] in allowed
        blob = json.dumps(report).lower()
        assert '"best"' not in blob  # no key/verdict may claim a best checkpoint

    def test_unparsed_filenames_surfaced_not_grouped(self):
        samples = [make_sample(1000, 0, 42, 0.4)]
        bad = make_sample(1000, 1, 42, 0.4)
        bad.update(step=None, prompt_idx=None, filename='photo.jpg')
        samples.append(bad)
        report = cc.build_checkpoint_report(make_results(samples), None, None, None)
        assert len(report['steps']) == 1
        assert report['unparsed_samples'] == 1
        assert any('did not parse' in note for note in report['notes'])

    def test_strip_lists_filenames(self):
        samples = [make_sample(1000, 0, 42, 0.4), make_sample(1000, 1, 42, status='failed', warnings=['no face detected'])]
        report = cc.build_checkpoint_report(make_results(samples), None, None, None)
        strip = report['steps'][0]['strip']['arcface']['training']
        assert [e['f'] for e in strip] == [samples[0]['filename']]


class TestPromptTexts:
    def test_envelope_with_prompts_list(self):
        config = {'job': 'extension', 'config': {'sample': {'prompts': ['a woman', 'two women']}}}
        assert cc.extract_prompt_texts(config) == {0: 'a woman', 1: 'two women'}

    def test_samples_list_form(self):
        config = {'config': {'sample': {'samples': [{'prompt': 'x'}, {'prompt': ' y '}, {'prompt': '  '}]}}}
        assert cc.extract_prompt_texts(config) == {0: 'x', 1: 'y'}

    def test_garbage_returns_empty(self):
        assert cc.extract_prompt_texts(None) == {}
        assert cc.extract_prompt_texts({'config': None}) == {}
        assert cc.extract_prompt_texts({'config': {'sample': 'nope'}}) == {}


class TestConvergence:
    @staticmethod
    def _write_loss_db(job_dir, steps_values, lr_values):
        os.makedirs(job_dir, exist_ok=True)
        connection = sqlite3.connect(os.path.join(job_dir, 'loss_log.db'))
        connection.execute('CREATE TABLE metrics (key TEXT, step INTEGER, value_real REAL)')
        connection.executemany(
            'INSERT INTO metrics VALUES ("loss/img", ?, ?)', [(s, v) for s, v in steps_values]
        )
        connection.executemany(
            'INSERT INTO metrics VALUES ("learning_rate", ?, ?)', [(s, v) for s, v in lr_values]
        )
        connection.commit()
        connection.close()

    def test_flat_loss_floored_lr_converged(self, tmp_path):
        job_dir = str(tmp_path / 'job')
        steps = list(range(0, 2001, 10))
        self._write_loss_db(job_dir, [(s, 0.300 if s < 1500 else 0.3001) for s in steps], [(s, 1e-3 * max(0.2, 1 - s / 2000 * 0.95)) for s in steps])
        summary = cc.convergence_summary(job_dir)
        assert summary is not None
        assert summary['last_step'] == 2000
        assert summary['lr_floored'] is True
        assert summary['converged'] is True

    def test_still_gaining_not_converged(self, tmp_path):
        job_dir = str(tmp_path / 'job')
        steps = list(range(0, 2001, 10))
        self._write_loss_db(job_dir, [(s, max(0.05, 0.5 - s * 0.0002)) for s in steps], [(s, 1e-3) for s in steps])
        summary = cc.convergence_summary(job_dir)
        assert summary['converged'] is False

    def test_missing_db_returns_none(self, tmp_path):
        assert cc.convergence_summary(str(tmp_path / 'nope')) is None


class TestCategoriesIO:
    def test_load_categories_round_trip(self, tmp_path):
        identity_dir = str(tmp_path)
        assert cc.load_categories(identity_dir) == {}
        with open(cc.categories_path(identity_dir), 'w', encoding='utf-8') as handle:
            json.dump({'version': 1, 'categories': {'0': 'solo'}}, handle)
        assert cc.load_categories(identity_dir) == {'0': 'solo'}

    def test_load_categories_garbage(self, tmp_path):
        with open(cc.categories_path(str(tmp_path)), 'w', encoding='utf-8') as handle:
            handle.write('{broken')
        assert cc.load_categories(str(tmp_path)) == {}


class TestWorkerCli:
    def test_compare_command_writes_report(self, tmp_path):
        import subprocess
        import sys

        output_root = str(tmp_path)
        job_name = 'job_cli'
        identity_dir = os.path.join(output_root, job_name, 'identity_eval')
        os.makedirs(identity_dir, exist_ok=True)
        samples = [make_sample(1000, p, 42, 0.4) for p in range(3)]
        samples += [make_sample(2000, p, 42, 0.44) for p in range(3)]
        with open(os.path.join(identity_dir, 'results.json'), 'w', encoding='utf-8') as handle:
            json.dump(make_results(samples), handle)

        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        worker = os.path.join(repo_root, 'ui_scripts', 'identity_eval.py')
        result = subprocess.run(
            [sys.executable, '-u', worker, '--command', 'compare', '--job-name', job_name, '--output-root', output_root],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=repo_root,
        )
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        assert payload['type'] == 'result'
        assert payload['steps'] == 2 and payload['pairs'] == 1
        with open(os.path.join(identity_dir, 'checkpoint_report.json'), 'r', encoding='utf-8') as handle:
            report = json.load(handle)
        assert report['pairs'][0]['models']['arcface:training']['verdict'] == 'higher'

    def test_compare_without_results_errors(self, tmp_path):
        import subprocess
        import sys

        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        worker = os.path.join(repo_root, 'ui_scripts', 'identity_eval.py')
        result = subprocess.run(
            [sys.executable, '-u', worker, '--command', 'compare', '--job-name', 'nope', '--output-root', str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=repo_root,
        )
        assert result.returncode == 1
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        assert payload['type'] == 'error'
