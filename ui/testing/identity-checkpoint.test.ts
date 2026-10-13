// Checkpoint comparison (Phase 2b) pure-logic tests: verdict phrasing,
// category sanitizing, strip domain math and prompt-index extraction.
import assert from 'node:assert/strict';
import test from 'node:test';
import {
  countsLabel,
  convergencePhrase,
  excludedReasonLabel,
  fmtScore,
  fmtSigned,
  invertPairComparison,
  pairInOrientation,
  promptIndicesInReport,
  reportFreshByMtimes,
  sanitizeCategories,
  stripDomain,
  stripX,
  stepValue,
  verdictPhrase,
  type CheckpointReport,
  type PairComparison,
  type PairModelComparison,
} from '../src/utils/identityCheckpoint';

const comparison = (over: Partial<PairModelComparison>): PairModelComparison => ({
  n_pairs: 4,
  diff_median: 0.04,
  ci_low: 0.03,
  ci_high: 0.05,
  wins_b: 4,
  wins_a: 0,
  ties: 0,
  verdict: 'higher',
  excluded: [],
  ...over,
});

test('verdictPhrase: higher only when CI excludes zero, names the checkpoints', () => {
  const phrase = verdictPhrase(comparison({}), 'arcface', '1500 (EMA)', '2000 (EMA)');
  assert.equal(phrase, '2000 (EMA) shows higher ArcFace identity than 1500 (EMA) on comparable prompts (CI excludes 0)');
  assert.ok(!phrase.toLowerCase().includes('best'));
});

test('verdictPhrase: lower direction names the earlier checkpoint', () => {
  const phrase = verdictPhrase(comparison({ verdict: 'lower', diff_median: -0.02, ci_low: -0.03, ci_high: -0.01, wins_a: 4, wins_b: 0 }), 'sface', '1000', '1500');
  assert.equal(phrase, '1000 shows higher SFace identity than 1500 on comparable prompts (CI excludes 0)');
});

test('verdictPhrase: CI including zero says no detectable difference', () => {
  const phrase = verdictPhrase(comparison({ verdict: 'none', diff_median: 0.001, ci_low: -0.01, ci_high: 0.012, wins_b: 2, wins_a: 2 }), 'arcface', '1000', '1500');
  assert.equal(phrase, 'no detectable ArcFace identity difference between 1000 and 1500 on comparable prompts');
});

test('verdictPhrase: too few pairs is explicit, never a winner', () => {
  const phrase = verdictPhrase(comparison({ verdict: 'insufficient', n_pairs: 1 }), 'arcface', '1000', '1500');
  assert.equal(phrase, 'too few comparable pairs (1) to judge ArcFace identity');
  const none = verdictPhrase(comparison({ n_pairs: 0, verdict: 'insufficient', diff_median: null, ci_low: null, ci_high: null }), 'arcface', '1000', '1500');
  assert.ok(none.startsWith('no comparable'));
});

test('excludedReasonLabel maps machine reasons to readable text', () => {
  assert.ok(excludedReasonLabel('missing_at_a').includes('earlier'));
  assert.ok(excludedReasonLabel('missing_at_b').includes('later'));
  assert.ok(excludedReasonLabel('not_evaluated').includes('evaluable'));
  assert.equal(excludedReasonLabel('weird'), 'weird');
});

test('countsLabel joins evaluated/ambiguous/no-face/failed', () => {
  assert.equal(countsLabel({ evaluated: 12, ambiguous: 1, no_face: 2, failed: 0 }), '12/1/2/0');
});

test('fmtScore / fmtSigned format three decimals and signs', () => {
  assert.equal(fmtScore(0.4415), '0.442');
  assert.equal(fmtScore(null), '–');
  assert.equal(fmtSigned(0.02), '+0.020');
  assert.equal(fmtSigned(-0.02), '-0.020');
  assert.equal(fmtSigned(null), '–');
});

test('sanitizeCategories mirrors the route rules', () => {
  const clean = sanitizeCategories({ '0': '  solo ', '1': 'multi', 'x': 'nope', '2': '', '3': 5 as unknown as string, '4': 'a'.repeat(60) });
  assert.deepEqual(clean, { '0': 'solo', '1': 'multi', '4': 'a'.repeat(40) });
  assert.deepEqual(sanitizeCategories(null), {});
});

test('convergencePhrase stays on the loss question', () => {
  assert.ok(convergencePhrase({ converged: true } as never).includes('converged'));
  assert.ok(convergencePhrase({ converged: false } as never).includes('gaining'));
});

const report = (over: Partial<CheckpointReport>): CheckpointReport => ({
  version: 1,
  job: 'job_t',
  generated_at: 'now',
  results_completed: true,
  sources: ['training'],
  models: ['arcface'],
  refsets: { training: { n: 5, loo: { arcface: 0.58 } } },
  prompt_texts: { '0': 'a woman', '1': 'two women' },
  categories: {},
  steps: [],
  pairs: [],
  convergence: null,
  notes: [],
  ...over,
});

test('promptIndicesInReport unions config prompts and excluded pair ids', () => {
  const doc = report({
    prompt_texts: { '0': 'a', '1': 'b' },
    pairs: [
      {
        a: { step: 1000, is_raw: false, label: '1000' },
        b: { step: 2000, is_raw: false, label: '2000' },
        models: {
          'arcface:training': comparison({
            excluded: [
              { id: 'step 1000 · prompt 3 · seed 42', reason: 'missing_at_b' },
              { id: 'step 2000 · prompt 2 · seed 9', reason: 'missing_at_a' },
            ],
          }),
        },
      },
    ],
    steps: [
      {
        step: 1000,
        is_raw: false,
        label: '1000',
        counts: { evaluated: 0, ambiguous: 0, no_face: 0, failed: 0 },
        n_prompts: 0,
        stats: {},
        by_category: {},
        strip: {
          arcface: {
            training: [
              { f: 'x__000001000_07_seed-42_steps-20.jpg', s: 0.4 },
              { f: 'x__000001000_08_seed-42_steps-20.jpg', s: 0.41 },
            ],
          },
        },
      },
    ],
  });
  assert.deepEqual(promptIndicesInReport(doc), [0, 1, 2, 3, 7, 8]);
});

test('stripDomain pads and collapses degenerate ranges', () => {
  const steps = [
    {
      step: 1000,
      is_raw: false,
      label: '1000',
      counts: { evaluated: 0, ambiguous: 0, no_face: 0, failed: 0 },
      n_prompts: 0,
      stats: {},
      by_category: {},
      strip: { arcface: { training: [{ f: 'a.jpg', s: 0.4 }, { f: 'b.jpg', s: 0.5 }] } },
    },
  ] as CheckpointReport['steps'];
  const domain = stripDomain(steps, 'arcface', 'training');
  assert.ok(domain.min < 0.4 && domain.max > 0.5);
  // degenerate: single distinct score
  const flat = stripDomain([{ ...steps[0], strip: { arcface: { training: [{ f: 'a.jpg', s: 0.4 }] } } }] as never, 'arcface', 'training');
  assert.ok(flat.min < 0.4 && flat.max > 0.4);
  // empty: falls back around 0
  const empty = stripDomain([], 'arcface', 'training');
  assert.ok(empty.min === -0.02 && empty.max === 0.02);
});

test('stripX maps scores into the plot width', () => {
  const domain = { min: 0.3, max: 0.5 };
  assert.ok(Math.abs(stripX(0.3, domain, 100) - 4) < 1e-9);
  assert.ok(Math.abs(stripX(0.5, domain, 100) - 96) < 1e-9);
  const mid = stripX(0.4, domain, 100);
  assert.ok(mid > 4 && mid < 96);
});

// ------------------------------------------------------- review-fix regressions

test('reportFreshByMtimes: newer loss log (main db or WAL) invalidates the cache', () => {
  const fresh = { results: 100, categories: 90, loss: 80, lossWal: -1, report: 120 };
  assert.equal(reportFreshByMtimes(fresh), true);
  // training advanced the loss log without touching results.json
  assert.equal(reportFreshByMtimes({ ...fresh, loss: 130 }), false);
  // WAL-mode write: only the -wal sidecar moved
  assert.equal(reportFreshByMtimes({ ...fresh, lossWal: 130 }), false);
  // no loss log at all (job never trained locally)
  assert.equal(reportFreshByMtimes({ ...fresh, loss: -1 }), true);
  // newer results or categories also invalidate
  assert.equal(reportFreshByMtimes({ ...fresh, results: 130 }), false);
  assert.equal(reportFreshByMtimes({ ...fresh, categories: 130 }), false);
  // missing results or report -> never fresh
  assert.equal(reportFreshByMtimes({ ...fresh, results: -1 }), false);
  assert.equal(reportFreshByMtimes({ ...fresh, report: -1 }), false);
});

test('pairInOrientation serves a reversed selection by inverting the stored pair', () => {
  const stored: PairComparison = {
    a: { step: 1000, is_raw: false, label: '1000' },
    b: { step: 2000, is_raw: false, label: '2000' },
    models: {
      'arcface:training': comparison({
        diff_median: 0.04,
        ci_low: 0.03,
        ci_high: 0.05,
        wins_a: 0,
        wins_b: 4,
        verdict: 'higher',
        excluded: [{ id: 'step 1000 · prompt 3 · seed 42', reason: 'missing_at_b', status_a: 'evaluated' }],
      }),
    },
  };
  const doc = report({ pairs: [stored] });
  // exact orientation returns the stored pair unchanged
  assert.equal(pairInOrientation(doc, stepValue(stored.a), stepValue(stored.b)), stored);
  // reversed selection: same endpoints, inverted evidence
  const reversed = pairInOrientation(doc, stepValue(stored.b), stepValue(stored.a));
  assert.ok(reversed);
  assert.equal(reversed.a.label, '2000');
  assert.equal(reversed.b.label, '1000');
  const cmp = reversed.models['arcface:training'];
  assert.equal(cmp.diff_median, -0.04);
  assert.equal(cmp.ci_low, -0.05); // bounds swap AND negate
  assert.equal(cmp.ci_high, -0.03);
  assert.equal(cmp.wins_a, 4);
  assert.equal(cmp.wins_b, 0);
  assert.equal(cmp.verdict, 'lower');
  assert.equal(reversed.models['arcface:training'].excluded[0].status_b, 'evaluated');
  // verdict phrasing follows the inversion: 2000 remains the higher checkpoint
  const phrase = verdictPhrase(cmp, 'arcface', reversed.a.label, reversed.b.label);
  assert.equal(phrase, '2000 shows higher ArcFace identity than 1000 on comparable prompts (CI excludes 0)');
  // double inversion round-trips
  const roundTrip = invertPairComparison(reversed);
  assert.equal(roundTrip.a.label, '1000');
  assert.equal(roundTrip.models['arcface:training'].diff_median, 0.04);
  assert.equal(roundTrip.models['arcface:training'].verdict, 'higher');
  // unrelated selection stays null
  assert.equal(pairInOrientation(doc, '3000:0', '4000:0'), null);
});

test('promptIndicesInReport accepts prompt indices of 100+ in strip filenames', () => {
  const doc = report({
    prompt_texts: {},
    steps: [
      {
        step: 1000,
        is_raw: false,
        label: '1000',
        counts: { evaluated: 0, ambiguous: 0, no_face: 0, failed: 0 },
        n_prompts: 0,
        stats: {},
        by_category: {},
        strip: {
          arcface: {
            training: [
              { f: 'x__000001000_99_seed-42_steps-20.jpg', s: 0.4 },
              { f: 'x__000001000_100_seed-42_steps-20.jpg', s: 0.41 }, // :02d pads to minimum width
              { f: 'x__000001000_007_seed-42_steps-20.jpg', s: 0.42 },
            ],
          },
        },
      },
    ],
  });
  assert.deepEqual(promptIndicesInReport(doc), [7, 99, 100]);
});
