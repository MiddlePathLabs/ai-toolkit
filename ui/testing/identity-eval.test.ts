import assert from 'node:assert/strict';
import test from 'node:test';
import {
  buildIdentityIndex,
  groupSummariesByStep,
  identityFilenameKey,
  identityStepPrompt,
  mergeSelection,
  normalizeRefPath,
  overlayInfo,
  rankDisagreements,
  sampleScore,
  selectionForImage,
  sortPathsByScore,
  type IdentitySampleResult,
  type IdentitySelection,
} from '../src/utils/identityEval';

const sample = (over: Partial<IdentitySampleResult> & { filename: string }): IdentitySampleResult => ({
  path: `/out/samples/${over.filename}`,
  step: null,
  prompt_idx: null,
  seed: null,
  is_raw: false,
  status: 'evaluated',
  face_count: 1,
  error: null,
  primary: over.primary ?? {
    arcface: { training: { mean: 0.5, max: 0.6 } },
    sface: { training: { mean: 0.4, max: 0.5 } },
  },
  faces: [],
  warnings: [],
  ...over,
});

const mean = (arcface: number | null, sface: number | null, source: 'training' | 'combined' = 'training') => ({
  arcface: arcface === null ? {} : { [source]: { mean: arcface, max: arcface + 0.1 } },
  sface: sface === null ? {} : { [source]: { mean: sface, max: sface + 0.1 } },
});

test('filename key and step/prompt parsing', () => {
  const path = 'E:\\out\\samples\\1791073894128__000001500_03_seed-42_steps-20.jpg';
  assert.equal(identityFilenameKey(path), '1791073894128__000001500_03_seed-42_steps-20.jpg');
  assert.deepEqual(identityStepPrompt(path), { step: 1500, promptIdx: 3 });
  assert.equal(identityStepPrompt('/out/photo.jpg').step, null);
});

test('buildIdentityIndex is keyed by filename and tolerates null results', () => {
  const results = {
    version: 1,
    job: 'j',
    completed: true,
    created_at: '',
    updated_at: '',
    sources: ['training'],
    refsets: {},
    samples: [sample({ filename: 'a.jpg' }), sample({ filename: 'b.jpg', status: 'failed', primary: null })],
  } as never;
  const index = buildIdentityIndex(results);
  assert.equal(index.size, 2);
  assert.equal(index.get('b.jpg')?.status, 'failed');
  assert.equal(buildIdentityIndex(null).size, 0);
});

test('per-step summary grouping counts statuses and takes medians', () => {
  const samples = [
    sample({ filename: 'a.jpg', step: 1500, primary: mean(0.5, 0.4) }),
    sample({ filename: 'b.jpg', step: 1500, primary: mean(0.7, 0.2) }),
    sample({ filename: 'c.jpg', step: 1500, status: 'ambiguous', primary: null }),
    sample({ filename: 'd.jpg', step: 1500, status: 'failed', primary: null, warnings: ['no face detected'] }),
    sample({ filename: 'e.jpg', step: 2000, primary: mean(0.55, 0.35) }),
    sample({ filename: 'unparsed.jpg' }), // step null -> excluded from grouping
  ];
  const summaries = groupSummariesByStep(samples, 'training');
  assert.deepEqual(
    summaries.map(s => s.step),
    [1500, 2000],
  );
  const step1500 = summaries[0];
  assert.equal(step1500.evaluated, 2);
  assert.equal(step1500.ambiguous, 1);
  assert.equal(step1500.failed, 1);
  assert.equal(step1500.medianArcface, 0.6);
  assert.ok(Math.abs((step1500.medianSface ?? 0) - 0.3) < 1e-9);
  assert.equal(step1500.lowest[0].filename, 'a.jpg'); // lowest arcface first
  assert.deepEqual(step1500.lowest.map(s => s.filename), ['a.jpg', 'b.jpg']);
  assert.deepEqual(step1500.highest.map(s => s.filename), ['b.jpg', 'a.jpg']); // highest arcface first
  assert.equal(summaries[1].medianArcface, 0.55);
});

test('highest/lowest rank by SFace in SFace-only mode, never fabricate scores', () => {
  // no ArcFace scores anywhere -> ranking falls back to SFace and the lists
  // are ordered by it, not by input order with missing scores shown as 0.000
  const samples = [
    sample({ filename: 'a.jpg', step: 1500, primary: mean(null, 0.1) }),
    sample({ filename: 'b.jpg', step: 1500, primary: mean(null, 0.9) }),
    sample({ filename: 'c.jpg', step: 1500, primary: mean(null, 0.5) }),
  ];
  const [summary] = groupSummariesByStep(samples, 'training');
  assert.equal(summary.rankModel, 'sface');
  assert.deepEqual(summary.highest.map(s => s.filename), ['b.jpg', 'c.jpg', 'a.jpg']);
  assert.deepEqual(summary.lowest.map(s => s.filename), ['a.jpg', 'c.jpg', 'b.jpg']);
});

test('highest/lowest exclude samples missing the ranking model score', () => {
  const samples = [
    sample({ filename: 'scored.jpg', step: 1500, primary: mean(0.7, null) }),
    sample({ filename: 'unscored1.jpg', step: 1500, primary: mean(null, 0.4) }),
    sample({ filename: 'unscored2.jpg', step: 1500, primary: mean(null, 0.8) }),
  ];
  const [summary] = groupSummariesByStep(samples, 'training');
  assert.equal(summary.rankModel, 'arcface');
  assert.deepEqual(summary.highest.map(s => s.filename), ['scored.jpg']);
  assert.deepEqual(summary.lowest.map(s => s.filename), ['scored.jpg']);
  // nothing scored at all -> no ranking, empty lists
  const [none] = groupSummariesByStep(
    [sample({ filename: 'x.jpg', step: 100, primary: { arcface: {}, sface: {} } })],
    'training',
  );
  assert.equal(none.rankModel, null);
  assert.deepEqual(none.highest, []);
  assert.deepEqual(none.lowest, []);
});

test('overlay info formats the compact score line', () => {
  const info = overlayInfo(sample({ filename: 'a.jpg', primary: mean(0.42, 0.381) }), 'training');
  assert.equal(info?.label, 'A 0.420 · S 0.381');
  assert.equal(info?.status, 'evaluated');
  const missingSface = overlayInfo(sample({ filename: 'a.jpg', primary: mean(0.42, null) }), 'training');
  assert.equal(missingSface?.label, 'A 0.420');
  const ambiguous = overlayInfo(sample({ filename: 'a.jpg', status: 'ambiguous', primary: null }), 'training');
  assert.equal(ambiguous?.label, 'ambiguous');
  assert.equal(overlayInfo(undefined, 'training'), null);
});

test('rank disagreement flags only large rank swings', () => {
  // 9 evaluated samples; threshold = rank diff > 3
  const names = ['s0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8'];
  // arcface ranks: s0 best..s8 worst; sface nearly agrees except s4 drops to last (rank 8) -> diff 4 > 3
  const arcfaceOrder = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1];
  const sfaceOrder = [0.9, 0.8, 0.7, 0.6, 0.05, 0.4, 0.3, 0.2, 0.15];
  const samples = names.map((name, i) =>
    sample({ filename: `${name}.jpg`, primary: mean(arcfaceOrder[i], sfaceOrder[i]) }),
  );
  const flags = rankDisagreements(samples, 'training');
  assert.equal(flags.get('s4.jpg'), true);
  assert.equal(flags.get('s0.jpg'), false);
  assert.equal(flags.get('s8.jpg'), false); // 8 -> 7, diff 1
  // small sets never flag
  const small = [sample({ filename: 'a.jpg', primary: mean(0.9, 0.1) }), sample({ filename: 'b.jpg', primary: mean(0.1, 0.9) })];
  const none = rankDisagreements(small, 'training');
  assert.equal(none.get('a.jpg'), false);
});

test('sortPathsByScore orders worst-first and pushes unscored last', () => {
  const index = buildIdentityIndex({
    version: 1,
    job: 'j',
    completed: true,
    created_at: '',
    updated_at: '',
    sources: ['training'],
    refsets: {},
    samples: [
      sample({ filename: 'high.jpg', primary: mean(0.9, 0.9) }),
      sample({ filename: 'low.jpg', primary: mean(0.1, 0.1) }),
      sample({ filename: 'mid.jpg', primary: mean(0.5, 0.5) }),
      sample({ filename: 'unscored.jpg', status: 'failed', primary: null }),
      sample({ filename: 'amb.jpg', status: 'ambiguous', primary: null }),
    ],
  } as never);
  const sorted = sortPathsByScore(
    ['/x/unscored.jpg', '/x/high.jpg', '/x/amb.jpg', '/x/low.jpg', '/x/mid.jpg'],
    index,
    'arcface',
    'training',
    'worst-first',
  );
  assert.deepEqual(sorted.map(identityFilenameKey), ['low.jpg', 'mid.jpg', 'high.jpg', 'amb.jpg', 'unscored.jpg']);
  const bestFirst = sortPathsByScore(['/x/low.jpg', '/x/mid.jpg', '/x/high.jpg'], index, 'arcface', 'training', 'best-first');
  assert.deepEqual(bestFirst.map(identityFilenameKey), ['high.jpg', 'mid.jpg', 'low.jpg']);
  // missing model scores sort like unscored
  const sfaceOnlyMissing = buildIdentityIndex({
    version: 1,
    job: 'j',
    completed: true,
    created_at: '',
    updated_at: '',
    sources: ['training'],
    refsets: {},
    samples: [sample({ filename: 'a.jpg', primary: mean(0.9, null) }), sample({ filename: 'b.jpg', primary: mean(0.2, null) })],
  } as never);
  const sfaceSorted = sortPathsByScore(['/x/a.jpg', '/x/b.jpg'], sfaceOnlyMissing, 'sface', 'training');
  assert.equal(sampleScore(sfaceOnlyMissing.get('a.jpg'), 'sface', 'training'), null);
  assert.deepEqual(sfaceSorted.map(identityFilenameKey), ['a.jpg', 'b.jpg']); // both unscored: stable by name
});

test('selection merge round-trips exclusions and face picks', () => {
  const empty: IdentitySelection = { version: 1, training_refs: {}, external_refs: {} };
  const withTraining = mergeSelection(empty, {
    trainingRefs: { 'C:\\Data\\img1.jpg': { excluded: true }, '/data/img2.jpg': { face_index: 1 } },
  });
  assert.equal(withTraining.training_refs['C:\\Data\\img1.jpg'].excluded, true);
  const withExternal = mergeSelection(withTraining, {
    externalRefs: { abc: { face_index: 0 }, def: { excluded: true, face_index: 3 } },
    trainingRefs: { 'C:\\Data\\img1.jpg': { excluded: false, face_index: null } },
  });
  // update overrides, other keys survive
  assert.equal(withExternal.training_refs['C:\\Data\\img1.jpg'].excluded, false);
  assert.equal(withExternal.training_refs['C:\\Data\\img1.jpg'].face_index, null);
  assert.equal(withExternal.training_refs['/data/img2.jpg'].face_index, 1);
  assert.equal(withExternal.external_refs.def.face_index, 3);
  // invalid entries are dropped, not stored
  const sanitized = mergeSelection(withExternal, { trainingRefs: { bad: { face_index: 99 } as never, worse: 'nope' as never } });
  assert.equal('bad' in sanitized.training_refs, false);
  assert.equal('worse' in sanitized.training_refs, false);
});

test('selection keys round-trip between panel writes and the reader', () => {
  // the panel writes under normalizeRefPath(path); selectionForImage must
  // read the same entry back from the raw path (same canonical form the
  // worker's canonical_ref_key uses)
  const rawPath = 'E:\\Data\\MixedCase\\Img.JPG';
  const selection = mergeSelection({ version: 1, training_refs: {}, external_refs: {} }, {
    trainingRefs: { [normalizeRefPath(rawPath)]: { face_index: 1, excluded: false } },
  });
  const read = selectionForImage(selection, { path: rawPath, name: 'Img.JPG' }, 'training');
  assert.equal(read.face_index, 1);
  assert.equal(read.excluded, false);
  // external refs key on the sha filename, no path normalization involved
  const withExternal = mergeSelection(selection, { externalRefs: { abc: { excluded: true } } });
  assert.equal(selectionForImage(withExternal, { path: '/x/abc.jpg', name: 'abc' }, 'external').excluded, true);
});

test('ref path normalization matches the worker keys', () => {
  assert.equal(normalizeRefPath('C:\\DATA\\Sub\\Img.JPG'), 'c:/data/sub/img.jpg');
  assert.equal(normalizeRefPath('/data//sub/'), '/data/sub');
  assert.equal(normalizeRefPath('/data/sub'), '/data/sub');
});
