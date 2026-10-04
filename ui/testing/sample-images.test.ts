import assert from 'node:assert/strict';
import test from 'node:test';
import type { SampleConfig, SampleItem } from '../src/types';
import {
  buildSampleMatrix,
  getAdjacentSamplePath,
  getSampleItems,
  getSampleMetadata,
  parseSampleFilename,
} from '../src/utils/sampleImages';

const config = (samples: SampleItem[], overrides: Partial<SampleConfig> = {}): SampleConfig =>
  ({
    seed: 100,
    walk_seed: false,
    sample_steps: 20,
    samples,
    ...overrides,
  }) as SampleConfig;
const filename = (step: number, index: number, raw = false, time = 1000, metadata = true) =>
  `${raw ? 'RAW_' : ''}${time}__${String(step).padStart(9, '0')}_${index}${metadata ? '_seed-4294967295_steps-30' : ''}.mp4`;
const cells = (paths: string[], settings: SampleConfig) => buildSampleMatrix(paths, settings).map(row => row.paths);

test('parses new metadata and RAW prefix on Windows and POSIX paths', () => {
  assert.deepEqual(parseSampleFilename('E:\\output\\RAW_1791073894128__000001400_03_seed-4294967295_steps-12.mp4'), {
    filename: 'RAW_1791073894128__000001400_03_seed-4294967295_steps-12.mp4',
    timestamp: 1791073894128,
    trainingStep: 1400,
    sampleIndex: 3,
    isRaw: true,
    seed: 4294967295,
    inferenceSteps: 12,
    hasMetadata: true,
  });
  assert.equal(parseSampleFilename('/output/1791073894128__000001400_0.jpg')?.hasMetadata, false);
  assert.equal(parseSampleFilename('RAW_1791073894128__000001400_0.webp')?.isRaw, true);
  assert.equal(parseSampleFilename('1000__000000001_0_seed--1_steps-20.png')?.seed, -1);
});

test('rejects partial metadata, unsafe integers, and unrelated filenames', () => {
  for (const path of [
    'photo.jpg',
    '1000__10_0_seed-1.jpg',
    '1000__10_x.png',
    '1000__10_0_steps-2.png',
    '9007199254740992__10_0.png',
    '1000__10_0_seed-9007199254740992_steps-20.png',
  ]) {
    assert.equal(parseSampleFilename(path), null, path);
  }
});

test('new files group by training step with original columns, not RAW prefix or per-image time', () => {
  const settings = config([{ prompt: 'raw', raw_weights: true }, { prompt: 'ema' }, { prompt: 'missing' }]);
  const raw = filename(100, 0, true, 1002);
  const ema = filename(100, 1, false, 1001);
  const later = filename(200, 1, false, 2000);
  assert.deepEqual(cells([later, raw, ema], settings), [
    [raw, ema, null],
    [null, later, null],
  ]);
});

test('legacy RAW and EMA pass-local indices map through their configured partitions', () => {
  const settings = config([
    { prompt: 'ema0' },
    { prompt: 'raw0', raw_weights: true },
    { prompt: 'ema1' },
    { prompt: 'raw1', raw_weights: true },
  ]);
  const ema0 = filename(1400, 0, false, 1000, false);
  const ema1 = filename(1400, 1, false, 1001, false);
  const raw0 = filename(1400, 0, true, 1002, false);
  const raw1 = filename(1400, 1, true, 1003, false);
  const rows = buildSampleMatrix([raw1, ema1, raw0, ema0], settings);
  assert.deepEqual(
    rows.map(row => row.paths),
    [[ema0, raw0, ema1, raw1]],
  );
  assert.equal(getSampleItems(settings)[rows[0].paths.indexOf(raw1)].prompt, 'raw1');
  assert.deepEqual(cells([ema0, raw1], settings), [[ema0, null, null, raw1]]);
});

test('all-normal legacy files keep full global indices even with configured raw flags', () => {
  const settings = config([{ prompt: 'raw', raw_weights: true }, { prompt: 'ema' }, { prompt: 'other' }]);
  const first = filename(10, 0, false, 1000, false);
  const last = filename(10, 2, false, 1000, false);
  assert.deepEqual(cells([last, first], settings), [[first, null, last]]);
});

test('same-step reruns sort by timestamp without compacting missing columns', () => {
  const settings = config([{ prompt: 'a' }, { prompt: 'b' }, { prompt: 'c' }]);
  const a = filename(10, 0, false, 1000);
  const c = filename(10, 2, false, 1001);
  const rerunA = filename(10, 0, false, 2000);
  const rerunB = filename(10, 1, false, 2001);
  assert.deepEqual(cells([rerunB, c, rerunA, a], settings), [
    [a, null, c],
    [rerunA, rerunB, null],
  ]);
  assert.deepEqual(cells([c], settings), [[null, null, c]]);
  assert.deepEqual(cells([], settings), []);
});

test('arrow navigation skips holes horizontally but keeps the configured column vertically', () => {
  const settings = config([{ prompt: 'a' }, { prompt: 'b' }, { prompt: 'c' }]);
  const a = filename(10, 0);
  const c = filename(10, 2);
  const b2 = filename(20, 1);
  const a3 = filename(30, 0);
  const c3 = filename(30, 2);
  const rows = buildSampleMatrix([c3, b2, c, a3, a], settings);
  assert.equal(getAdjacentSamplePath(rows, a, 'right'), c);
  assert.equal(getAdjacentSamplePath(rows, c, 'left'), a);
  assert.equal(getAdjacentSamplePath(rows, c, 'right'), null);
  assert.equal(getAdjacentSamplePath(rows, c, 'down'), c3);
  assert.equal(getAdjacentSamplePath(rows, a3, 'up'), a);
  assert.equal(getAdjacentSamplePath(rows, b2, 'up'), null);
  assert.equal(getAdjacentSamplePath(rows, 'deleted', 'down'), null);
});

test('filename metadata wins over saved config and prompt seed/step overrides', () => {
  const settings = config([{ prompt: 'raw --seed 15 --steps 4', seed: 12, sample_steps: 7, raw_weights: true }]);
  assert.deepEqual(getSampleMetadata(filename(10, 0, true), settings, 0), {
    trainingStep: 10,
    seed: 4294967295,
    inferenceSteps: 30,
    isRaw: true,
  });
});

test('legacy fallback inherits prior configured seeds and honors walk_seed and prompt overrides', () => {
  const settings = config([
    { prompt: 'a', seed: 42 },
    { prompt: 'b', sample_steps: 8 },
    { prompt: 'c --d 7 --seed 9 --s 11 --steps 13' },
    { prompt: 'd' },
  ]);
  const legacy = filename(10, 0, false, 1000, false);
  assert.equal(getSampleMetadata(legacy, settings, 1).seed, 42);
  assert.equal(getSampleMetadata(legacy, settings, 1).inferenceSteps, 8);
  assert.equal(getSampleMetadata(legacy, settings, 2).seed, 9);
  assert.equal(getSampleMetadata(legacy, settings, 2).inferenceSteps, 13);
  assert.equal(getSampleMetadata(legacy, settings, 3).seed, 42);
  assert.equal(getSampleMetadata(legacy, { ...settings, walk_seed: true }, 1).seed, 101);
});

test('random seeds are unknown, including inherited -1 and prompt overrides', () => {
  const settings = config([{ prompt: 'a', seed: -1 }, { prompt: 'b' }, { prompt: 'c --seed -1', seed: 40 }]);
  const legacy = filename(10, 0, false, 1000, false);
  assert.equal(getSampleMetadata(legacy, settings, 0).seed, null);
  assert.equal(getSampleMetadata(legacy, settings, 1).seed, null);
  assert.equal(getSampleMetadata(legacy, settings, 2).seed, null);
  assert.equal(getSampleMetadata('1000__000000010_0_seed--1_steps-20.png', settings, 0).seed, null);
  assert.equal(getSampleMetadata(legacy, null, 0).seed, null);
});

test('legacy prompts-only config supplies matrix columns and metadata fallbacks', () => {
  const settings = { seed: 10, sample_steps: 5, walk_seed: true, prompts: ['a', 'b'] } as SampleConfig;
  const second = filename(10, 1, false, 1000, false);
  assert.deepEqual(cells([second], settings), [[null, second]]);
  assert.equal(getSampleItems(settings)[1].prompt, 'b');
  assert.equal(getSampleMetadata(second, settings, 1).seed, 11);
  assert.deepEqual(cells([second], { ...settings, samples: [] }), [[null, second]]);
});

test('planned holes, tombstones and unexplained holes have distinct statuses and never acquire paths', () => {
  const settings = config([{ prompt: 'a' }, { prompt: 'b' }, { prompt: 'c' }, { prompt: 'legacy hole' }]);
  const present = filename(10, 0);
  const planned = filename(10, 1);
  const deleted = filename(10, 2);
  const rows = buildSampleMatrix([present], settings, 1, {
    plannedSamples: [present, planned, deleted],
    deletedSamples: [deleted],
  });
  assert.deepEqual(rows, [
    {
      key: '10:1000',
      paths: [present, null, null, null],
      missing: [null, 'not-generated', 'deleted', 'unavailable'],
    },
  ]);
  assert.equal(getAdjacentSamplePath(rows, present, 'right'), null);
  assert.equal(getAdjacentSamplePath(rows, planned, 'left'), null);
});

test('fully deleted rows remain at a stable key across deletion and extension changes', () => {
  const settings = config([{ prompt: 'a' }, { prompt: 'b' }]);
  const first = filename(10, 0);
  const second = filename(10, 1);
  const plannedSamples = [first, second];
  const before = buildSampleMatrix([first, second], settings, 1, { plannedSamples, deletedSamples: [] });
  const tombstones = [first.replace('.mp4', '.png'), second.replace('.mp4', '.webp')];
  const deleted = buildSampleMatrix([], settings, 1, { plannedSamples, deletedSamples: tombstones });
  assert.equal(deleted.length, 1);
  assert.equal(deleted[0].key, before[0].key);
  assert.deepEqual(deleted[0].paths, [null, null]);
  assert.deepEqual(deleted[0].missing, ['deleted', 'deleted']);
  const restored = first.replace('.mp4', '.jpg');
  const regenerated = buildSampleMatrix([restored], settings, 1, { plannedSamples, deletedSamples: tombstones });
  assert.equal(regenerated[0].key, before[0].key);
  assert.deepEqual(regenerated[0].paths, [restored, null]);
  assert.deepEqual(regenerated[0].missing, [null, 'deleted']);
});

test('evidence uses the same legacy raw and normal partition mapping as real media', () => {
  const settings = config([
    { prompt: 'ema0' },
    { prompt: 'raw0', raw_weights: true },
    { prompt: 'ema1' },
    { prompt: 'raw1', raw_weights: true },
  ]);
  const ema0 = filename(10, 0, false, 1000, false);
  const ema1 = filename(10, 1, false, 1001, false);
  const raw0 = filename(10, 0, true, 1002, false);
  const raw1 = filename(10, 1, true, 1003, false);
  const rows = buildSampleMatrix([ema0], settings, 1, {
    plannedSamples: [raw0],
    deletedSamples: [ema1, raw1],
  });
  assert.deepEqual(rows, [
    {
      key: '10:1000',
      paths: [ema0, null, null, null],
      missing: [null, 'not-generated', 'deleted', 'deleted'],
    },
  ]);
});

test('deleted raw-only legacy rows keep their configured column instead of compacting to zero', () => {
  const settings = config([{ prompt: 'ema' }, { prompt: 'raw', raw_weights: true }]);
  const raw = filename(10, 0, true, 1000, false);
  const rows = buildSampleMatrix([], settings, 1, { plannedSamples: [], deletedSamples: [raw] });
  assert.deepEqual(rows[0].paths, [null, null]);
  assert.deepEqual(rows[0].missing, ['unavailable', 'deleted']);
});

test('ghost reruns preserve whole empty rows and navigation skips them', () => {
  const settings = config([{ prompt: 'a' }, { prompt: 'b' }]);
  const before = filename(10, 0, false, 1000);
  const deleted = filename(10, 0, false, 2000);
  const planned = filename(10, 1, false, 2000);
  const after = filename(20, 0, false, 3000);
  const rows = buildSampleMatrix([after, before], settings, 1, {
    plannedSamples: [planned],
    deletedSamples: [deleted],
  });
  assert.deepEqual(
    rows.map(row => row.key),
    ['10:1000', '10:2000', '20:3000'],
  );
  assert.deepEqual(rows[1].missing, ['deleted', 'not-generated']);
  assert.deepEqual(rows[1].paths, [null, null]);
  assert.equal(getAdjacentSamplePath(rows, before, 'down'), after);
});
