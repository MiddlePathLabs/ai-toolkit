import assert from 'node:assert/strict';
import test from 'node:test';
import type { SampleConfig, SampleItem } from '../src/types';
import type { SampleRow } from '../src/utils/sampleImages';
import {
  applyNearestCrossJobStep,
  createCrossJobSelection,
  findExactStepRow,
  findNearestStepRow,
  getComparisonColumns,
  getComparisonNotice,
  getComparisonRowLabels,
  getComparisonWeightLabel,
  getCrossJobStepNotice,
  getDefaultComparisonColumns,
  getDefaultTrainedColumn,
  getTrainingPace,
  getTrainingPaceNotice,
  matchComparisonColumn,
  nearestStepOffer,
  readComparisonJob,
  reconcileCrossJobSelection,
  selectCrossJobColumn,
  selectCrossJobRow,
} from '../src/components/sampleComparison';
import { createComparisonPlayback } from '../src/components/sampleComparisonPlayback';
import type { ComparisonPlaybackState, ComparisonVideo } from '../src/components/sampleComparisonPlayback';

const config = (samples: SampleItem[], overrides: Partial<SampleConfig> = {}): SampleConfig =>
  ({ seed: 10, walk_seed: false, sample_steps: 20, samples, ...overrides }) as SampleConfig;
const filename = (index: number, seed: number, raw = false, timestamp = 1000) =>
  `${raw ? 'RAW_' : ''}${timestamp}__000001400_${index}_seed-${seed}_steps-20.mp4`;
const row = (paths: (string | null)[], key = '1400:1000'): SampleRow => ({
  key,
  paths,
  missing: paths.map(path => (path ? null : 'unavailable')),
});

test('defaults to the available raw/non-raw pair with matching prompt and resolved seed', () => {
  const settings = config([
    { prompt: 'a forest --seed 10', raw_weights: true },
    { prompt: 'a forest --seed 12' },
    { prompt: 'a forest --seed 12', raw_weights: true },
    { prompt: 'a lake' },
  ]);
  const columns = getComparisonColumns(
    row([filename(0, 10, true), filename(1, 12), filename(2, 12, true), filename(3, 12)]),
    settings,
  );
  assert.deepEqual(getDefaultComparisonColumns(columns), [2, 1]);
  assert.equal(getComparisonNotice(columns[2], columns[1]), null);
});

test('an available mismatched pair wins over a matching but missing pane', () => {
  const settings = config([{ prompt: 'a forest', raw_weights: true }, { prompt: 'a lake' }, { prompt: 'a forest' }]);
  const columns = getComparisonColumns(row([filename(0, 10, true), filename(1, 20), null]), settings);
  assert.deepEqual(getDefaultComparisonColumns(columns), [0, 1]);
  assert.match(getComparisonNotice(columns[0], columns[1]) ?? '', /Prompts differ\. Seeds differ\./);
});

test('two real non-raw samples are preferred over a missing raw candidate', () => {
  const columns = getComparisonColumns(
    row([null, filename(1, 10), filename(2, 10)]),
    config([{ prompt: 'same', raw_weights: true }, { prompt: 'same' }, { prompt: 'same' }]),
  );
  assert.deepEqual(getDefaultComparisonColumns(columns), [1, 2]);
});

test('random or unresolved seeds are never described as a seed match', () => {
  const columns = getComparisonColumns(
    row([filename(0, -1, true), filename(1, -1)]),
    config([{ prompt: 'same', raw_weights: true }, { prompt: 'same' }]),
  );
  assert.match(getComparisonNotice(columns[0], columns[1]) ?? '', /Seed match cannot be verified/);
});

test('legacy seed resolution respects walking seeds when comparing duplicate prompts', () => {
  const columns = getComparisonColumns(
    row(['RAW_1000__000001400_0.mp4', '1001__000001400_0.mp4']),
    config([{ prompt: 'same', raw_weights: true }, { prompt: 'same' }], { walk_seed: true }),
  );
  assert.match(getComparisonNotice(columns[0], columns[1]) ?? '', /Seeds differ/);
});

test('missing comparison columns remain null and one-column rows never invent a second sample', () => {
  const ghosts = getComparisonColumns(
    row([null, null]),
    config([{ prompt: 'same', raw_weights: true }, { prompt: 'same' }]),
  );
  assert.deepEqual(getDefaultComparisonColumns(ghosts), [0, 1]);
  assert.deepEqual(
    ghosts.map(column => column.path),
    [null, null],
  );
  assert.match(getComparisonNotice(ghosts[0], ghosts[1]) ?? '', /not available/);
  assert.deepEqual(getDefaultComparisonColumns(getComparisonColumns(row([filename(0, 10)]), null)), [0, -1]);
});

test('non-raw labels only say EMA when training actually enables EMA', () => {
  const [column] = getComparisonColumns(row([filename(0, 10)]), config([{ prompt: 'same' }]));
  assert.equal(getComparisonWeightLabel(column, false), 'Non-raw weights');
  assert.equal(getComparisonWeightLabel(column, true), 'EMA weights');
});

test('training row labels distinguish reruns and retain the identity of fully missing runs', () => {
  const labels = getComparisonRowLabels([
    row([filename(0, 10)]),
    row([filename(0, 10, false, 2000)], '1400:2000'),
    row([null], '1400:3000'),
  ]);
  assert.match(labels[0], /Run 1.*1000/);
  assert.match(labels[1], /Run 2.*2000/);
  assert.match(labels[2], /Run 3.*3000/);
  assert.equal(new Set(labels).size, 3);
});
const atStep = (step: number, index: number, seed: number, raw = false, timestamp = step * 10) =>
  `${raw ? 'RAW_' : ''}${timestamp}__${String(step).padStart(9, '0')}_${index}_seed-${seed}_steps-20.mp4`;
const paceConfig = (
  batch: number | null,
  accumulation: number | null,
  folders: string[],
  extras: Record<string, unknown> = {},
) =>
  JSON.stringify({
    config: {
      process: [
        {
          train: {
            batch_size: batch,
            gradient_accumulation: accumulation,
            ema_config: { use_ema: true },
            ...extras,
          },
          datasets: folders.map(folder_path => ({ folder_path })),
          sample: { samples: [{ prompt: 'a forest' }, { prompt: 'a forest', raw_weights: true }] },
        },
      ],
    },
  });

test('an exact step match uses the latest run and never substitutes a nearby step', () => {
  const rows = [
    row([atStep(1000, 0, 1)], '1000:1000'),
    row([atStep(1400, 0, 1, false, 1401)], '1400:1401'),
    row([atStep(1400, 0, 1, false, 1402)], '1400:1402'),
    row([atStep(2000, 0, 1)], '2000:2000'),
  ];
  assert.equal(findExactStepRow(rows, 1400)?.key, '1400:1402');
  assert.equal(findExactStepRow(rows, 1500), null);
});

test('the nearest step prefers the closer sample, then the later step, then the latest run', () => {
  const rows = [
    row([atStep(1000, 0, 1)], '1000:1000'),
    row([atStep(2000, 0, 1, false, 2001)], '2000:2001'),
    row([atStep(2000, 0, 1, false, 2002)], '2000:2002'),
    row([atStep(3000, 0, 1)], '3000:3000'),
  ];
  assert.equal(findNearestStepRow(rows, 1600)?.key, '2000:2002');
  assert.equal(findNearestStepRow(rows, 1500)?.key, '2000:2002');
  assert.equal(findNearestStepRow([], 1500), null);
});

test('prompt matching prefers the same weight type and seed and rejects a different prompt', () => {
  const settings = config([
    { prompt: 'a forest', raw_weights: true },
    { prompt: 'a forest' },
    { prompt: 'a forest' },
    { prompt: 'a lake' },
  ]);
  const columns = getComparisonColumns(
    row([atStep(1400, 0, 1, true), atStep(1400, 1, 2), atStep(1400, 2, 1), atStep(1400, 3, 1)]),
    settings,
  );
  assert.equal(matchComparisonColumn(columns, columns[2]), 2);
  assert.equal(matchComparisonColumn(columns, columns[0]), 0);
  assert.equal(matchComparisonColumn(columns, { ...columns[1], metadata: { ...columns[1].metadata, seed: 9 } }), 1);
  assert.equal(matchComparisonColumn(columns, { ...columns[3], prompt: 'a mountain' }), null);
  const missingPrompt = getComparisonColumns(
    row([null, atStep(1400, 1, 1)]),
    config([{ prompt: 'a forest' }, { prompt: 'a lake' }]),
  );
  assert.equal(matchComparisonColumn(missingPrompt, missingPrompt[0]), 0);
  assert.equal(getDefaultTrainedColumn(columns), 1);
});

test('a cross-job pair starts on trained weights and leaves a missing step empty', () => {
  const leftSettings = config([{ prompt: 'a forest', raw_weights: true }, { prompt: 'a forest' }]);
  const rightSettings = config([{ prompt: 'a lake' }, { prompt: 'a forest' }]);
  const leftRows = [row([atStep(1400, 0, 4, true), atStep(1400, 1, 4)], '1400:1400')];
  const rightRows = [row([atStep(1000, 0, 4), atStep(1000, 1, 4)], '1000:1000')];
  const selection = createCrossJobSelection(leftRows, leftSettings, rightRows, rightSettings, '1400:1400');
  assert.deepEqual(selection, {
    left: { rowKey: '1400:1400', column: 1 },
    right: { rowKey: null, column: 1 },
  });
  assert.equal(selectCrossJobRow(selection, 'left', '1400:1400', leftRows, rightRows, true).right.rowKey, null);
});

test('syncing steps never snaps, and the nearest action names both steps', () => {
  const leftRows = [row([atStep(1400, 0, 1)], '1400:1400'), row([atStep(2000, 0, 1)], '2000:2000')];
  const rightRows = [row([atStep(1000, 0, 1)], '1000:1000'), row([atStep(2000, 0, 1)], '2000:2000')];
  const selection = {
    left: { rowKey: '1400:1400', column: 0 },
    right: { rowKey: '2000:2000', column: 0 },
  };
  const synced = selectCrossJobRow(selection, 'left', '1400:1400', leftRows, rightRows, true);
  assert.equal(synced.right.rowKey, null);
  assert.equal(selectCrossJobRow(selection, 'left', '1400:1400', leftRows, rightRows, false).right.rowKey, '2000:2000');
  const offer = nearestStepOffer(synced, leftRows, rightRows);
  assert.deepEqual(offer, { side: 'right', selectedStep: 1400, nearestStep: 1000, rowKey: '1000:1000' });
  assert.equal(applyNearestCrossJobStep(synced, leftRows, rightRows)?.right.rowKey, '1000:1000');
  assert.equal(applyNearestCrossJobStep(synced, leftRows, rightRows)?.left.rowKey, '1400:1400');
});

test('matching prompts updates the other pane and a prompt miss leaves it alone', () => {
  const leftColumns = getComparisonColumns(
    row([atStep(1400, 0, 1), atStep(1400, 1, 1)]),
    config([{ prompt: 'a forest' }, { prompt: 'a lake' }]),
  );
  const rightColumns = getComparisonColumns(
    row([atStep(1400, 0, 1), atStep(1400, 1, 1)]),
    config([{ prompt: 'a lake' }, { prompt: 'a forest' }]),
  );
  const selection = { left: { rowKey: '1400:1400', column: 0 }, right: { rowKey: '1400:1400', column: 0 } };
  assert.equal(selectCrossJobColumn(selection, 'left', 0, leftColumns, rightColumns, true).right.column, 1);
  assert.equal(selectCrossJobColumn(selection, 'left', 0, leftColumns, rightColumns, false).right.column, 0);
  const unmatched = selectCrossJobColumn(
    selection,
    'left',
    0,
    leftColumns,
    getComparisonColumns(row([atStep(1400, 0, 1)]), config([{ prompt: 'a mountain' }])),
    true,
  );
  assert.equal(unmatched.right.column, 0);
});

test('a later exact sample fills an empty synced pane and does not fill an unsynced pane', () => {
  const before = [row([atStep(1000, 0, 1)], '1000:1000')];
  const after = [...before, row([atStep(1400, 0, 1)], '1400:1400')];
  const selection = { left: { rowKey: '1400:1400', column: 0 }, right: { rowKey: null, column: 0 } };
  const leftRows = [row([atStep(1400, 0, 1)], '1400:1400')];
  assert.equal(reconcileCrossJobSelection(selection, leftRows, after, true).right.rowKey, '1400:1400');
  assert.equal(reconcileCrossJobSelection(selection, leftRows, before, false).right.rowKey, null);
});

test('a missing saved row is dropped, and sync refills only from a surviving step', () => {
  const stale = { left: { rowKey: 'gone-left', column: 2 }, right: { rowKey: 'gone-right', column: 3 } };
  const leftRows = [row([atStep(1400, 0, 1)], '1400:1400')];
  const rightRows = [row([atStep(1400, 0, 1)], '1400:9000'), row([atStep(2000, 0, 1)], '2000:2000')];
  const unsynced = reconcileCrossJobSelection(stale, leftRows, rightRows, false);
  assert.equal(unsynced.left.rowKey, null);
  assert.equal(unsynced.right.rowKey, null);
  assert.equal(unsynced.left.column, 2);
  const synced = reconcileCrossJobSelection(
    { left: { rowKey: '1400:1400', column: 1 }, right: { rowKey: 'missing', column: 4 } },
    leftRows,
    rightRows,
    true,
  );
  assert.equal(synced.right.rowKey, '1400:9000');
  assert.equal(synced.left.column, 1);
  assert.equal(selectCrossJobRow(stale, 'left', 'missing', leftRows, rightRows, true), stale);
});

test('step and training-pace notices name the mismatch and stay quiet when the runs align', () => {
  assert.equal(
    getCrossJobStepNotice(1400, null, 'alpha', 'beta'),
    `beta has no sample at step ${Number(1400).toLocaleString()}.`,
  );
  assert.equal(
    getCrossJobStepNotice(1400, 1500, 'alpha', 'beta'),
    `Training steps differ (${Number(1400).toLocaleString()} vs ${Number(1500).toLocaleString()}).`,
  );
  assert.equal(getCrossJobStepNotice(1400, 1400, 'alpha', 'beta'), null);
  const same = getTrainingPace(paceConfig(1, 1, ['E:/data/a']));
  assert.equal(getTrainingPaceNotice(same, getTrainingPace(paceConfig(1, 1, ['E:\\data\\a\\']))), null);
  assert.match(
    getTrainingPaceNotice(same, getTrainingPace(paceConfig(2, 1, ['E:/data/a']))) ?? '',
    /^Batch size differs/,
  );
  assert.match(
    getTrainingPaceNotice(same, getTrainingPace(paceConfig(1, 1, ['E:/data/a'], { gradient_accumulation_steps: 4 }))) ??
      '',
    /Gradient accumulation differs/,
  );
  assert.match(
    getTrainingPaceNotice(
      getTrainingPace(
        JSON.stringify({ config: { process: [{ train: { gradient_accumulation_steps: 4 }, datasets: [] }] } }),
      ),
      getTrainingPace(JSON.stringify({ config: { process: [{ train: { gradient_accumulation: 4 }, datasets: [] }] } })),
    ) ?? '',
    /Legacy gradient_accumulation_steps counts every batch as a step/,
  );
  assert.equal(
    getTrainingPaceNotice(same, getTrainingPace(paceConfig(1, 1, ['E:/data/a'], { gradient_accumulation_steps: 1 }))),
    null,
  );
  assert.equal(getTrainingPaceNotice(same, getTrainingPace(paceConfig(1, 1, ['e:/DATA/a']))), null);
  assert.equal(getTrainingPaceNotice(null, same), 'Training settings cannot be compared.');
  assert.equal(readComparisonJob(paceConfig(1, 1, ['E:/data/a'])).hasEma, true);
  assert.equal(readComparisonJob('{"config":{"process":[{"train":{"ema_config":{"use_ema":false}}}]}}').hasEma, false);
});

class FakeVideo extends EventTarget implements ComparisonVideo {
  duration = 8;
  readyState = 4;
  seeking = false;
  ended = false;
  error = null;
  paused = true;
  plays = 0;
  pauses = 0;
  playResult: (() => Promise<void>) | null = null;
  private time = 0;

  get currentTime() {
    return this.time;
  }
  set currentTime(value: number) {
    this.time = value;
    this.ended = false;
    this.seeking = true;
    this.dispatchEvent(new Event('seeking'));
  }

  play() {
    this.plays++;
    this.paused = false;
    return this.playResult?.() ?? Promise.resolve();
  }

  pause() {
    this.pauses++;
    this.paused = true;
  }

  advance(time: number) {
    this.time = time;
  }
  finishSeek() {
    this.seeking = false;
    this.dispatchEvent(new Event('seeked'));
  }
}

function playbackHarness() {
  const left = new FakeVideo();
  const right = new FakeVideo();
  const frames = new Map<number, () => void>();
  let frameId = 0;
  let state: ComparisonPlaybackState | null = null;
  const controller = createComparisonPlayback(
    [left, right],
    next => {
      state = next;
    },
    {
      request: callback => {
        frames.set(++frameId, callback);
        return frameId;
      },
      cancel: handle => {
        frames.delete(handle);
      },
    },
  );
  return {
    left,
    right,
    controller,
    frames,
    get state() {
      return state;
    },
    frame: () => {
      const callbacks = [...frames.values()];
      frames.clear();
      callbacks.forEach(callback => callback());
    },
  };
}

async function settlePlayback() {
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
}

test('paired playback starts both videos and pauses both on either buffering event', async t => {
  const harness = playbackHarness();
  t.after(() => harness.controller.dispose());
  harness.controller.play();
  await settlePlayback();
  assert.equal(harness.left.plays, 1);
  assert.equal(harness.right.plays, 1);
  assert.equal(harness.state?.playing, true);
  harness.right.readyState = 2;
  harness.right.dispatchEvent(new Event('waiting'));
  assert.equal(harness.left.paused, true);
  assert.equal(harness.right.paused, true);
  assert.equal(harness.state?.pending, true);
  harness.frame();
  assert.equal(harness.left.plays, 1);
  harness.right.readyState = 4;
  harness.right.dispatchEvent(new Event('canplay'));
  await settlePlayback();
  assert.equal(harness.state?.playing, true);
  assert.equal(harness.left.plays, 2);
  assert.equal(harness.right.plays, 2);
});

test('a seek pauses both until both settle, and clamps to the common shorter duration', async t => {
  const harness = playbackHarness();
  t.after(() => harness.controller.dispose());
  harness.right.duration = 6;
  harness.controller.play();
  await settlePlayback();
  harness.controller.seek(4);
  assert.equal(harness.left.currentTime, 4);
  assert.equal(harness.right.currentTime, 4);
  assert.equal(harness.left.paused, true);
  assert.equal(harness.right.paused, true);
  harness.left.finishSeek();
  assert.equal(harness.left.plays, 1);
  harness.right.finishSeek();
  await settlePlayback();
  assert.equal(harness.state?.playing, true);
  harness.controller.seek(99);
  assert.equal(harness.left.currentTime, 6);
  assert.equal(harness.right.currentTime, 6);
  assert.equal(harness.state?.duration, 6);
  harness.controller.pause();
  harness.controller.restart();
  assert.equal(harness.left.currentTime, 0);
  assert.equal(harness.right.currentTime, 0);
});

test('drift correction pauses both and aligns the lagging video before resuming', async t => {
  const harness = playbackHarness();
  t.after(() => harness.controller.dispose());
  harness.controller.play();
  await settlePlayback();
  harness.left.advance(4);
  harness.right.advance(3.6);
  harness.frame();
  assert.equal(harness.left.paused, true);
  assert.equal(harness.right.paused, true);
  assert.equal(harness.right.currentTime, 4);
  harness.right.finishSeek();
  await settlePlayback();
  assert.equal(harness.state?.playing, true);
});

test('the shorter right-hand video stops both and clamps the longer pane at its end', async t => {
  const harness = playbackHarness();
  t.after(() => harness.controller.dispose());
  harness.right.duration = 5;
  harness.controller.play();
  await settlePlayback();
  harness.left.advance(4.98);
  harness.right.advance(5);
  harness.frame();
  assert.equal(harness.left.paused, true);
  assert.equal(harness.right.paused, true);
  assert.equal(harness.left.currentTime, 5);
  assert.equal(harness.state?.playing, false);
  assert.equal(harness.state?.pending, false);
  assert.equal(harness.frames.size, 0);
});

test('pausing while buffering cancels the intent to resume when data arrives', async t => {
  const harness = playbackHarness();
  t.after(() => harness.controller.dispose());
  harness.controller.play();
  await settlePlayback();
  harness.right.dispatchEvent(new Event('waiting'));
  harness.controller.pause();
  harness.right.dispatchEvent(new Event('canplay'));
  harness.frame();
  await settlePlayback();
  assert.equal(harness.left.plays, 1);
  assert.equal(harness.right.plays, 1);
  assert.equal(harness.state?.pending, false);
});

test('one rejected play request stops both panes and reports the real failure', async t => {
  const harness = playbackHarness();
  t.after(() => harness.controller.dispose());
  harness.right.playResult = () => Promise.reject(new Error('Playback blocked'));
  harness.controller.play();
  await settlePlayback();
  assert.equal(harness.left.paused, true);
  assert.equal(harness.right.paused, true);
  assert.match(harness.state?.error ?? '', /Playback blocked/);
  assert.equal(harness.frames.size, 0);
});

test('closing during pending play cancels frames and cannot restart detached videos', async () => {
  const harness = playbackHarness();
  let resolvePlay: (() => void) | null = null;
  harness.right.playResult = () =>
    new Promise<void>(resolve => {
      resolvePlay = resolve;
    });
  harness.controller.play();
  harness.controller.dispose();
  const complete = resolvePlay as (() => void) | null;
  complete?.();
  await settlePlayback();
  harness.right.dispatchEvent(new Event('canplay'));
  assert.equal(harness.left.paused, true);
  assert.equal(harness.right.paused, true);
  assert.equal(harness.left.plays, 1);
  assert.equal(harness.frames.size, 0);
});
