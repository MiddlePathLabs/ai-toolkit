import assert from 'node:assert/strict';
import test from 'node:test';
import type { SampleConfig, SampleItem } from '../src/types';
import type { SampleRow } from '../src/utils/sampleImages';
import {
  getComparisonColumns,
  getComparisonNotice,
  getComparisonRowLabels,
  getComparisonWeightLabel,
  getDefaultComparisonColumns,
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
