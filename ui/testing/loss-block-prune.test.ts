const assert = require('node:assert/strict');
const test = require('node:test');
const Module = require('module');
const path = require('path');

// sucrase-node transpiles TS but does not resolve tsconfig "paths" aliases.
// Map "@/..." to ./src/... so jobConfig's runtime imports resolve.
const srcDir = path.resolve(__dirname, '..', 'src');
const origResolveFilename = Module._resolveFilename;
Module._resolveFilename = function (request: string, parent: any, ...rest: any[]) {
  if (typeof request === 'string' && request.startsWith('@/')) {
    request = path.join(srcDir, request.slice(2));
  }
  return origResolveFilename.call(this, request, parent, ...rest);
};

const {
  migrateJobConfig,
  pruneUntouchedLossBlocks,
  defaultJobConfig,
  defaultDepthConsistencyConfig,
} = require('../src/app/jobs/new/jobConfig');
const { objectCopy } = require('../src/utils/basic');

const LOSS_BLOCK_KEYS = [
  'depth_consistency',
  'normal_id',
  'body_proportion',
  'face_id',
  'subject_mask',
  'body_shape',
  'vae_anchor',
];

// What the form holds after load: migration has merged complete disabled
// objects into every loss block.
const formState = (): any => migrateJobConfig(objectCopy(defaultJobConfig));

test('untouched loss blocks are stripped from the outbound config', () => {
  const pruned = pruneUntouchedLossBlocks(formState());
  const proc = pruned.config.process[0];
  for (const key of LOSS_BLOCK_KEYS) {
    assert.equal(key in proc, false, `${key} should be pruned`);
  }
  // everything the user did configure stays
  assert.ok(proc.train && proc.model && proc.sample && proc.network && proc.save);
});

test('enabled blocks survive', () => {
  const job = formState();
  job.config.process[0].depth_consistency.loss_weight = 0.001;
  job.config.process[0].subject_mask.enabled = true;
  const pruned = pruneUntouchedLossBlocks(job);
  assert.equal(pruned.config.process[0].depth_consistency.loss_weight, 0.001);
  assert.equal(pruned.config.process[0].subject_mask.enabled, true);
  assert.equal('normal_id' in pruned.config.process[0], false);
});

test('customized-but-disabled blocks survive so staged settings are not lost', () => {
  const job = formState();
  job.config.process[0].depth_consistency.preview_every = 7;
  const pruned = pruneUntouchedLossBlocks(job);
  assert.equal(pruned.config.process[0].depth_consistency.preview_every, 7);
});

test('prune does not mutate the live form state', () => {
  const job = formState();
  const before = JSON.stringify(job.config.process[0].depth_consistency);
  pruneUntouchedLossBlocks(job);
  assert.equal(JSON.stringify(job.config.process[0].depth_consistency), before);
});

test('property key order does not affect pruning', () => {
  const job = formState();
  const dc = job.config.process[0].depth_consistency;
  job.config.process[0].depth_consistency = Object.fromEntries(Object.keys(dc).reverse().map(k => [k, dc[k]]));
  const pruned = pruneUntouchedLossBlocks(job);
  assert.equal('depth_consistency' in pruned.config.process[0], false);
});

test('round trip: a pruned config inflates back into full form state', () => {
  const pruned = pruneUntouchedLossBlocks(formState());
  assert.equal('depth_consistency' in pruned.config.process[0], false);
  const reloaded = migrateJobConfig(objectCopy(pruned));
  assert.deepEqual(reloaded.config.process[0].depth_consistency, defaultDepthConsistencyConfig);
});

// Make this file a module so its top-level `require`-bound consts stay file-scoped
// and do not collide with sibling test scripts under `tsc -p tsconfig.json`.
export {};
