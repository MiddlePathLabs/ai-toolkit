const assert = require('node:assert/strict');
const test = require('node:test');
const Module = require('module');
const path = require('path');

// sucrase-node transpiles TS but does not resolve tsconfig "paths" aliases.
// Map "@/..." to ./src/... so runtime imports resolve.
const srcDir = path.resolve(__dirname, '..', 'src');
const origResolveFilename = Module._resolveFilename;
Module._resolveFilename = function (request: string, parent: any, ...rest: any[]) {
  if (typeof request === 'string' && request.startsWith('@/')) {
    request = path.join(srcDir, request.slice(2));
  }
  return origResolveFilename.call(this, request, parent, ...rest);
};

const { resetProcessNameTags } = require('../src/helpers/nameTag');

test('resets a real process name to the [name] tag', () => {
  const job: any = {
    config: {
      name: 'cloned_job',
      process: [{ type: 'diffusion_trainer', name: 'source_job' }],
    },
  };
  resetProcessNameTags(job);
  assert.equal(job.config.process[0].name, '[name]');
  // top-level name untouched
  assert.equal(job.config.name, 'cloned_job');
});

test('resets every process in a multi-process config', () => {
  const job: any = {
    config: {
      name: 'job',
      process: [
        { type: 'diffusion_trainer', name: 'train_a' },
        { type: 'diffusion_trainer', name: 'train_b' },
      ],
    },
  };
  resetProcessNameTags(job);
  assert.deepEqual(
    job.config.process.map((p: any) => p.name),
    ['[name]', '[name]'],
  );
});

test('idempotent for configs already storing the tag', () => {
  const job: any = {
    config: {
      name: 'job',
      process: [{ type: 'diffusion_trainer', name: '[name]' }],
    },
  };
  resetProcessNameTags(job);
  assert.equal(job.config.process[0].name, '[name]');
});

test('leaves processes without a name key and configs without a process array alone', () => {
  const noName: any = { config: { name: 'job', process: [{ type: 'diffusion_trainer' }] } };
  resetProcessNameTags(noName);
  assert.equal('name' in noName.config.process[0], false);

  const noProcess: any = { config: { name: 'job' } };
  assert.equal(resetProcessNameTags(noProcess), noProcess);

  assert.equal(resetProcessNameTags(undefined as any), undefined);
});
