import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import test from 'node:test';
import YAML from 'yaml';
import {
  AdmissionUnavailableError,
  parseRawYaml,
  resolvePythonPath,
  runAdmission,
  shouldRunTrainingAdmission,
  validateStoredConfigBeforeMutation,
} from '../cron/admission';
import startJob, { type StartJobStore } from '../cron/actions/startJob';
import { NextRequest } from 'next/server';
import { GET as queueStart } from '../src/app/api/queue/[queueID]/start/route';
import prisma from '../src/server/prisma';
import type { JobConfig } from '../src/types';
import {
  formatAdmissionError,
  applyAdmissionRemediation,
  createAdmissionValidator,
  getAdmissionCandidateKey,
  getAdmissionRemediations,
  getCurrentAdmissionValidation,
  type AdmissionResult,
  type AdmissionValidation,
} from '../src/utils/admission';

const toolkitRoot = path.resolve(__dirname, '..', '..');
const sharedAdmissionFixtures = path.join(toolkitRoot, 'testing', 'fixtures', 'admission');
const uiAdmissionFixtures = path.join(__dirname, 'fixtures', 'admission');

type QueueFixtureJob = { id: string; name: string; job_config: string; status: string };
interface QueueFixtureStore {
  job: {
    findMany: (args: unknown) => Promise<Array<Omit<QueueFixtureJob, 'status'>>>;
  };
  queue: {
    findUnique: (args: unknown) => Promise<{ id: number } | null>;
    create: (args: unknown) => Promise<unknown>;
    update: (args: unknown) => Promise<unknown>;
  };
}

const queueStore = prisma as unknown as QueueFixtureStore;
const queueRequest = () => new NextRequest('http://localhost/api/queue/0/start');

const readFixture = (fixturePath: string): unknown =>
  YAML.parse(fs.readFileSync(fixturePath, 'utf8'));

test('Python resolver honors override, embedded, virtualenv, and system precedence', () => {
  const root = '/toolkit';
  const base = {
    toolkitRoot: root,
    platform: 'linux' as const,
    env: { NODE_ENV: 'test' as const },
  };
  assert.equal(
    resolvePythonPath({
      ...base,
      env: { ...base.env, AI_TOOLKIT_PYTHON: '/custom/python' },
      exists: () => false,
    }),
    '/custom/python',
  );
  assert.equal(
    resolvePythonPath({
      ...base,
      exists: candidate => candidate === path.join(root, '..', 'python_embeded', 'python'),
    }),
    path.join(root, '..', 'python_embeded', 'python'),
  );
  assert.equal(
    resolvePythonPath({
      ...base,
      exists: candidate => candidate === path.join(root, '.venv', 'bin', 'python'),
    }),
    path.join(root, '.venv', 'bin', 'python'),
  );
  assert.equal(
    resolvePythonPath({
      ...base,
      exists: candidate => candidate === path.join(root, 'venv', 'bin', 'python'),
    }),
    path.join(root, 'venv', 'bin', 'python'),
  );
  assert.equal(resolvePythonPath({ ...base, exists: () => false }), 'python3');
});

test('UI admission runs canonical Python against shared accepted and rejected fixtures', async () => {
  // Accepted fixtures may carry non-blocking warnings; pin the exact set (same
  // expectation as testing/test_admission.py). The H3 recipe uses automagic3,
  // whose fused backward never applies the default max_grad_norm.
  const acceptedFixtures: [string, string[]][] = [
    ['accepted_h3_character.yaml', ['admission.fused_backward_clip_noop']],
    ['accepted_krea_lora.yaml', []],
  ];
  for (const [fixture, expectedWarnings] of acceptedFixtures) {
    const result = await runAdmission(readFixture(path.join(sharedAdmissionFixtures, fixture)), 'ui-fixture');
    assert.equal(result.valid, true, fixture);
    assert.deepEqual(
      result.diagnostics.map(diagnostic => [diagnostic.rule_id, diagnostic.severity]),
      expectedWarnings.map(ruleId => [ruleId, 'warning']),
      fixture,
    );
  }

  const rejected = await runAdmission(
    readFixture(path.join(uiAdmissionFixtures, 'rejected_h3_mean_flow.yaml')),
    'ui-fixture',
  );
  assert.equal(rejected.valid, false);
  assert.ok(rejected.diagnostics.some(diagnostic => diagnostic.rule_id === 'admission.mean_flow'));
});

test('invalid raw YAML is distinguishable from the last valid parent config', () => {
  const parsed = parseRawYaml('config:\n  process: [\n');
  assert.equal(parsed.valid, false);
});

test('named repairs preserve stored values until clicked and change only their own setting', () => {
  const config = {
    config: {
      process: [
        {
          model: { arch: 'minimax_h3_vsa', assistant_lora_path: 'adapter.safetensors' },
          train: { loss_type: 'mean_flow' },
          datasets: [{ folder_path: '/data/video', do_i2v: true, control_path: '/data/controls' }],
        },
      ],
    },
  } as unknown as JobConfig;
  const result: AdmissionResult = {
    valid: false,
    diagnostics: [
      {
        rule_id: 'admission.mean_flow',
        severity: 'error',
        fields: ['config.process[0].train.loss_type', 'config.process[0].model.arch'],
        reason: 'unsupported',
        remedy: 'choose mse',
      },
      {
        rule_id: 'admission.h3_fast_conditioning',
        severity: 'error',
        fields: ['config.process[0].datasets[0].do_i2v', 'config.process[0].datasets[0].control_path', 'config.process[0].model.arch'],
        reason: 'unsupported',
        remedy: 'disable i2v',
      },
    ],
    deferred: [],
  };

  const process = config.config.process[0];
  const visible = getAdmissionRemediations(config, result);
  assert.deepEqual(visible.map(item => item.path), [
    'config.process[0].train.loss_type',
    'config.process[0].datasets[0].do_i2v',
  ]);
  assert.equal(process.datasets[0].do_i2v, true);
  assert.equal(process.train.loss_type as string, 'mean_flow');

  const mse = applyAdmissionRemediation(config, result, visible[0].id);
  assert.equal(mse.config.process[0].train.loss_type, 'mse');
  assert.deepEqual(mse.config.process[0].datasets, config.config.process[0].datasets);
  assert.deepEqual(mse.config.process[0].model, config.config.process[0].model);
  assert.equal(process.train.loss_type as string, 'mean_flow');

  const noConditioning = applyAdmissionRemediation(mse, result, visible[1].id);
  assert.equal(noConditioning.config.process[0].datasets[0].do_i2v, false);
  assert.equal(noConditioning.config.process[0].datasets[0].folder_path, '/data/video');
  assert.equal(noConditioning.config.process[0].datasets[0].control_path, '/data/controls');
});

test('a named MSE repair makes a rejected imported fixture pass canonical admission', async () => {
  const config = readFixture(path.join(uiAdmissionFixtures, 'rejected_h3_mean_flow.yaml')) as JobConfig;
  const rejected = await runAdmission(config, 'ui-repair-test');
  const action = getAdmissionRemediations(config, rejected).find(item => item.rule_id === 'admission.mean_flow');
  assert.ok(action);
  const repaired = applyAdmissionRemediation(config, rejected, action.id);
  assert.deepEqual(repaired.config.process[0].model, config.config.process[0].model);
  assert.deepEqual(repaired.config.process[0].datasets, config.config.process[0].datasets);
  const accepted = await runAdmission(repaired, 'ui-repair-test');
  assert.equal(accepted.valid, true);
});

test('standalone voice bucket repair preserves the contextual dataset path and passes static admission', async t => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'aitk-admission-repair-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const voicePath = path.join(root, 'voice.wav');
  fs.writeFileSync(voicePath, '');
  const config = {
    config: {
      process: [{
        model: { arch: 'minimax_h3', name_or_path: 'Comfy-Org/MiniMax-H3' },
        train: { loss_type: 'mse', noise_scheduler: 'flowmatch' },
        datasets: [{ dataset_path: voicePath, folder_path: root, do_audio: true, buckets: false }],
      }],
    },
  } as unknown as JobConfig;
  const rejected = await runAdmission(config, 'ui-voice-repair-test');
  assert.equal(rejected.valid, false);
  const actions = getAdmissionRemediations(config, rejected);
  assert.deepEqual(actions.map(action => action.path), ['config.process[0].datasets[0].buckets']);
  assert.equal(actions[0].label, 'Enable buckets for this dataset');
  const repaired = applyAdmissionRemediation(config, rejected, actions[0].id);
  assert.deepEqual(repaired.config.process[0].datasets, [{
    dataset_path: voicePath, folder_path: root, do_audio: true, buckets: true,
  }]);
  assert.deepEqual(repaired.config.process[0].model, config.config.process[0].model);
  assert.equal(config.config.process[0].datasets[0].buckets, false);
  const accepted = await runAdmission(repaired, 'ui-voice-repair-test');
  assert.equal(accepted.valid, true);
  assert.ok(accepted.deferred.some(item => item.rule_id === 'admission.h3_audio_duration_runtime'));
});

test('only a known scalar rule offers removal; contextual paths and structured values stay intact', () => {
  const config = {
    config: {
      process: [{
        model: { arch: 'minimax_h3', name_or_path: '/models/h3', assistant_lora_path: '/models/adapter' },
        train: { gradient_accumulation_steps: -1, gradient_accumulation: 2 },
        datasets: [{ folder_path: '/data/voice', buckets: false }],
      }],
    },
  } as unknown as JobConfig;
  const result: AdmissionResult = {
    valid: false,
    diagnostics: [{
      rule_id: 'admission.legacy_accumulation',
      severity: 'error',
      fields: [
        'config.process[0].train.gradient_accumulation_steps',
        'config.process[0].model.name_or_path',
        'config.process[0].model.assistant_lora_path',
        'config.process[0].datasets[0].folder_path',
        'config.process[0].datasets[0]',
        'config.process[0].train',
      ],
      reason: 'unsupported legacy setting',
      remedy: 'use modern accumulation',
    }],
    deferred: [],
  };
  const actions = getAdmissionRemediations(config, result);
  assert.deepEqual(actions.map(action => action.path), ['config.process[0].train.gradient_accumulation_steps']);
  const repaired = applyAdmissionRemediation(config, result, actions[0].id);
  assert.deepEqual(repaired.config.process[0].train, { gradient_accumulation: 2 });
  assert.deepEqual(repaired.config.process[0].model, config.config.process[0].model);
  assert.deepEqual(repaired.config.process[0].datasets, config.config.process[0].datasets);
  assert.equal(applyAdmissionRemediation(config, result, 'admission.legacy_accumulation:config.process[0].datasets[0]'), config);
  const structured = JSON.parse(JSON.stringify(config));
  structured.config.process[0].train.gradient_accumulation_steps = { preserve: true };
  assert.deepEqual(getAdmissionRemediations(structured, result), []);
});

function deferredAdmission() {
  let resolve!: (result: AdmissionResult) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<AdmissionResult>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

test('Simple corrections clear stale rejection immediately and ignore late validation of the old config', async () => {
  const before = { config: { process: [{ train: { loss_type: 'mean_flow' } }] } } as unknown as JobConfig;
  const after = { config: { process: [{ train: { loss_type: 'mse' } }] } } as unknown as JobConfig;
  const rejected: AdmissionResult = { valid: false, diagnostics: [], deferred: [] };
  const accepted: AdmissionResult = { valid: true, diagnostics: [], deferred: [] };
  const oldRequest = deferredAdmission();
  const newRequest = deferredAdmission();
  const responses = [Promise.resolve(rejected), oldRequest.promise, newRequest.promise];
  const events: AdmissionValidation[] = [];
  const validator = createAdmissionValidator(state => events.push(state), () => responses.shift()!);
  const beforeKey = getAdmissionCandidateKey(before);
  validator.setCandidate(beforeKey);
  await validator.validate(before);
  assert.equal(getCurrentAdmissionValidation(events.at(-1)!, beforeKey)?.status, 'invalid');
  const lateValidation = validator.validate(before);
  const afterKey = getAdmissionCandidateKey(after);
  validator.setCandidate(afterKey);
  assert.equal(getCurrentAdmissionValidation(events.at(-1)!, afterKey), null);
  const currentValidation = validator.validate(after);
  assert.equal(getCurrentAdmissionValidation(events.at(-1)!, afterKey)?.status, 'pending');
  newRequest.resolve(accepted);
  assert.equal(await currentValidation, accepted);
  const publishedCount = events.length;
  oldRequest.resolve(rejected);
  assert.equal(await lateValidation, null);
  assert.equal(events.length, publishedCount);
  assert.equal(getCurrentAdmissionValidation(events.at(-1)!, afterKey)?.status, 'valid');
});

test('a late unavailable response cannot replace newer authoritative validation of the same candidate', async () => {
  const config = { config: { process: [] } } as unknown as JobConfig;
  const slow = deferredAdmission();
  const fast = deferredAdmission();
  const responses = [slow.promise, fast.promise];
  const events: AdmissionValidation[] = [];
  const validator = createAdmissionValidator(state => events.push(state), () => responses.shift()!);
  validator.setCandidate(getAdmissionCandidateKey(config));
  const previous = validator.validate(config);
  const authoritative = validator.validate(config);
  const accepted: AdmissionResult = { valid: true, diagnostics: [], deferred: [] };
  fast.resolve(accepted);
  assert.equal(await authoritative, accepted);
  slow.reject(new Error('previous request unavailable'));
  assert.equal(await previous, null);
  assert.equal(events.at(-1)?.status, 'valid');
  assert.equal(events.at(-1)?.unavailable, null);
});

test('raw syntax invalidation rejects in-flight results and recovery distinguishes runtime pending from unavailable', async () => {
  const config = { config: { process: [] } } as unknown as JobConfig;
  const pending = deferredAdmission();
  const accepted: AdmissionResult = { valid: true, diagnostics: [], deferred: [] };
  const runtimePending: AdmissionResult = {
    ...accepted,
    deferred: [{ rule_id: 'admission.runtime_deferred', severity: 'warning', fields: [], reason: 'loaded model needed', remedy: 'run final guard' }],
  };
  const events: AdmissionValidation[] = [];
  let calls = 0;
  const validator = createAdmissionValidator(state => events.push(state), async () => {
    calls += 1;
    if (calls === 1) return pending.promise;
    if (calls === 2) throw new Error('canonical service unavailable');
    return runtimePending;
  });
  validator.setCandidate(getAdmissionCandidateKey(config, 'config: {}'));
  const old = validator.validate(config, 'config: {}');
  validator.invalidate();
  const publishedCount = events.length;
  pending.resolve(accepted);
  assert.equal(await old, null);
  assert.equal(events.length, publishedCount);
  const recoveredRaw = 'config:\n  process: []\n';
  const recoveredKey = getAdmissionCandidateKey(config, recoveredRaw);
  validator.setCandidate(recoveredKey);
  assert.equal(getCurrentAdmissionValidation(events.at(-1)!, recoveredKey), null);
  await validator.validate(config, recoveredRaw);
  assert.equal(events.at(-1)?.status, 'unavailable');
  assert.match(events.at(-1)?.unavailable ?? '', /canonical service unavailable/);
  await validator.validate(config, recoveredRaw);
  assert.equal(events.at(-1)?.status, 'runtime_pending');
  assert.equal(events.at(-1)?.result?.valid, true);
  assert.equal(events.at(-1)?.unavailable, null);
});

test('stored training admission rejects invalid JSON before a start mutation', async () => {
  let runnerCalled = false;
  const failure = await validateStoredConfigBeforeMutation('{', {
    runner: async () => {
      runnerCalled = true;
      return { valid: true, diagnostics: [], deferred: [] };
    },
  });
  assert.equal(runnerCalled, false);
  assert.equal(failure?.status, 422);
  assert.match(String(failure?.body.error), /not valid JSON/);
});

test('stored training admission propagates canonical rejection before mutation', async () => {
  const failure = await validateStoredConfigBeforeMutation(JSON.stringify({ config: { process: [] } }), {
    source: 'api',
    context: { job_id: 'job-1', job_name: 'bad job' },
    runner: async (config, source) => {
      assert.deepEqual(config, { config: { process: [] } });
      assert.equal(source, 'api');
      return {
        valid: false,
        diagnostics: [
          {
            rule_id: 'admission.test',
            severity: 'error',
            fields: ['config.process'],
            reason: 'unsupported',
            remedy: 'fix it',
          },
        ],
        deferred: [],
      };
    },
  });
  assert.equal(failure?.status, 422);
  assert.equal(failure?.body.job_id, 'job-1');
  assert.equal((failure?.body.diagnostics as Array<{ rule_id: string }>)[0].rule_id, 'admission.test');
});

test('stored training admission fails closed when canonical admission is unavailable', async () => {
  const failure = await validateStoredConfigBeforeMutation('{}', {
    runner: async () => {
      throw new AdmissionUnavailableError('test unavailable');
    },
  });
  assert.equal(failure?.status, 503);
  assert.equal(failure?.body.code, 'admission_unavailable');
});

test('admission scope follows the stored process type, not job metadata', () => {
  assert.equal(
    shouldRunTrainingAdmission({ config: { process: [{ type: 'diffusion_trainer' }] } }),
    true,
  );
  assert.equal(
    shouldRunTrainingAdmission({ config: { process: [{ type: 'Qwen3OmniCaptioner' }] } }),
    false,
  );
  assert.equal(
    shouldRunTrainingAdmission({ config: { process: [{ type: 'InferenceEngine' }] } }),
    false,
  );
  assert.equal(shouldRunTrainingAdmission(undefined), true);
});

test('stored caption and inference configs skip training admission regardless of metadata', async () => {
  let runnerCalled = false;
  for (const type of ['Qwen3OmniCaptioner', 'InferenceEngine']) {
    const failure = await validateStoredConfigBeforeMutation(
      JSON.stringify({ config: { process: [{ type }] } }),
      {
        runner: async () => {
          runnerCalled = true;
          return { valid: true, diagnostics: [], deferred: [] };
        },
      },
    );
    assert.equal(failure, null);
  }
  assert.equal(runnerCalled, false);
});

test('launch errors render canonical diagnostic reasons and remedies', () => {
  const message = formatAdmissionError({
    response: {
      data: {
        valid: false,
        diagnostics: [
          {
            rule_id: 'admission.test',
            severity: 'error',
            fields: ['config.process[0].train.loss_type'],
            reason: 'unsupported loss',
            remedy: 'choose mse',
          },
        ],
        deferred: [],
      },
    },
  });
  assert.match(message, /unsupported loss/);
  assert.match(message, /choose mse/);
});

test('unknown rules never offer generic deletion of model identity or whole array entries', () => {
  const config = {
    config: {
      process: [
        {
          model: { arch: 'minimax_h3_vsa', assistant_lora_path: 'adapter.safetensors' },
          datasets: [{ do_i2v: true }, { do_i2v: true }],
        },
      ],
    },
  } as unknown as JobConfig;
  const result: AdmissionResult = {
    valid: false,
    diagnostics: [
      {
        rule_id: 'admission.identity',
        severity: 'error',
        fields: ['config.process[0].model.arch', 'config.process[0].datasets[0]'],
        reason: 'unsupported',
        remedy: 'change the setting',
      },
    ],
    deferred: [],
  };
  assert.deepEqual(getAdmissionRemediations(config, result), []);
  assert.equal(applyAdmissionRemediation(config, result, 'admission.identity:config.process[0].datasets[0]'), config);
});

test('queue start admits only queued jobs and leaves valid queue state unchanged until all pass', async () => {
  const queueWrites: unknown[] = [];
  let query: unknown;
  const originalFindMany = queueStore.job.findMany;
  const originalFindUnique = queueStore.queue.findUnique;
  const originalCreate = queueStore.queue.create;
  const originalUpdate = queueStore.queue.update;
  const acceptedConfig = JSON.stringify({ config: { process: [{ type: 'InferenceEngine' }] } });

  try {
    queueStore.job.findMany = async args => {
      query = args;
      const filter = args as { where: { status: string } };
      const jobs: QueueFixtureJob[] = [
        { id: 'running-legacy', name: 'running', job_config: '{', status: 'running' },
        { id: 'stopping-legacy', name: 'stopping', job_config: '{', status: 'stopping' },
        { id: 'queued-1', name: 'first', job_config: acceptedConfig, status: 'queued' },
        { id: 'queued-2', name: 'second', job_config: acceptedConfig, status: 'queued' },
      ];
      return jobs
        .filter(job => job.status === filter.where.status)
        .map(({ status: _status, ...job }) => job);
    };
    queueStore.queue.findUnique = async () => ({ id: 7 });
    queueStore.queue.create = async args => args;
    queueStore.queue.update = async args => {
      queueWrites.push(args);
      return args;
    };

    const response = await queueStart(queueRequest(), {
      params: Promise.resolve({ queueID: '0' }),
    });
    assert.equal(response.status, 200);
    const queryRecord = query as { where: { status: string } };
    assert.equal(queryRecord.where.status, 'queued');
    assert.equal(queueWrites.length, 1);
  } finally {
    queueStore.job.findMany = originalFindMany;
    queueStore.queue.findUnique = originalFindUnique;
    queueStore.queue.create = originalCreate;
    queueStore.queue.update = originalUpdate;
  }
});

test('invalid queued admission stops before any queue mutation', async () => {
  let queueLookupCount = 0;
  const originalFindMany = queueStore.job.findMany;
  const originalFindUnique = queueStore.queue.findUnique;
  const originalCreate = queueStore.queue.create;
  const originalUpdate = queueStore.queue.update;

  try {
    queueStore.job.findMany = async args => {
      const filter = args as { where: { status: string } };
      const jobs: QueueFixtureJob[] = [
        { id: 'running-legacy', name: 'running', job_config: '{', status: 'running' },
        { id: 'queued-invalid', name: 'invalid', job_config: '{', status: 'queued' },
      ];
      return jobs
        .filter(job => job.status === filter.where.status)
        .map(({ status: _status, ...job }) => job);
    };
    queueStore.queue.findUnique = async () => {
      queueLookupCount++;
      return { id: 7 };
    };
    queueStore.queue.create = async () => {
      throw new Error('queue create must not run after rejection');
    };
    queueStore.queue.update = async () => {
      throw new Error('queue update must not run after rejection');
    };

    const response = await queueStart(queueRequest(), {
      params: Promise.resolve({ queueID: '0' }),
    });
    assert.equal(response.status, 422);
    assert.equal(queueLookupCount, 0);
  } finally {
    queueStore.job.findMany = originalFindMany;
    queueStore.queue.findUnique = originalFindUnique;
    queueStore.queue.create = originalCreate;
    queueStore.queue.update = originalUpdate;
  }
});

test('unavailable launch is persisted as a visible queued job error', async () => {
  const updates: unknown[] = [];
  const fakeJob = {
    id: 'job-unavailable',
    name: 'unavailable',
    job_config: JSON.stringify({ config: { process: [{ type: 'diffusion_trainer' }] } }),
    status: 'queued',
  };
  const store = {
    job: {
      findUnique: async () => fakeJob,
      updateMany: async (args: unknown) => {
        updates.push(args);
        return { count: 1 };
      },
      update: async () => {
        throw new Error('launch mutation must not run when admission is unavailable');
      },
    },
  } as unknown as StartJobStore;

  await startJob('job-unavailable', {
    store,
    admissionRunner: async () => {
      throw new AdmissionUnavailableError('interpreter unavailable');
    },
  });

  assert.equal(updates.length, 1);
  const update = updates[0] as {
    where: { id: string; status: string };
    data: Record<string, unknown>;
  };
  assert.deepEqual(update.where, { id: 'job-unavailable', status: 'queued' });
  assert.equal(update.data.status, 'error');
  assert.equal(update.data.pid, null);
  assert.match(String(update.data.info), /Canonical admission is unavailable/);
  assert.equal('stop' in update.data, false);
  assert.equal('return_to_queue' in update.data, false);
});
