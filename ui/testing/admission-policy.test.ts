import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
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
  getIncompatibleSettings,
  removeIncompatibleSettings,
  type AdmissionResult,
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
  const acceptedFixtures = ['accepted_h3_character.yaml', 'accepted_krea_lora.yaml'];
  for (const fixture of acceptedFixtures) {
    const result = await runAdmission(readFixture(path.join(sharedAdmissionFixtures, fixture)), 'ui-fixture');
    assert.equal(result.valid, true, fixture);
    assert.deepEqual(result.diagnostics, [], fixture);
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

test('incompatible values remain visible until an explicit removal action', () => {
  const config = {
    config: {
      process: [
        {
          model: { arch: 'minimax_h3_vsa', assistant_lora_path: 'adapter.safetensors' },
          train: { loss_type: 'mean_flow' },
          datasets: [{ do_i2v: true }],
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
        fields: ['config.process[0].train.loss_type'],
        reason: 'unsupported',
        remedy: 'choose mse',
      },
      {
        rule_id: 'admission.h3_fast_conditioning',
        severity: 'error',
        fields: ['config.process[0].datasets[0].do_i2v'],
        reason: 'unsupported',
        remedy: 'disable i2v',
      },
    ],
    deferred: [],
  };

  const process = config.config.process[0];
  const visible = getIncompatibleSettings(config, result);
  assert.deepEqual(visible.map(item => item.path), [
    'config.process[0].train.loss_type',
    'config.process[0].datasets[0].do_i2v',
  ]);
  assert.equal(process.datasets[0].do_i2v, true);

  const cleaned = removeIncompatibleSettings(config, result);
  assert.equal(cleaned.config.process[0].datasets[0].do_i2v, undefined);
  assert.equal(cleaned.config.process[0].train.loss_type, undefined);
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

test('conflict removal protects model identity and whole array entries', () => {
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
  assert.deepEqual(getIncompatibleSettings(config, result), []);
  assert.deepEqual(removeIncompatibleSettings(config, result), config);
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
