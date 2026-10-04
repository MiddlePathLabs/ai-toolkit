import assert from 'node:assert/strict';
import test from 'node:test';
import {
  AdmissionUnavailableError,
  parseRawYaml,
  shouldRunTrainingAdmission,
  validateStoredConfigBeforeMutation,
} from '../cron/admission';
import type { JobConfig } from '../src/types';
import {
  formatAdmissionError,
  getIncompatibleSettings,
  removeIncompatibleSettings,
  type AdmissionResult,
} from '../src/utils/admission';

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
