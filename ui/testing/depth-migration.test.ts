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

const { migrateJobConfig, defaultDepthConsistencyConfig } = require('../src/app/jobs/new/jobConfig');

const baseProcess = () => ({
  type: 'diffusion_trainer',
  training_folder: 'output',
  device: 'cuda',
  trigger_word: null,
  performance_log_every: 10,
  network: {
    type: 'lora',
    linear: 32,
    linear_alpha: 32,
    conv: 16,
    conv_alpha: 16,
    lokr_full_rank: true,
    lokr_factor: -1,
    network_kwargs: { ignore_if_contains: [] },
  },
  save: { dtype: 'bf16', save_every: 250, max_step_saves_to_keep: 4, save_format: 'diffusers', push_to_hub: false },
  datasets: [],
  train: {
    batch_size: 1,
    steps: 10,
    gradient_accumulation: 1,
    train_unet: true,
    train_text_encoder: false,
    gradient_checkpointing: true,
    noise_scheduler: 'flowmatch',
    optimizer: 'adamw8bit',
    timestep_type: 'sigmoid',
    content_or_style: 'balanced',
    optimizer_params: { weight_decay: 1e-4 },
    unload_text_encoder: false,
    cache_text_embeddings: false,
    lr: 0.0001,
    dtype: 'bf16',
    skip_first_sample: false,
    force_first_sample: false,
    disable_sampling: false,
    diff_output_preservation: false,
    diff_output_preservation_multiplier: 1.0,
    diff_output_preservation_class: 'person',
    switch_boundary_every: 1,
    loss_type: 'mse',
  },
  logging: { log_every: 1, use_ui_logger: true },
  model: {
    name_or_path: 'krea/Krea-2-Raw',
    quantize: true,
    qtype: 'qfloat8',
    quantize_te: true,
    qtype_te: 'qfloat8',
    arch: 'krea2',
    low_vram: false,
    model_kwargs: {},
    compile: false,
  },
  sample: {
    sampler: 'euler',
    sample_every: 250,
    sample_start_step: 0,
    width: 1024,
    height: 1024,
    samples: [],
    neg: '',
    seed: 42,
    walk_seed: false,
    guidance_scale: 1,
    sample_steps: 9,
    num_frames: 1,
    fps: 24,
  },
});

const jobOf = (processOverrides: any) => ({
  job: 'extension',
  config: { name: 'x', process: [processOverrides] },
  meta: { name: '[name]', version: '1.0' },
});

test('pre-depth job gains a complete disabled depth object and keeps train.loss_split absent', () => {
  const job = jobOf(baseProcess());
  assert.equal('depth_consistency' in job.config.process[0], false);
  assert.equal('loss_split' in job.config.process[0].train, false);

  migrateJobConfig(job);

  const depth = job.config.process[0].depth_consistency;
  assert.deepEqual(depth, { ...defaultDepthConsistencyConfig });
  assert.equal(depth.loss_weight, 0);
  assert.equal(depth.model_id, 'depth-anything/Depth-Anything-V2-Small-hf');
  assert.equal(depth.input_size, 518);
  assert.equal(depth.mask_source, 'none');
  // Auto state is sacred: migration must not insert train.loss_split.
  assert.equal('loss_split' in job.config.process[0].train, false);
});

test('partial saved depth object preserves supplied values and fills every missing field', () => {
  const proc: any = baseProcess();
  proc.depth_consistency = { loss_weight: 0.05, input_size: 1024 };
  const job = jobOf(proc);

  migrateJobConfig(job);

  const d = job.config.process[0].depth_consistency;
  // supplied values preserved
  assert.equal(d.loss_weight, 0.05);
  assert.equal(d.input_size, 1024);
  // every other field filled with the safe default
  assert.equal(d.loss_min_t, 0);
  assert.equal(d.loss_max_t, 1);
  assert.equal(d.model_id, 'depth-anything/Depth-Anything-V2-Small-hf');
  assert.equal(d.pixel_blur_sigma, 0);
  assert.equal(d.ssi_weight, 1);
  assert.equal(d.grad_weight, 0.5);
  assert.equal(d.grad_scales, 4);
  assert.equal(d.mask_source, 'none');
  assert.equal(d.grad_checkpoint, true);
  assert.equal(d.preview_every, 100);
  assert.equal(d.preview_only, false);
  assert.equal(d.preview_max_keep, 500);
});

test('train.loss_split is preserved across all three serialized states and never inserted when absent', () => {
  // absent -> stays absent (Auto)
  const j1 = jobOf(baseProcess());
  migrateJobConfig(j1);
  assert.equal('loss_split' in j1.config.process[0].train, false);

  // explicit null -> stays null (off / Sum)
  const j2 = jobOf(baseProcess());
  (j2.config.process[0].train as any).loss_split = null;
  migrateJobConfig(j2);
  assert.equal('loss_split' in j2.config.process[0].train, true);
  assert.equal((j2.config.process[0].train as any).loss_split, null);

  // 'diffusion_depth' -> preserved (Alternate)
  const j3 = jobOf(baseProcess());
  (j3.config.process[0].train as any).loss_split = 'diffusion_depth';
  migrateJobConfig(j3);
  assert.equal((j3.config.process[0].train as any).loss_split, 'diffusion_depth');
});

// Make this file a module so its top-level `require`-bound consts stay file-scoped
// and do not collide with sibling test scripts under `tsc -p tsconfig.json`.
export {};
