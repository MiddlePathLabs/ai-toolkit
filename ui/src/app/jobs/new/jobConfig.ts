'use client';
import { isMac } from '@/helpers/basic';
import { defaultSampleConfig } from '@/helpers/defaultSamples';
import { migrateNoisingConfig } from '@/helpers/noisingConfig';
import { JobConfig, SampleConfig, DatasetConfig, SliderConfig, DepthConsistencyConfig, NormalIDConfig } from '@/types';

export const defaultDatasetConfig: DatasetConfig = {
  folder_path: '/path/to/images/folder',
  mask_path: null,
  mask_min_value: 0.1,
  default_caption: '',
  caption_ext: 'txt',
  caption_dropout_rate: 0.05,
  cache_latents_to_disk: false,
  is_reg: false,
  network_weight: 1,
  resolution: [512, 768, 1024],
  controls: [],
  shrink_video_to_frames: true,
  num_frames: 1,
  flip_x: false,
  flip_y: false,
  num_repeats: 1,
};

export const defaultSliderConfig: SliderConfig = {
  guidance_strength: 3.0,
  anchor_strength: 1.0,
  positive_prompt: 'person who is happy',
  negative_prompt: 'person who is sad',
  target_class: 'person',
  anchor_class: '',
};

export const defaultCompileOptions = {
  block_compile: true,
};

// Safe disabled defaults for the process-level depth anchor. Enable = set
// loss_weight > 0 (UI uses 0.001); disable = loss_weight 0. There is no
// `enabled` field. DA2-Small + input_size 518 is the shipped baseline.
export const defaultDepthConsistencyConfig: DepthConsistencyConfig = {
  loss_weight: 0,
  loss_min_t: 0,
  loss_max_t: 1,
  model_id: 'depth-anything/Depth-Anything-V2-Small-hf',
  input_size: 518,
  pixel_blur_sigma: 0,
  ssi_weight: 1,
  grad_weight: 0.5,
  grad_scales: 4,
  mask_source: 'none',
  grad_checkpoint: true,
  preview_every: 100,
  preview_only: false,
  preview_max_keep: 500,
};

// Safe disabled defaults for the process-level normal anchor. Enable = set
// loss_weight > 0; disable = loss_weight 0. There is no `enabled` field. The
// Sapiens 0.3B perceptor downloads lazily only when normal loss is enabled.
export const defaultNormalIDConfig: NormalIDConfig = {
  loss_weight: 0,
  loss_min_t: 0.4,
  loss_max_t: 0.8,
  model_id: 'facebook/sapiens-normal-0.3b',
  grad_checkpoint: true,
  preview_every: 100,
  preview_only: false,
  preview_max_keep: 500,
};

export const defaultJobConfig: JobConfig = {
  job: 'extension',
  config: {
    name: 'my_first_lora_v1',
    process: [
      {
        type: 'diffusion_trainer',
        training_folder: 'output',
        sqlite_db_path: './aitk_db.db',
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
          network_kwargs: {
            ignore_if_contains: [],
          },
        },
        save: {
          dtype: 'bf16',
          save_every: 250,
          max_step_saves_to_keep: 4,
          save_format: 'diffusers',
          push_to_hub: false,
        },
        datasets: [defaultDatasetConfig],
        train: {
          batch_size: 1,
          bypass_guidance_embedding: true,
          steps: 3000,
          gradient_accumulation: 1,
          train_unet: true,
          train_text_encoder: false,
          gradient_checkpointing: true,
          noise_scheduler: 'flowmatch',
          optimizer: 'adamw8bit',
          timestep_type: 'sigmoid',
          content_or_style: 'balanced',
          optimizer_params: {
            weight_decay: 1e-4,
          },
          unload_text_encoder: false,
          cache_text_embeddings: false,
          lr: 0.0001,
          ema_config: {
            use_ema: false,
            ema_decay: 0.99,
          },
          weight_noise: {
            enabled: false,
            mode: 'relative',
            sigma: 0.00125,
            bound_norm: false,
            log_every: 50,
          },
          gradient_noise: {
            enabled: false,
            mode: 'neelakantan',
            sigma: 0.001,
            eta: 0.01,
            gamma: 0.55,
            log_every: 50,
          },
          skip_first_sample: false,
          force_first_sample: false,
          disable_sampling: false,
          dtype: 'bf16',
          diff_output_preservation: false,
          diff_output_preservation_multiplier: 1.0,
          diff_output_preservation_class: 'person',
          switch_boundary_every: 1,
          loss_type: 'mse',
        },
        logging: {
          log_every: 1,
          use_ui_logger: true,
        },
        model: {
          name_or_path: 'ostris/Flex.1-alpha',
          quantize: true,
          qtype: 'qfloat8',
          quantize_te: true,
          qtype_te: 'qfloat8',
          arch: 'flex1',
          low_vram: false,
          model_kwargs: {},
          compile: false,
        },
        sample: defaultSampleConfig,
        depth_consistency: { ...defaultDepthConsistencyConfig },
        normal_id: { ...defaultNormalIDConfig },
      },
    ],
  },
  meta: {
    name: '[name]',
    version: '1.0',
  },
};

export const migrateJobConfig = (jobConfig: JobConfig): JobConfig => {
  // upgrade prompt strings to samples
  if (
    jobConfig?.config?.process &&
    jobConfig.config.process[0]?.sample &&
    Array.isArray(jobConfig.config.process[0].sample.prompts) &&
    jobConfig.config.process[0].sample.prompts.length > 0
  ) {
    let newSamples = [];
    for (const prompt of jobConfig.config.process[0].sample.prompts) {
      newSamples.push({
        prompt: prompt,
      });
    }
    jobConfig.config.process[0].sample.samples = newSamples;
    delete jobConfig.config.process[0].sample.prompts;
  }

  // upgrade job from ui_trainer to diffusion_trainer
  if (jobConfig?.config?.process && jobConfig.config.process[0]?.type === 'ui_trainer') {
    jobConfig.config.process[0].type = 'diffusion_trainer';
  }

  if ('auto_memory' in jobConfig.config.process[0].model) {
    jobConfig.config.process[0].model.layer_offloading = (jobConfig.config.process[0].model.auto_memory ||
      false) as boolean;
    delete jobConfig.config.process[0].model.auto_memory;
  }

  if (!('logging' in jobConfig.config.process[0])) {
    //@ts-ignore
    jobConfig.config.process[0].logging = {
      log_every: 1,
      use_ui_logger: true,
    };
  }

  const train = jobConfig.config.process[0].train;
  migrateNoisingConfig(train);
  // Merge a complete disabled depth object into any partial saved object.
  // Preserves saved values, fills fields added after the first depth release.
  // Do NOT migrate train.loss_split here: omission IS the Auto state, and
  // inserting null would silently turn Auto into explicit off.
  jobConfig.config.process[0].depth_consistency = {
    ...defaultDepthConsistencyConfig,
    ...(jobConfig.config.process[0].depth_consistency ?? {}),
  };
  // Merge a complete disabled normal object into any partial saved object.
  jobConfig.config.process[0].normal_id = {
    ...defaultNormalIDConfig,
    ...(jobConfig.config.process[0].normal_id ?? {}),
  };
  if (isMac()) {
    jobConfig.config.process[0].device = 'mps';
  }

  return jobConfig;
};
