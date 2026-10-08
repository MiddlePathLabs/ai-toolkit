import type { GradientNoiseConfig, WeightNoiseConfig } from '@/types';

type NoisingTrainConfig = {
  weight_noise?: Partial<WeightNoiseConfig>;
  gradient_noise?: Partial<GradientNoiseConfig>;
};

// Mirrors WeightNoiseConfig / GradientNoiseConfig defaults in toolkit/config_modules.py.
export const defaultWeightNoiseConfig: WeightNoiseConfig = {
  enabled: false,
  mode: 'relative',
  sigma: 0.0125,
  bound_norm: false,
  log_every: 50,
};

export const defaultGradientNoiseConfig: GradientNoiseConfig = {
  enabled: false,
  mode: 'neelakantan',
  sigma: 0.001,
  eta: 0.01,
  gamma: 0.55,
  log_every: 50,
};

export const migrateNoisingConfig = <T extends NoisingTrainConfig>(train: T): T => {
  train.weight_noise = {
    ...defaultWeightNoiseConfig,
    ...(train.weight_noise ?? {}),
  };
  train.gradient_noise = {
    ...defaultGradientNoiseConfig,
    ...(train.gradient_noise ?? {}),
  };
  return train;
};
