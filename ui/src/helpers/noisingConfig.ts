import type { GradientNoiseConfig, WeightNoiseConfig } from '@/types';

type NoisingTrainConfig = {
  weight_noise?: Partial<WeightNoiseConfig>;
  gradient_noise?: Partial<GradientNoiseConfig>;
};

export const migrateNoisingConfig = <T extends NoisingTrainConfig>(train: T): T => {
  train.weight_noise = {
    enabled: false,
    mode: 'relative',
    sigma: 0.00125,
    bound_norm: false,
    log_every: 50,
    ...(train.weight_noise ?? {}),
  };
  train.gradient_noise = {
    enabled: false,
    mode: 'neelakantan',
    sigma: 0.001,
    eta: 0.01,
    gamma: 0.55,
    log_every: 50,
    ...(train.gradient_noise ?? {}),
  };
  return train;
};
