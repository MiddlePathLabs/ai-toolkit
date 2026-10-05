import type { JobConfig } from '@/types';

// The trainer builds its save folder from each process's own `name`
// (BaseTrainProcess.save_root), while checkpoint filenames and the UI's job
// folder use the top-level config.name. A real process name carried over by a
// clone/edit/import keeps writing checkpoints, samples and training_state.pt
// into the source job's folder even after the job is renamed. The '[name]' tag
// is replaced with config.name when the trainer loads the config
// (toolkit/config.py), so storing the tag keeps every location pointing at
// output/<config.name>.
export const resetProcessNameTags = (jobConfig: JobConfig): JobConfig => {
  const processes = jobConfig?.config?.process;
  if (!Array.isArray(processes)) return jobConfig;
  for (const process of processes) {
    if (process && typeof process === 'object' && 'name' in process) {
      process.name = '[name]';
    }
  }
  return jobConfig;
};
