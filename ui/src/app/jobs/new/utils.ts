import { GroupedSelectOption, JobConfig, SelectOption } from '@/types';
import { ModelArch } from './options';
import { objectCopy } from '@/utils/basic';
import { setNestedValue } from '@/utils/hooks';
import { requestAdmission, type AdmissionResult } from '@/utils/admission';

const expandDatasetDefaults = (
  defaults: { [key: string]: any },
  numDatasets: number,
): { [key: string]: any } => {
  // expands the defaults for datasets[x] to datasets[0], datasets[1], etc.
  const expandedDefaults: { [key: string]: any } = { ...defaults };
  for (const key in defaults) {
    if (key.includes('datasets[x].')) {
      for (let i = 0; i < numDatasets; i++) {
        const datasetKey = key.replace('datasets[x].', `datasets[${i}].`);
        const v = defaults[key];
        expandedDefaults[datasetKey] = Array.isArray(v) ? [...v] : objectCopy(v);
      }
      delete expandedDefaults[key];
    }
  }
  return expandedDefaults;
};

export const buildModelArchChange = (
  modelArchs: ModelArch[],
  currentArchName: string,
  newArchName: string,
  jobConfig: JobConfig,
): JobConfig | null => {
  const currentArch = modelArchs.find(a => a.name === currentArchName);
  const newArch = modelArchs.find(model => model.name === newArchName);
  if (!currentArch || !newArch || currentArch.name === newArchName) return null;

  const numDatasets = jobConfig.config.process[0].datasets.length;
  const newDefaults = expandDatasetDefaults(newArch.defaults || {}, numDatasets);
  let candidate = objectCopy(jobConfig);
  const setValue = (value: unknown, key: string) => {
    candidate = setNestedValue(candidate, value, key);
  };

  // Preserve every value from the previous model. Presentation cards may hide
  // unsupported sections, but only the explicit conflict action may remove them.
  setValue(newArchName, 'config.process[0].model.arch');
  for (let index = 0; index < numDatasets; index += 1) {
    setValue(newArch.controls ?? [], `config.process[0].datasets[${index}].controls`);
  }
  for (const key in newDefaults) {
    const value = newDefaults[key][0];
    if (value !== undefined) setValue(value, key);
  }
  return candidate;
};

export const handleModelArchChange = async (
  modelArchs: ModelArch[],
  currentArchName: string,
  newArchName: string,
  jobConfig: JobConfig,
  setJobConfig: (value: unknown, key?: string) => void,
  onValidated?: (result: AdmissionResult, candidate: JobConfig) => void,
): Promise<AdmissionResult | null> => {
  const candidate = buildModelArchChange(modelArchs, currentArchName, newArchName, jobConfig);
  if (!candidate) return null;
  const result = await requestAdmission(candidate);
  onValidated?.(result, candidate);
  setJobConfig(candidate);
  return result;
};
