import { apiClient } from './api';
import type { JobConfig } from '../types';

export type AdmissionSeverity = 'error' | 'warning';

export interface AdmissionDiagnostic {
  rule_id: string;
  severity: AdmissionSeverity;
  fields: string[];
  reason: string;
  remedy: string;
  phase?: 'static' | 'runtime';
}

export interface AdmissionResult {
  valid: boolean;
  diagnostics: AdmissionDiagnostic[];
  deferred: AdmissionDiagnostic[];
  resolved?: unknown[];
  source?: string;
}

export class AdmissionClientError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'AdmissionClientError';
  }
}

const isAdmissionResult = (value: unknown): value is AdmissionResult => {
  if (!value || typeof value !== 'object') return false;
  const result = value as Record<string, unknown>;
  return typeof result.valid === 'boolean' && Array.isArray(result.diagnostics) && Array.isArray(result.deferred);
};

export const requestAdmission = async (config: unknown, rawYaml?: string): Promise<AdmissionResult> => {
  try {
    const response = await apiClient.post<AdmissionResult>('/api/admission', {
      config,
      ...(rawYaml === undefined ? {} : { raw_yaml: rawYaml }),
    });
    if (!isAdmissionResult(response.data)) throw new AdmissionClientError('Canonical admission returned an invalid result.');
    return response.data;
  } catch (error: unknown) {
    const responseData = readResponseData(error);
    if (isAdmissionResult(responseData)) return responseData;
    const message = readErrorMessage(error);
    throw new AdmissionClientError(message || 'Canonical admission is unavailable; the configuration was not changed.');
  }
};
const readResponseData = (error: unknown): unknown => {
  if (!error || typeof error !== 'object' || !('response' in error)) return undefined;
  const response = error.response;
  if (!response || typeof response !== 'object' || !('data' in response)) return undefined;
  return response.data;
};

const readErrorMessage = (error: unknown): string | null => {
  const data = readResponseData(error);
  if (data && typeof data === 'object' && 'error' in data && typeof data.error === 'string') return data.error;
  return error instanceof Error ? error.message : null;
};

export type AdmissionValidationStatus = 'pending' | 'invalid' | 'valid' | 'runtime_pending' | 'unavailable';

export interface AdmissionValidation {
  candidateKey: string;
  status: AdmissionValidationStatus;
  result: AdmissionResult | null;
  unavailable: string | null;
}

export interface AdmissionValidator {
  setCandidate: (candidateKey: string) => void;
  invalidate: () => void;
  validate: (config: JobConfig, rawYaml?: string) => Promise<AdmissionResult | null>;
}

export const getAdmissionCandidateKey = (config: unknown, rawYaml?: string): string =>
  JSON.stringify([config, rawYaml ?? null]);

export const getCurrentAdmissionValidation = (
  validation: AdmissionValidation | null,
  candidateKey: string,
): AdmissionValidation | null => validation?.candidateKey === candidateKey ? validation : null;

export const createAdmissionValidator = (
  onChange: (validation: AdmissionValidation) => void,
  runner: typeof requestAdmission = requestAdmission,
): AdmissionValidator => {
  let candidateKey: string | undefined;
  let requestId = 0;
  const invalidate = () => { requestId += 1; };
  const setCandidate = (key: string) => {
    if (candidateKey === key) return;
    candidateKey = key;
    invalidate();
  };
  const validate = async (config: JobConfig, rawYaml?: string): Promise<AdmissionResult | null> => {
    const key = getAdmissionCandidateKey(config, rawYaml);
    if (key !== candidateKey) return null;
    const id = ++requestId;
    onChange({ candidateKey: key, status: 'pending', result: null, unavailable: null });
    try {
      const result = await runner(config, rawYaml);
      if (key !== candidateKey || id !== requestId) return null;
      const status = !result.valid ? 'invalid' : result.deferred.length > 0 ? 'runtime_pending' : 'valid';
      onChange({ candidateKey: key, status, result, unavailable: null });
      return result;
    } catch (error) {
      if (key !== candidateKey || id !== requestId) return null;
      onChange({ candidateKey: key, status: 'unavailable', result: null, unavailable: formatAdmissionError(error) });
      return null;
    }
  };
  return { setCandidate, invalidate, validate };
};

type PathPart = string | number | '*';
type SettingValue = string | number | boolean;

interface RemediationRule {
  field: RegExp;
  label: string;
  value?: SettingValue;
}

const processField = (suffix: string): RegExp =>
  new RegExp(`^config\\.process\\[\\d+\\]\\.${suffix}$`);
const datasetField = (suffix: string): RegExp =>
  processField(`datasets\\[\\d+\\]\\.${suffix}`);

const remediationRules: Record<string, RemediationRule[]> = {
  'admission.mean_flow': [
    { field: processField('train\\.loss_type'), label: 'Set loss type to MSE', value: 'mse' },
  ],
  'admission.dopsd_loss': [
    { field: processField('train\\.loss_type'), label: 'Set loss type to MSE', value: 'mse' },
  ],
  'admission.h3_standalone_audio': [
    { field: datasetField('buckets'), label: 'Enable buckets for this dataset', value: true },
    { field: datasetField('caption_dropout_rate'), label: 'Disable voice caption dropout', value: 0 },
  ],
  'admission.h3_target_fps': [
    { field: datasetField('fps'), label: 'Set target FPS to 24', value: 24 },
  ],
  'admission.h3_fast_conditioning': [
    { field: datasetField('do_i2v'), label: 'Disable image-to-video conditioning', value: false },
  ],
  'admission.legacy_accumulation': [
    { field: processField('train\\.gradient_accumulation_steps'), label: 'Remove legacy accumulation setting' },
  ],
  'admission.flow_unaugmented_target': [
    { field: processField('train\\.loss_target'), label: 'Use the noise loss target', value: 'noise' },
  ],
  'admission.krea_low_vram_auxiliary': [
    { field: processField('model\\.low_vram'), label: 'Disable low-VRAM tiled decoding', value: false },
  ],
  'admission.latent_cache_augmentation': [
    { field: datasetField('cache_latents'), label: 'Disable in-memory latent caching', value: false },
    { field: datasetField('cache_latents_to_disk'), label: 'Disable disk latent caching', value: false },
  ],
  'admission.temporary_safety_block': [
    { field: processField('network\\.(?:network_kwargs\\.)?module_dropout'), label: 'Remove unsupported module dropout' },
    { field: processField('train\\.loss_target'), label: 'Use the noise loss target', value: 'noise' },
    { field: processField('train\\.do_guidance_loss_cfg_zero'), label: 'Disable CFG-Zero guidance loss', value: false },
  ],
};

const parsePath = (path: string): PathPart[] => {
  const parts: PathPart[] = [];
  const pattern = /([^.[\]]+)|\[(\d+|x)\]/g;
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(path))) parts.push(match[1] ?? (match[2] === 'x' ? '*' : Number(match[2])));
  return parts;
};

const expandPaths = (value: unknown, parts: PathPart[], prefix: PathPart[] = []): PathPart[][] => {
  if (parts.length === 0) return [prefix];
  const [part, ...rest] = parts;
  if (part === '*') {
    if (!Array.isArray(value)) return [];
    return value.flatMap((entry, index) => expandPaths(entry, rest, [...prefix, index]));
  }
  if (!value || typeof value !== 'object') return [];
  const record = value as Record<string, unknown>;
  if (rest.length > 0 && !(String(part) in record)) return [];
  return expandPaths(record[String(part)], rest, [...prefix, part]);
};

const readPath = (value: unknown, path: PathPart[]): unknown => {
  let current = value;
  for (const part of path) {
    if (typeof part === 'number' && Array.isArray(current)) current = current[part];
    else if (typeof part === 'string' && current && typeof current === 'object') current = (current as Record<string, unknown>)[part];
    else return undefined;
  }
  return current;
};

export interface AdmissionRemediation {
  id: string;
  path: string;
  currentValue: unknown;
  value?: SettingValue;
  label: string;
  rule_id: string;
}

export const getAdmissionRemediations = (config: JobConfig, result: AdmissionResult): AdmissionRemediation[] => {
  const actions: AdmissionRemediation[] = [];
  for (const diagnostic of result.diagnostics.filter(item => item.severity === 'error')) {
    for (const field of diagnostic.fields) {
      for (const parts of expandPaths(config, parsePath(field))) {
        const path = parts.map((part, index) => typeof part === 'number' ? `[${part}]` : index === 0 ? part : `.${part}`).join('');
        for (const rule of remediationRules[diagnostic.rule_id] ?? []) {
          if (!rule.field.test(path)) continue;
          const currentValue = readPath(config, parts);
          if (currentValue !== null && typeof currentValue === 'object') continue;
          if (currentValue === rule.value) continue;
          const id = `${diagnostic.rule_id}:${path}`;
          if (actions.some(action => action.id === id)) continue;
          actions.push({ id, path, currentValue, value: rule.value, label: rule.label, rule_id: diagnostic.rule_id });
        }
      }
    }
  }
  return actions;
};

export const applyAdmissionRemediation = (config: JobConfig, result: AdmissionResult, actionId: string): JobConfig => {
  const action = getAdmissionRemediations(config, result).find(item => item.id === actionId);
  if (!action) return config;
  const next = JSON.parse(JSON.stringify(config)) as JobConfig;
  const parts = parsePath(action.path);
  const leaf = parts[parts.length - 1];
  const parent = readPath(next, parts.slice(0, -1));
  if (!parent || typeof parent !== 'object' || Array.isArray(parent) || typeof leaf !== 'string') return config;
  if (action.value === undefined) delete (parent as Record<string, unknown>)[leaf];
  else (parent as Record<string, unknown>)[leaf] = action.value;
  return next;
};

export const formatAdmissionDiagnostics = (diagnostics: AdmissionDiagnostic[]): string =>
  diagnostics
    .map(diagnostic => {
      const fields = diagnostic.fields.length > 0 ? ` [${diagnostic.fields.join(', ')}]` : '';
      return `[${diagnostic.rule_id}]${fields} Reason: ${diagnostic.reason} Remedy: ${diagnostic.remedy}`;
    })
    .join('\n') || 'Canonical admission rejected this configuration.';

export const formatAdmissionError = (error: unknown): string => {
  const responseData = readResponseData(error);
  if (isAdmissionResult(responseData)) return formatAdmissionDiagnostics(responseData.diagnostics);
  if (error instanceof AdmissionClientError) return error.message;
  return readErrorMessage(error) || 'The request was blocked by canonical admission.';
};
