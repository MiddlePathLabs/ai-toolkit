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

type PathPart = string | number | '*';

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
  if (!(String(part) in record)) return [];
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

const deletePath = (value: unknown, path: PathPart[]): void => {
  if (path.length === 0 || !value || typeof value !== 'object') return;
  const parentPath = path.slice(0, -1);
  const leaf = path[path.length - 1];
  const parent = readPath(value, parentPath);
  if (!parent || typeof parent !== 'object') return;
  if (Array.isArray(parent) && typeof leaf === 'number') parent.splice(leaf, 1);
  else if (typeof leaf === 'string') delete (parent as Record<string, unknown>)[leaf];
};

const removableField = (field: string): boolean => !field.endsWith('.model.arch') && field !== 'config.process';


const compareRemovalPaths = (left: string, right: string): number => {
  const a = parsePath(left);
  const b = parsePath(right);
  // Remove nested values before their parents, then remove array siblings from
  // the end so deleting one item cannot invalidate a later concrete index.
  if (a.length !== b.length) return b.length - a.length;
  for (let index = 0; index < a.length; index += 1) {
    const leftPart = a[index];
    const rightPart = b[index];
    if (typeof leftPart === 'number' && typeof rightPart === 'number' && leftPart !== rightPart) {
      return rightPart - leftPart;
    }
    if (typeof leftPart === 'string' && typeof rightPart === 'string' && leftPart !== rightPart) {
      return rightPart.localeCompare(leftPart);
    }
  }
  return 0;
};

export interface IncompatibleSetting {
  path: string;
  value: unknown;
  rule_id: string;
}

export const getIncompatibleSettings = (config: JobConfig, result: AdmissionResult): IncompatibleSetting[] => {
  const settings: IncompatibleSetting[] = [];
  for (const diagnostic of result.diagnostics.filter(item => item.severity === 'error')) {
    for (const field of diagnostic.fields.filter(removableField)) {
      const paths = expandPaths(config, parsePath(field));
      for (const resolvedPath of paths) {
        const value = readPath(config, resolvedPath);
        if (value !== undefined && typeof resolvedPath[resolvedPath.length - 1] === 'string') {
          const path = resolvedPath
            .map((part, index) =>
              typeof part === 'number' ? `[${part}]` : index === 0 ? part : `.${part}`,
            )
            .join('');
          settings.push({ path, value, rule_id: diagnostic.rule_id });
        }
      }
    }
  }
  return settings.filter((setting, index, all) => all.findIndex(item => item.path === setting.path) === index);
};

export const removeIncompatibleSettings = (config: JobConfig, result: AdmissionResult): JobConfig => {
  const next = JSON.parse(JSON.stringify(config)) as JobConfig;
  for (const setting of getIncompatibleSettings(config, result).sort((left, right) => compareRemovalPaths(left.path, right.path))) {
    deletePath(next, parsePath(setting.path));
  }
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
