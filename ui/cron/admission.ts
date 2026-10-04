import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import YAML from 'yaml';
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

export class AdmissionUnavailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'AdmissionUnavailableError';
  }
}
export const formatAdmissionDiagnostics = (diagnostics: AdmissionDiagnostic[]): string =>
  diagnostics
    .map(diagnostic => {
      const fields = diagnostic.fields.length > 0 ? ` [${diagnostic.fields.join(', ')}]` : '';
      return `[${diagnostic.rule_id}]${fields} Reason: ${diagnostic.reason} Remedy: ${diagnostic.remedy}`;
    })
    .join('\n') || 'Canonical admission rejected this configuration.';

const isWindows = process.platform === 'win32';
const toolkitRoot = path.resolve('@', '..', '..');

export const resolveManagedPythonPath = (): string | null => {
  const configured = process.env.AI_TOOLKIT_PYTHON;
  const candidates = configured
    ? [configured]
    : [
        path.join(toolkitRoot, '..', 'python_embeded', isWindows ? 'python.exe' : 'python'),
        path.join(toolkitRoot, 'python_embeded', isWindows ? 'python.exe' : 'python'),
      ];
  return candidates.find(candidate => fs.existsSync(candidate)) ?? null;
};


const isAdmissionResult = (value: unknown): value is AdmissionResult => {
  if (!value || typeof value !== 'object') return false;
  const result = value as Record<string, unknown>;
  return typeof result.valid === 'boolean' && Array.isArray(result.diagnostics) && Array.isArray(result.deferred);
};

export const runAdmission = (config: unknown, source = 'api'): Promise<AdmissionResult> => {
  const python = resolveManagedPythonPath();
  if (!python) return Promise.reject(new AdmissionUnavailableError('managed interpreter was not found'));

  return new Promise((resolve, reject) => {
    const child = spawn(python, ['-m', 'toolkit.admission', '--stdin', '--source', source], {
      cwd: toolkitRoot,
      stdio: ['pipe', 'pipe', 'pipe'],
      windowsHide: true,
    });
    let stdout = '';
    let stderr = '';
    let settled = false;
    const timer = setTimeout(() => {
      child.kill();
      if (!settled) {
        settled = true;
        reject(new AdmissionUnavailableError('managed admission timed out'));
      }
    }, 15000);

    child.stdout.on('data', chunk => {
      stdout += chunk.toString();
    });
    child.stderr.on('data', chunk => {
      stderr += chunk.toString();
    });
    child.once('error', error => {
      clearTimeout(timer);
      if (!settled) {
        settled = true;
        reject(new AdmissionUnavailableError(error.message));
      }
    });
    child.once('close', code => {
      clearTimeout(timer);
      if (settled) return;
      settled = true;
      let parsed: unknown;
      try {
        parsed = JSON.parse(stdout);
      } catch {
        reject(new AdmissionUnavailableError(stderr.trim() || 'managed admission returned no JSON'));
        return;
      }
      if (!isAdmissionResult(parsed)) {
        reject(new AdmissionUnavailableError('managed admission returned an invalid result'));
        return;
      }
      if (code !== 0 && parsed.valid) {
        reject(new AdmissionUnavailableError(stderr.trim() || `managed admission exited with ${code}`));
        return;
      }
      resolve(parsed);
    });
    child.stdin.end(JSON.stringify(config));
  });
};

export type AdmissionRunner = (config: unknown, source: string) => Promise<AdmissionResult>;

export interface StoredConfigAdmissionFailure {
  status: 422 | 503;
  body: Record<string, unknown>;
}

export interface StoredConfigAdmissionOptions {
  source?: string;
  runner?: AdmissionRunner;
  context?: Record<string, unknown>;
}

const nonTrainingProcessType = (type: string): boolean => {
  const normalized = type.trim().toLowerCase();
  return normalized === 'inferenceengine' || normalized.endsWith('captioner');
};

export const shouldRunTrainingAdmission = (config: unknown): boolean => {
  if (!config || typeof config !== 'object') return true;
  const root = config as Record<string, unknown>;
  const configValue = root.config;
  if (!configValue || typeof configValue !== 'object') return true;
  const processValue = (configValue as Record<string, unknown>).process;
  if (!Array.isArray(processValue) || processValue.length === 0) return true;
  return processValue.some(process => {
    if (!process || typeof process !== 'object') return true;
    const type = (process as Record<string, unknown>).type;
    return typeof type !== 'string' || !nonTrainingProcessType(type);
  });
};

export async function validateStoredConfigBeforeMutation(
  rawConfig: string,
  options: StoredConfigAdmissionOptions = {},
): Promise<StoredConfigAdmissionFailure | null> {
  const context = options.context ?? {};
  let config: unknown;
  try {
    config = JSON.parse(rawConfig);
  } catch {
    return {
      status: 422,
      body: { error: 'Stored job configuration is not valid JSON.', ...context },
    };
  }
  if (!shouldRunTrainingAdmission(config)) return null;

  try {
    const admission = await (options.runner ?? runAdmission)(config, options.source ?? 'api');
    if (admission.valid) return null;
    return { status: 422, body: { ...admission, ...context } };
  } catch (error) {
    if (error instanceof AdmissionUnavailableError) {
      return {
        status: 503,
        body: {
          error: 'Canonical admission is unavailable. The configuration was not validated.',
          code: 'admission_unavailable',
          ...context,
        },
      };
    }
    return {
      status: 503,
      body: {
        error: 'Canonical admission failed. The configuration was not validated.',
        code: 'admission_unavailable',
        ...context,
      },
    };
  }
}

export type RawYamlResult =
  | { valid: true; value: unknown }
  | { valid: false; message: string; line: number };

export const parseRawYaml = (text: string): RawYamlResult => {
  try {
    const document = YAML.parseDocument(text);
    if (document.errors.length > 0) {
      const first = document.errors[0];
      const line = first.linePos?.[0]?.line ?? 1;
      return { valid: false, message: first.message, line };
    }
    return { valid: true, value: document.toJSON() };
  } catch (error) {
    return { valid: false, message: error instanceof Error ? error.message : String(error), line: 1 };
  }
};

export const rawYamlDiagnostic = (rawYaml: string): AdmissionResult | null => {
  const parsed = parseRawYaml(rawYaml);
  if (parsed.valid === true) return null;
  return {
    valid: false,
    diagnostics: [
      {
        rule_id: 'admission.raw_yaml_invalid',
        severity: 'error',
        fields: ['config'],
        reason: `The YAML editor contains invalid syntax: ${parsed.message} (line ${parsed.line}).`,
        remedy: 'Fix the highlighted YAML syntax before saving or starting this job.',
        phase: 'static',
      },
    ],
    deferred: [],
    source: 'ui',
  };
};
