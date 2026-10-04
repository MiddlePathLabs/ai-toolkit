'use client';

import { AlertTriangle, CircleAlert, Wrench } from 'lucide-react';
import type { JobConfig } from '@/types';
import {
  getAdmissionRemediations,
  applyAdmissionRemediation,
  type AdmissionDiagnostic,
  type AdmissionResult,
  type AdmissionValidationStatus,
} from '@/utils/admission';

interface Props {
  config: JobConfig;
  result: AdmissionResult | null;
  unavailable?: string | null;
  status?: AdmissionValidationStatus;
  onRemediate?: (config: JobConfig) => void;
}

const renderValue = (value: unknown): string => {
  if (value === undefined) return '(not set)';
  if (typeof value === 'string') return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
};

const Diagnostic = ({ diagnostic }: { diagnostic: AdmissionDiagnostic }) => (
  <li className="border-l-2 border-red-400 pl-3">
    <div className="font-medium">{diagnostic.reason}</div>
    <div className="text-xs text-red-200/80">
      <span className="font-mono">{diagnostic.rule_id}</span>
      {diagnostic.fields.length > 0 && <span> · {diagnostic.fields.join(', ')}</span>}
    </div>
    <div className="text-xs text-gray-300">Remedy: {diagnostic.remedy}</div>
  </li>
);

export default function AdmissionFeedback({ config, result, unavailable, status, onRemediate }: Props) {
  if (!result && !unavailable && status !== 'pending') return null;
  const errors = result?.diagnostics.filter(item => item.severity === 'error') ?? [];
  const warnings = result?.diagnostics.filter(item => item.severity === 'warning') ?? [];
  const deferred = result?.deferred ?? [];
  const actions = result ? getAdmissionRemediations(config, result) : [];
  const pending = status === 'pending';
  const title = pending ? 'Validating current configuration' :
    unavailable ? 'Validation unavailable' :
    errors.length > 0 ? 'Configuration cannot run' :
    deferred.length > 0 ? 'Static validation passed; runtime checks pending' : 'Configuration accepted';

  return (
    <section
      aria-live="polite"
      className="mx-4 mt-3 rounded-md border border-amber-500/50 bg-gray-900/80 p-3 text-sm text-gray-100"
    >
      <div className="flex items-center gap-2 font-semibold">
        {unavailable || errors.length > 0 ? <CircleAlert className="h-4 w-4 text-red-400" /> : <AlertTriangle className="h-4 w-4 text-amber-300" />}
        <span>{title}</span>
      </div>
      {pending && <p className="mt-2 text-gray-300">The current edits have not been validated yet. Save will be available after validation passes.</p>}
      {unavailable && <p className="mt-2 text-red-200">{unavailable} Save is disabled until this configuration can be validated.</p>}
      {errors.length > 0 && (
        <ul className="mt-2 space-y-2 text-red-100">
          {errors.map((diagnostic, index) => <Diagnostic diagnostic={diagnostic} key={`${diagnostic.rule_id}-${index}`} />)}
        </ul>
      )}
      {warnings.length > 0 && (
        <div className="mt-3 border-t border-gray-700 pt-2">
          <div className="font-medium text-amber-200">Warnings</div>
          <ul className="mt-1 space-y-2 text-amber-100">{warnings.map((diagnostic, index) => <Diagnostic diagnostic={diagnostic} key={`${diagnostic.rule_id}-${index}`} />)}</ul>
        </div>
      )}
      {deferred.length > 0 && (
        <div className="mt-3 border-t border-gray-700 pt-2">
          <div className="font-medium text-amber-200">Runtime checks pending</div>
          {result?.valid && <p className="mt-1 text-xs text-gray-300">Static validation passed, so this configuration can be saved. These checks must pass when training loads the model and data.</p>}
          <ul className="mt-1 space-y-2 text-amber-100">{deferred.map((diagnostic, index) => <Diagnostic diagnostic={diagnostic} key={`${diagnostic.rule_id}-${index}`} />)}</ul>
        </div>
      )}
      {errors.length > 0 && result && onRemediate && (
        <div className="mt-3 border-t border-gray-700 pt-3">
          <div className="font-medium">Stored values are unchanged</div>
          <p className="mt-1 text-xs text-gray-300">Apply a named repair below or edit the configuration. Hidden fields are not removed automatically. Repairs never remove dataset paths or model identities.</p>
          {actions.length > 0 && (
            <ul className="mt-2 space-y-3 text-xs text-gray-200">
              {actions.map(action => (
                <li key={action.id}>
                  <div className="font-mono">
                    {action.path}: {renderValue(action.currentValue)} → {action.value === undefined ? '(remove setting)' : renderValue(action.value)}
                  </div>
                  <button
                    type="button"
                    className="mt-1 inline-flex items-center gap-2 rounded bg-gray-700 px-3 py-1.5 text-xs font-medium hover:bg-gray-600"
                    onClick={() => onRemediate(applyAdmissionRemediation(config, result, action.id))}
                  >
                    <Wrench className="h-3.5 w-3.5" />
                    {action.label}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </section>
  );
}
