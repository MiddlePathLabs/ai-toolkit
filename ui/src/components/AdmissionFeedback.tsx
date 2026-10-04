'use client';

import { AlertTriangle, CircleAlert, Trash2 } from 'lucide-react';
import type { JobConfig } from '@/types';
import {
  getIncompatibleSettings,
  removeIncompatibleSettings,
  type AdmissionDiagnostic,
  type AdmissionResult,
} from '@/utils/admission';

interface Props {
  config: JobConfig;
  result: AdmissionResult | null;
  unavailable?: string | null;
  onRemove?: (config: JobConfig) => void;
}

const renderValue = (value: unknown): string => {
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

export default function AdmissionFeedback({ config, result, unavailable, onRemove }: Props) {
  if (!result && !unavailable) return null;
  const errors = result?.diagnostics.filter(item => item.severity === 'error') ?? [];
  const warnings = result?.diagnostics.filter(item => item.severity === 'warning') ?? [];
  const deferred = result?.deferred ?? [];
  const settings = result ? getIncompatibleSettings(config, result) : [];

  return (
    <section
      aria-live="polite"
      className="mx-4 mt-3 rounded-md border border-amber-500/50 bg-gray-900/80 p-3 text-sm text-gray-100"
    >
      <div className="flex items-center gap-2 font-semibold">
        {unavailable || errors.length > 0 ? <CircleAlert className="h-4 w-4 text-red-400" /> : <AlertTriangle className="h-4 w-4 text-amber-300" />}
        <span>{unavailable ? 'Validation unavailable' : errors.length > 0 ? 'Configuration cannot run' : 'Validation status'}</span>
      </div>
      {unavailable && <p className="mt-2 text-red-200">{unavailable} Save and Start stay disabled until validation is available.</p>}
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
          <div className="font-medium text-amber-200">Deferred runtime checks</div>
          <ul className="mt-1 space-y-2 text-amber-100">{deferred.map((diagnostic, index) => <Diagnostic diagnostic={diagnostic} key={`${diagnostic.rule_id}-${index}`} />)}</ul>
        </div>
      )}
      {settings.length > 0 && result && onRemove && (
        <div className="mt-3 border-t border-gray-700 pt-3">
          <div className="font-medium">Incompatible values remain stored</div>
          <p className="mt-1 text-xs text-gray-300">Nothing is removed automatically when a model card hides a field. Review these values and explicitly remove them if they are no longer wanted.</p>
          <ul className="mt-2 space-y-1 text-xs text-gray-200">
            {settings.map(setting => (
              <li className="font-mono" key={setting.path}>
                {setting.path} = {renderValue(setting.value)}
              </li>
            ))}
          </ul>
          <button
            type="button"
            className="mt-3 inline-flex items-center gap-2 rounded bg-red-700 px-3 py-1.5 text-xs font-medium hover:bg-red-600"
            onClick={() => onRemove(removeIncompatibleSettings(config, result))}
          >
            <Trash2 className="h-3.5 w-3.5" />
            Remove listed incompatible settings
          </button>
        </div>
      )}
    </section>
  );
}
