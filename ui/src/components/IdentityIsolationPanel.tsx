'use client';

import { useEffect, useRef, useState } from 'react';
import Card from '@/components/Card';
import { IdentityIsolationConfig, IdentityIsolationSubject, ProcessConfig } from '@/types';

type Preset = NonNullable<IdentityIsolationConfig['preset']>;
type Caption = {
  original: string;
  training: string;
  preservation: string;
  subject_index?: number;
  source_path?: string;
};
type Exposure = {
  basis?: string;
  source_original_count?: number;
  generated_original_count?: number;
  effective_count?: number;
  caveat?: string;
  datasets?: {
    subject_index: number | null;
    category: string;
    resolution: number | number[];
    original_count: number;
    effective_count: number;
    probability: number;
    expected_sample_exposures: number;
  }[];
};
type Preview = {
  caption?: Caption;
  captions?: Caption[];
  resolved?: unknown;
  exposure?: Exposure;
  composites?: { path: string; subject_index: number; caption: string; thumbnail: string }[];
};
const inputClass =
  'w-full text-sm px-3 py-1 bg-gray-950 dark:bg-gray-800 border border-gray-700 rounded-sm text-gray-100';
const buttonClass = 'px-3 py-1 text-sm border border-gray-600 rounded-sm hover:bg-gray-800 disabled:opacity-50';
const emptySubject = (): IdentityIsolationSubject => ({ folder_path: '', trigger: '', class_name: 'person' });

export function isolationPreset(preset: Preset, subjects: IdentityIsolationSubject[] = []): IdentityIsolationConfig {
  return {
    preset,
    subjects,
    fused_captions: preset === 'full',
    composites: preset === 'full',
    composite_count: 16,
    composite_lr_multiplier: 2,
    dop_enabled: preset !== 'off',
    dop_interval: preset === 'full' ? 2 : 1,
    dop_weight: preset === 'full' ? 2 : 1,
    balance_subjects: false,
    stranger_asset_dir: 'data/identity_isolation/strangers',
    generated_dir: 'data/identity_isolation/generated',
    seed: 42,
  };
}

function CaptionPreview({ caption }: { caption: Caption }) {
  return (
    <dl className="space-y-1 text-sm break-words">
      {caption.source_path && (
        <>
          <dt className="text-gray-400">Source</dt>
          <dd>{caption.source_path}</dd>
        </>
      )}
      <dt className="text-gray-400">Original</dt>
      <dd className="whitespace-pre-wrap">{caption.original}</dd>
      <dt className="text-gray-400">Training</dt>
      <dd className="whitespace-pre-wrap">{caption.training}</dd>
      <dt className="text-gray-400">Preservation</dt>
      <dd className="whitespace-pre-wrap">{caption.preservation}</dd>
    </dl>
  );
}

function ExposurePreview({ exposure }: { exposure: Exposure }) {
  if (!exposure.datasets) return <p className="text-sm">No active Isolation exposure report.</p>;
  return (
    <section className="space-y-2 text-sm">
      <h3 className="font-medium">Exposure statistics</h3>
      <p>
        {exposure.basis === 'actual_loader_slots' ? 'Actual loader slots' : 'Configured estimate'}. Source originals:{' '}
        {exposure.source_original_count ?? '—'}; generated originals: {exposure.generated_original_count ?? '—'};
        effective samples: {exposure.effective_count ?? '—'}.
      </p>
      <div className="overflow-x-auto">
        <table className="w-full text-left">
          <thead>
            <tr>
              {[
                'Subject',
                'Category',
                'Resolution',
                'Original files',
                'Effective samples',
                'Sample probability',
                'Expected exposures',
              ].map(label => (
                <th key={label} className="border-b border-gray-700 p-2 font-medium">
                  {label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {exposure.datasets.map((row, index) => (
              <tr key={index}>
                <td className="p-2">{row.subject_index === null ? 'Unmapped' : row.subject_index + 1}</td>
                <td className="p-2">{row.category}</td>
                <td className="p-2">{String(row.resolution)}</td>
                <td className="p-2">{row.original_count}</td>
                <td className="p-2">{row.effective_count}</td>
                <td className="p-2">{(row.probability * 100).toFixed(1)}%</td>
                <td className="p-2">{row.expected_sample_exposures.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-gray-400">{exposure.caveat}</p>
    </section>
  );
}

export default function IdentityIsolationPanel({
  process,
  onChange,
}: {
  process: ProcessConfig;
  onChange: (value: IdentityIsolationConfig) => void;
}) {
  const config = process.identity_isolation;
  const value = { ...isolationPreset(config?.preset ?? 'off'), ...config };
  const [caption, setCaption] = useState('');
  const [preview, setPreview] = useState<Preview | null>(null);
  const [error, setError] = useState('');
  const [pending, setPending] = useState<string | null>(null);
  const requestId = useRef(0);
  useEffect(() => {
    requestId.current += 1;
    setPreview(null);
    setError('');
    setPending(null);
  }, [process]);
  const invalidate = () => {
    requestId.current += 1;
    setPreview(null);
    setError('');
    setPending(null);
  };
  const update = (patch: Partial<IdentityIsolationConfig>) => {
    invalidate();
    onChange({ ...value, ...patch });
  };
  const subjects = value.subjects ?? [];
  const updateSubject = (index: number, patch: Partial<IdentityIsolationSubject>) => {
    update({ subjects: subjects.map((subject, i) => (i === index ? { ...subject, ...patch } : subject)) });
  };
  const run = async (action: 'caption' | 'inspect' | 'prepare') => {
    const id = ++requestId.current;
    setPending(action);
    setError('');
    setPreview(null);
    try {
      const response = await fetch('/api/identity-isolation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, process, caption }),
      });
      const data = await response.json();
      if (!response.ok || !data.ok) throw new Error(data.error || 'Identity Isolation request failed.');
      if (id === requestId.current) setPreview(data.result);
    } catch (failure) {
      if (id === requestId.current) setError(failure instanceof Error ? failure.message : String(failure));
    } finally {
      if (id === requestId.current) setPending(null);
    }
  };
  const textField = (key: 'stranger_asset_dir' | 'generated_dir', label: string) => (
    <label className="block text-sm">
      {label}
      <input
        className={inputClass}
        value={value[key] ?? ''}
        onChange={event => update({ [key]: event.target.value })}
      />
    </label>
  );
  const numberField = (
    key: 'composite_count' | 'composite_lr_multiplier' | 'dop_interval' | 'dop_weight' | 'seed',
    label: string,
    min: number,
    step: number | 'any' = 1,
  ) => (
    <label className="block text-sm">
      {label}
      <input
        type="number"
        className={inputClass}
        value={value[key] ?? ''}
        min={min}
        step={step}
        onChange={event => {
          if (event.target.value !== '') update({ [key]: Number(event.target.value) });
        }}
      />
    </label>
  );
  const toggle = (key: 'fused_captions' | 'composites' | 'dop_enabled' | 'balance_subjects', label: string) => (
    <label className="flex items-center gap-2 text-sm">
      <input
        type="checkbox"
        checked={value[key] ?? false}
        onChange={event => update({ [key]: event.target.checked })}
      />
      {label}
    </label>
  );

  return (
    <Card title="LoRA Identity Isolation" collapsible>
      <div className="space-y-4" role="group" aria-label="LoRA Identity Isolation settings">
        <label className="block text-sm">
          Preset
          <select
            className={inputClass}
            value={value.preset}
            onChange={event => {
              const preset = event.target.value as Preset;
              invalidate();
              onChange(isolationPreset(preset, preset === 'full' && !subjects.length ? [emptySubject()] : subjects));
            }}
          >
            <option value="off">Off</option>
            <option value="dop_only">DOP only</option>
            <option value="full">Full Isolation</option>
          </select>
        </label>
        <p className="text-sm text-gray-400">
          Presets supply initial values below. Explicit component settings win, including fusion-only configurations.
          Off does not disable legacy training DOP. Existing configurations are unchanged until you edit these controls.{' '}
          When Isolation DOP is enabled, existing legacy DOP fields must agree: enable legacy DOP and match its
          multiplier, or remove duplicate legacy fields in advanced configuration.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {toggle('fused_captions', 'Fuse trigger and class in captions')}
          {toggle('balance_subjects', 'Balance subject exposure')}
          {toggle('composites', 'Enable local stranger composites')}
          {toggle('dop_enabled', 'Enable isolation DOP')}
          {numberField('composite_count', 'Composites per subject', 1)}
          {numberField('composite_lr_multiplier', 'Composite learning-rate multiplier', 0.000001, 'any')}
          {numberField('dop_interval', 'DOP interval', 1)}
          {numberField('dop_weight', 'DOP weight', 0.000001, 'any')}
          {textField('stranger_asset_dir', 'Local stranger asset directory')}
          {textField('generated_dir', 'Generated composite directory')}
          {numberField('seed', 'Composite seed', 0)}
        </div>
        <fieldset className="space-y-3 border-t border-gray-700 pt-3">
          <legend className="text-sm">Subjects (1–4)</legend>
          {subjects.map((subject, index) => (
            <fieldset key={index} className="border border-gray-700 p-3">
              <legend className="text-sm">Subject {index + 1}</legend>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                {(['folder_path', 'trigger', 'class_name'] as const).map(key => (
                  <label key={key} className="block text-sm">
                    {{ folder_path: 'Dataset folder', trigger: 'Trigger', class_name: 'Class name' }[key]}
                    <input
                      className={inputClass}
                      value={subject[key]}
                      onChange={event => updateSubject(index, { [key]: event.target.value })}
                    />
                  </label>
                ))}
              </div>
              <button
                type="button"
                className={`${buttonClass} mt-3`}
                disabled={
                  subjects.length <= 1 && !!(value.fused_captions || value.composites || value.balance_subjects)
                }
                onClick={() => update({ subjects: subjects.filter((_, i) => i !== index) })}
              >
                Remove subject {index + 1}
              </button>
            </fieldset>
          ))}
          <button
            type="button"
            className={buttonClass}
            disabled={subjects.length >= 4}
            onClick={() => update({ subjects: [...subjects, emptySubject()] })}
          >
            Add subject
          </button>
        </fieldset>
        <p className="text-sm text-gray-400">
          Full Isolation requires Krea 2, declared Raw LoRA, AdamW or Rose, and data-loader workers = 0. A non-unit
          composite LR multiplier requires both gradient accumulation settings = 1, dataset pooling off, and bucket
          batches (or non-bucket batch size = 1). No optimizer, workers, accumulation, or training steps are changed
          here. Backend validation reports unsupported settings.
        </p>
        <div className="border-t border-gray-700 pt-3 space-y-3">
          <label className="block text-sm">
            Example caption
            <textarea
              className={inputClass}
              rows={3}
              value={caption}
              onChange={event => {
                invalidate();
                setCaption(event.target.value);
              }}
            />
          </label>
          <div className="flex flex-wrap gap-3">
            <button type="button" className={buttonClass} disabled={!!pending} onClick={() => run('caption')}>
              Preview caption
            </button>
            <button type="button" className={buttonClass} disabled={!!pending} onClick={() => run('inspect')}>
              Inspect datasets and exposure
            </button>
            <button type="button" className={buttonClass} disabled={!!pending} onClick={() => run('prepare')}>
              Prepare composites (local assets only)
            </button>
          </div>
          <p className="text-sm text-gray-400">
            Inspect does not generate files. Prepare explicitly writes composites to the generated directory using
            existing local assets only; it never downloads models or assets or starts training.
          </p>
          {pending && (
            <p role="status" className="text-sm">
              Running {pending}…
            </p>
          )}
          {error && (
            <p role="alert" className="text-sm text-red-400 whitespace-pre-wrap">
              {error}
            </p>
          )}
          {preview?.caption && <CaptionPreview caption={preview.caption} />}
          {preview?.captions?.map((item, index) => (
            <div key={index} className="border-t border-gray-700 pt-3">
              <CaptionPreview caption={item} />
            </div>
          ))}
          {preview?.exposure !== undefined && <ExposurePreview exposure={preview.exposure} />}
          {preview?.composites?.map(item => (
            <figure key={item.path} className="border-t border-gray-700 pt-3">
              {item.thumbnail.startsWith('data:image/png;base64,') && (
                <img
                  src={item.thumbnail}
                  alt={`Prepared composite for subject ${item.subject_index + 1}`}
                  className="max-w-full"
                />
              )}
              <figcaption className="text-sm whitespace-pre-wrap break-words mt-2">
                {item.caption}
                <br />
                {item.path}
              </figcaption>
            </figure>
          ))}
          {preview?.resolved !== undefined && (
            <details>
              <summary className="text-sm cursor-pointer">Resolved backend settings</summary>
              <pre className="text-xs overflow-auto whitespace-pre-wrap break-words mt-2">
                {JSON.stringify(preview.resolved, null, 2)}
              </pre>
            </details>
          )}
        </div>
      </div>
    </Card>
  );
}
