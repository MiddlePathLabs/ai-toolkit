// Checkpoint comparison section (Phase 2b, fork-only) for the identity panel.
//
// Renders the precomputed checkpoint_report.json: per-checkpoint stats and
// distribution strips, paired A/B comparison with bootstrap CIs, prompt
// category tagging, and the loss-based convergence panel (kept separate
// from identity numbers by design). Phrasing rule: "highest median on
// comparable prompts", never a best-checkpoint verdict.
import { useEffect, useMemo, useState } from 'react';
import classNames from 'classnames';
import { LuLoader, LuRefreshCw } from 'react-icons/lu';
import {
  CATEGORY_PRESETS,
  countsLabel,
  excludedReasonLabel,
  convergencePhrase,
  fmtScore,
  fmtSigned,
  promptIndicesInReport,
  stripDomain,
  stripX,
  verdictPhrase,
  type CheckpointReport,
  type CheckpointStepEntry,
  type PairComparison,
  type ScoreModel,
  type ScoreSource,
} from '@/utils/identityCheckpoint';

const controlClass =
  'rounded border border-gray-300 bg-white px-2 py-1 text-sm text-gray-900 dark:border-gray-600 dark:bg-gray-900 dark:text-gray-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-blue-500';

function stepValue(step: { step: number; is_raw: boolean }): string {
  return `${step.step}:${step.is_raw ? 1 : 0}`;
}

function findPair(report: CheckpointReport, aValue: string, bValue: string): PairComparison | null {
  return report.pairs.find(p => stepValue(p.a) === aValue && stepValue(p.b) === bValue) ?? null;
}

function ScoreStrip({
  steps,
  model,
  source,
  loo,
  onOpenSample,
}: {
  steps: CheckpointStepEntry[];
  model: ScoreModel;
  source: ScoreSource;
  loo: number | null;
  onOpenSample: (filename: string) => void;
}) {
  const domain = useMemo(() => stripDomain(steps, model, source), [steps, model, source]);
  const rows = steps.filter(step => (step.strip?.[model]?.[source] ?? []).length > 0);
  if (rows.length === 0) return null;
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-[10px] text-gray-400 dark:text-gray-500">
        <span>{domain.min.toFixed(3)}</span>
        <span>
          distribution of all evaluated samples · dashed line = reference LOO band{loo !== null ? ` (${loo.toFixed(3)})` : ''}
        </span>
        <span>{domain.max.toFixed(3)}</span>
      </div>
      {rows.map(step => (
        <div key={stepValue(step)} className="flex items-center gap-1">
          <span className="w-16 shrink-0 truncate text-right text-[10px] text-gray-500 dark:text-gray-400" title={step.label}>
            {step.label}
          </span>
          <div className="relative h-4 flex-1 rounded bg-gray-200/60 dark:bg-gray-800">
            {loo !== null && (
              <div
                className="absolute inset-y-0 w-px border-l border-dashed border-gray-400 dark:border-gray-500"
                style={{ left: `${stripX(loo, domain, 100)}%` }}
              />
            )}
            {(step.strip?.[model]?.[source] ?? []).map(entry => (
              <button
                key={entry.f}
                type="button"
                className="absolute top-1/2 h-2 w-2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-blue-500/80 hover:scale-150 hover:bg-blue-600"
                style={{ left: `${stripX(entry.s, domain, 100)}%` }}
                title={`${entry.f}\n${model} ${fmtScore(entry.s)} (${source})`}
                onClick={() => onOpenSample(entry.f)}
              />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function CategorySelect({
  promptIdx,
  text,
  value,
  onChange,
}: {
  promptIdx: number;
  text: string | undefined;
  value: string;
  onChange: (next: string) => void;
}) {
  const isPreset = value === '' || (CATEGORY_PRESETS as readonly string[]).includes(value);
  return (
    <div className="flex items-center gap-1">
      <span className="w-6 shrink-0 text-right text-[10px] tabular-nums text-gray-500 dark:text-gray-400">#{promptIdx}</span>
      <span className="min-w-0 flex-1 truncate text-[10px] text-gray-600 dark:text-gray-300" title={text ?? `prompt ${promptIdx}`}>
        {text ?? '(prompt text not in job config)'}
      </span>
      <select
        className={classNames(controlClass, 'w-28 px-1 py-0.5 text-[10px]')}
        value={isPreset ? value : '__custom'}
        onChange={e => onChange(e.target.value === '__custom' ? `cat-${promptIdx}` : e.target.value)}
      >
        <option value="">untagged</option>
        {CATEGORY_PRESETS.map(preset => (
          <option key={preset} value={preset}>
            {preset}
          </option>
        ))}
        <option value="__custom">custom…</option>
      </select>
      {!isPreset && (
        <input
          className={classNames(controlClass, 'w-24 px-1 py-0.5 text-[10px]')}
          value={value}
          maxLength={40}
          onChange={e => onChange(e.target.value)}
          placeholder="category"
        />
      )}
    </div>
  );
}

export default function IdentityCheckpointSection({
  report,
  loading,
  onRefresh,
  onSaveCategories,
  onOpenSample,
}: {
  report: CheckpointReport | null;
  loading: boolean;
  onRefresh: () => void;
  onSaveCategories: (categories: Record<string, string>) => void;
  onOpenSample: (filename: string) => void;
}) {
  const models = report?.models ?? [];
  const [model, setModel] = useState<ScoreModel>('arcface');
  const effectiveModel: ScoreModel = models.includes(model) ? model : (models[0] as ScoreModel) ?? 'arcface';
  const sources = report?.sources ?? [];
  const [source, setSource] = useState<ScoreSource>('training');
  const effectiveSource: ScoreSource = sources.includes(source) ? source : (sources[0] as ScoreSource) ?? 'training';

  const stepOptions = report?.steps ?? [];
  const [aValue, setAValue] = useState('');
  const [bValue, setBValue] = useState('');
  const defaultA = stepOptions.length > 0 ? stepValue(stepOptions[0]) : '';
  const defaultB = stepOptions.length > 1 ? stepValue(stepOptions[stepOptions.length - 1]) : '';
  const effectiveA = stepOptions.some(s => stepValue(s) === aValue) ? aValue : defaultA;
  const effectiveB = stepOptions.some(s => stepValue(s) === bValue) ? bValue : defaultB;
  const pair = report && effectiveA && effectiveB ? findPair(report, effectiveA, effectiveB) : null;

  const promptIndices = useMemo(() => (report ? promptIndicesInReport(report) : []), [report]);
  const [draft, setDraft] = useState<Record<string, string>>({});
  const [dirty, setDirty] = useState(false);
  useEffect(() => {
    if (!dirty) setDraft(report?.categories ?? {});
  }, [report, dirty]);

  if (!report) {
    return (
      <div className="rounded border border-dashed border-gray-300 p-2 text-xs text-gray-500 dark:border-gray-600 dark:text-gray-400">
        Checkpoint comparison: no report yet — run an evaluation first.
      </div>
    );
  }
  const modelName = effectiveModel === 'arcface' ? 'ArcFace' : 'SFace';
  const loo = report.refsets?.[effectiveSource]?.loo?.[effectiveModel] ?? null;

  return (
    <div className="space-y-3 rounded border border-gray-300 p-2 dark:border-gray-600">
      <div className="flex items-center gap-2">
        <span className="text-xs font-semibold">Checkpoint comparison</span>
        <div className="flex overflow-hidden rounded border border-gray-300 text-[10px] dark:border-gray-600">
          {(['arcface', 'sface'] as const).map(m => (
            <button
              key={m}
              type="button"
              disabled={!models.includes(m)}
              className={classNames(
                'px-1.5 py-0.5 capitalize',
                effectiveModel === m ? 'bg-blue-500 text-white' : 'bg-white text-gray-700 hover:bg-gray-100 disabled:opacity-40 dark:bg-gray-900 dark:text-gray-300 dark:hover:bg-gray-800',
              )}
              onClick={() => setModel(m)}
            >
              {m === 'arcface' ? 'A' : 'S'}
            </button>
          ))}
        </div>
        {sources.length > 1 && (
          <select
            className={classNames(controlClass, 'ml-auto px-1 py-0.5 text-[10px]')}
            value={effectiveSource}
            onChange={e => setSource(e.target.value as ScoreSource)}
          >
            {sources.map(s => (
              <option key={s} value={s}>
                {s} refs
              </option>
            ))}
          </select>
        )}
        <button type="button" className={classNames(controlClass, 'px-1.5 py-0.5 text-[10px]')} title="Recompute report" onClick={onRefresh}>
          {loading ? <LuLoader className="inline animate-spin" size={11} /> : <LuRefreshCw className="inline" size={11} />}
        </button>
      </div>

      {/* per-checkpoint table: median (p10) + IQR tooltip; no best-checkpoint verdict */}
      <table className="w-full table-fixed text-xs tabular-nums">
        <thead>
          <tr className="text-gray-500 dark:text-gray-400">
            <th className="text-left font-medium">Checkpoint</th>
            <th className="font-medium" title="evaluated / ambiguous / no-face / failed samples">
              n ev/amb/nf/fail
            </th>
            <th className="font-medium" title={`highest median ${modelName} on comparable prompts — evidence, not a best-checkpoint verdict`}>
              {modelName} med
            </th>
            {report.steps.some(s => s.train_vs_external) && (
              <th className="font-medium" title="training-ref median − external-ref median; a rising gap can indicate memorization">
                T−E gap
              </th>
            )}
          </tr>
        </thead>
        <tbody>
          {report.steps.map(step => {
            const stats = step.stats?.[effectiveModel]?.[effectiveSource];
            const gap = step.train_vs_external?.[effectiveModel]?.gap;
            return (
              <tr key={stepValue(step)} className="border-t border-gray-200 dark:border-gray-700">
                <td className="py-0.5 text-left">{step.label}</td>
                <td className="py-0.5 text-center text-gray-500 dark:text-gray-400">{countsLabel(step.counts)}</td>
                <td className="py-0.5 text-center" title={stats?.iqr != null ? `IQR ${fmtScore(stats.iqr)} · p25 ${fmtScore(stats.p25)} · p75 ${fmtScore(stats.p75)} · n ${stats.n}` : undefined}>
                  {fmtScore(stats?.median)}
                  {stats?.p10 != null && <span className="text-gray-400"> p10 {fmtScore(stats.p10)}</span>}
                </td>
                {report.steps.some(s => s.train_vs_external) && (
                  <td className="py-0.5 text-center text-gray-500 dark:text-gray-400">{gap == null ? '–' : fmtSigned(gap)}</td>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>

      {/* distribution strips per checkpoint */}
      <ScoreStrip steps={report.steps} model={effectiveModel} source={effectiveSource} loo={loo} onOpenSample={onOpenSample} />

      {/* paired comparison */}
      <div className="space-y-1.5 rounded border border-gray-300 p-1.5 dark:border-gray-600">
        <div className="flex items-center gap-1 text-xs">
          <span className="font-medium">Paired comparison</span>
          <select className={classNames(controlClass, 'px-1 py-0.5 text-[10px]')} value={effectiveA} onChange={e => setAValue(e.target.value)}>
            {stepOptions.map(step => (
              <option key={stepValue(step)} value={stepValue(step)}>
                {step.label}
              </option>
            ))}
          </select>
          <span className="text-gray-400">vs</span>
          <select className={classNames(controlClass, 'px-1 py-0.5 text-[10px]')} value={effectiveB} onChange={e => setBValue(e.target.value)}>
            {stepOptions.filter(step => stepValue(step) !== effectiveA).map(step => (
              <option key={stepValue(step)} value={stepValue(step)}>
                {step.label}
              </option>
            ))}
          </select>
        </div>
        {!pair && <p className="text-[10px] text-gray-500 dark:text-gray-400">Pick two different checkpoints.</p>}
        {pair &&
          sources.map(sourceKey => {
            const comparison = pair.models[`${effectiveModel}:${sourceKey}`];
            if (!comparison) return null;
            return (
              <div key={sourceKey} className="space-y-0.5 text-[10px]">
                <div className="tabular-nums text-gray-700 dark:text-gray-200">
                  <span className="text-gray-500 dark:text-gray-400">{sourceKey} refs: </span>
                  Δmed {fmtSigned(comparison.diff_median)}{' '}
                  {comparison.ci_low != null && comparison.ci_high != null && (
                    <span className="text-gray-400">
                      (CI {fmtSigned(comparison.ci_low)} … {fmtSigned(comparison.ci_high)})
                    </span>
                  )}{' '}
                  · {pair.b.label} wins {comparison.wins_b}/{comparison.n_pairs}
                </div>
                <div
                  className={classNames(
                    comparison.verdict === 'higher' || comparison.verdict === 'lower'
                      ? 'font-medium text-gray-900 dark:text-gray-100'
                      : 'text-gray-500 dark:text-gray-400',
                  )}
                >
                  {verdictPhrase(comparison, effectiveModel, pair.a.label, pair.b.label)}
                </div>
                {comparison.excluded.length > 0 && (
                  <details className="text-gray-400 dark:text-gray-500">
                    <summary className="cursor-pointer select-none">
                      {comparison.excluded.length} pair{comparison.excluded.length === 1 ? '' : 's'} excluded from direct comparison
                    </summary>
                    <ul className="ml-3 list-disc">
                      {comparison.excluded.map(ex => (
                        <li key={ex.id}>
                          {ex.id} — {excludedReasonLabel(ex.reason)}
                        </li>
                      ))}
                    </ul>
                  </details>
                )}
              </div>
            );
          })}
      </div>

      {/* prompt categories */}
      {promptIndices.length > 0 && (
        <div className="space-y-1">
          <div className="flex items-center justify-between text-xs">
            <span className="font-medium" title="Tags split the stats because solo, multi and gate prompts behave very differently">
              Prompt categories
            </span>
            <button
              type="button"
              className={classNames(controlClass, 'px-1.5 py-0.5 text-[10px]')}
              disabled={!dirty || loading}
              onClick={() => {
                onSaveCategories(draft);
                setDirty(false);
              }}
            >
              Save tags
            </button>
          </div>
          {promptIndices.map(idx => (
            <CategorySelect
              key={idx}
              promptIdx={idx}
              text={report.prompt_texts?.[String(idx)]}
              value={draft[String(idx)] ?? ''}
              onChange={next => {
                setDraft(prev => ({ ...prev, [String(idx)]: next }));
                setDirty(true);
              }}
            />
          ))}
          {report.steps.some(s => Object.keys(s.by_category ?? {}).length > 1) && (
            <table className="w-full table-fixed text-[10px] tabular-nums">
              <thead>
                <tr className="text-gray-500 dark:text-gray-400">
                  <th className="text-left font-medium">Category</th>
                  {report.steps.map(step => (
                    <th key={stepValue(step)} className="font-medium" title={`${step.label}: median ${modelName} (${effectiveSource})`}>
                      {step.label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(
                  Object.fromEntries(
                    report.steps.flatMap(step =>
                      Object.entries(step.by_category ?? {}).map(([cat]) => [cat, true] as const),
                    ),
                  ),
                ).map(([cat]) => (
                  <tr key={cat} className="border-t border-gray-200 dark:border-gray-700">
                    <td className="py-0.5 text-left">{cat}</td>
                    {report.steps.map(step => (
                      <td key={stepValue(step)} className="py-0.5 text-center">
                        {fmtScore(step.by_category?.[cat]?.[effectiveModel]?.[effectiveSource]?.median)}
                        <span className="text-gray-400"> ({step.by_category?.[cat]?.[effectiveModel]?.[effectiveSource]?.n ?? 0})</span>
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {/* convergence gate: loss-based, clearly separate from identity numbers */}
      {report.convergence && (
        <div className="space-y-1 rounded border border-amber-400/50 bg-amber-500/5 p-1.5 text-[10px] dark:border-amber-500/40">
          <div className="font-semibold text-amber-700 dark:text-amber-300">
            Training convergence — loss heuristic, separate from identity
          </div>
          <p className="text-gray-600 dark:text-gray-300">{convergencePhrase(report.convergence)}</p>
          <p className="tabular-nums text-gray-500 dark:text-gray-400">
            last step {report.convergence.last_step.toLocaleString()} · late-500 loss change{' '}
            {report.convergence.late_500_change_pct == null ? '–' : `${report.convergence.late_500_change_pct.toFixed(1)}%`} · LR{' '}
            {report.convergence.lr_start.toExponential(1)} → {report.convergence.lr_end.toExponential(1)}
          </p>
        </div>
      )}

      <ul className="list-disc space-y-0.5 pl-4 text-[10px] text-gray-400 dark:text-gray-500">
        {report.notes.map(note => (
          <li key={note}>{note}</li>
        ))}
      </ul>
    </div>
  );
}
