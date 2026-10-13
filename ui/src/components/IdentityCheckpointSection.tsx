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
import IdentityRetentionGraph from './IdentityRetentionGraph';
import {
  CATEGORY_PRESETS,
  countsLabel,
  excludedReasonLabel,
  convergencePhrase,
  driftHiddenCount,
  driftKindLabel,
  driftSignals,
  fmtScore,
  fmtSigned,
  highestMedianSteps,
  pairInOrientation,
  promptIndicesInReport,
  retentionCategories,
  retentionWeightsKinds,
  stepValue,
  stripDomain,
  stripX,
  verdictPhrase,
  type CheckpointReport,
  type CheckpointStepEntry,
  type DriftSignal,
  type RetentionWeightsKind,
  type ScoreModel,
  type ScoreSource,
} from '@/utils/identityCheckpoint';

const controlClass =
  'rounded border border-gray-300 bg-white px-2 py-1 text-sm text-gray-900 dark:border-gray-600 dark:bg-gray-900 dark:text-gray-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-blue-500';

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
          distribution of all evaluated samples · dashed line = leave-one-out (LOO) reference band
          {loo !== null ? ` (${loo.toFixed(3)})` : ''}
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

// Phase 2d: statistical drift warnings. Amber like the convergence panel,
// but the wording is weaker on purpose: "possible drift" is the strongest
// claim the tool ever makes — anything stronger comes from the user's own
// review of the linked samples. Signals are per model and filtered to the
// section's model/source selection; the hidden count surfaces the rest.
function DriftWarnings({
  report,
  drift,
  signals,
  hidden,
  model,
  source,
  onOpenSample,
}: {
  report: CheckpointReport;
  drift: NonNullable<CheckpointReport['drift']>;
  signals: DriftSignal[];
  hidden: number;
  model: ScoreModel;
  source: ScoreSource;
  onOpenSample: (filename: string) => void;
}) {
  const modelName = model === 'arcface' ? 'ArcFace' : 'SFace';
  const scopeLabel = (signal: DriftSignal): string | null => {
    if (!signal.scope) return null;
    const entry = report.steps.find(step => step.step === signal.scope?.step && step.is_raw === signal.scope?.is_raw);
    return entry ? entry.label : `step ${signal.scope.step}`;
  };
  return (
    <div
      className={classNames(
        'space-y-1 rounded border p-1.5 text-[10px]',
        signals.length > 0
          ? 'border-amber-400/50 bg-amber-500/5 dark:border-amber-500/40'
          : 'border-gray-300 dark:border-gray-600',
      )}
    >
      <div className="flex items-baseline justify-between gap-2">
        <span
          className="font-semibold text-amber-700 dark:text-amber-300"
          title="Statistical warnings from the checkpoint data (similarity decline, spread growth, memorization signature, pose dependence, category spread, near-copies). Computed per model; ArcFace and SFace are never merged."
        >
          Identity drift warnings
        </span>
        <span className="text-gray-400 dark:text-gray-500">possible drift — never a verdict</span>
      </div>
      {signals.length === 0 ? (
        <p className="text-gray-600 dark:text-gray-300">
          No drift signals for {modelName} ({source} refs){hidden > 0 ? ` — ${hidden} hidden by this filter` : ''}.
        </p>
      ) : (
        <div className="space-y-1.5">
          {signals.map((signal, index) => (
            <div key={index} className="space-y-0.5">
              <div className="text-gray-500 dark:text-gray-400">
                <span className="font-medium text-gray-700 dark:text-gray-200">{driftKindLabel(signal.kind)}</span>
                {signal.source && <span> · {signal.source} refs</span>}
                {scopeLabel(signal) && <span> · {scopeLabel(signal)}</span>}
              </div>
              <p className="text-gray-600 dark:text-gray-300">{signal.message}</p>
              {signal.samples.length > 0 && (
                <div className="flex flex-wrap items-center gap-1">
                  <span className="text-gray-400 dark:text-gray-500">review at full resolution:</span>
                  {signal.samples.map((filename, i) => (
                    <button
                      key={filename}
                      type="button"
                      className="rounded border border-amber-400/60 px-1.5 py-0.5 tabular-nums text-gray-600 hover:bg-amber-500/10 dark:border-amber-500/40 dark:text-gray-300"
                      title={filename}
                      onClick={() => onOpenSample(filename)}
                    >
                      {i + 1}
                    </button>
                  ))}
                </div>
              )}
            </div>
          ))}
          {hidden > 0 && (
            <p className="text-gray-400 dark:text-gray-500">
              {hidden} warning{hidden === 1 ? '' : 's'} for the other model/source hidden by the {modelName} / {source}{' '}
              filter.
            </p>
          )}
        </div>
      )}
      {drift.notes.length > 0 && (
        <details className="text-gray-400 dark:text-gray-500">
          <summary className="cursor-pointer select-none">drift-check notes</summary>
          <ul className="ml-3 list-disc">
            {drift.notes.map(note => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        </details>
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
  onFilterStep,
}: {
  report: CheckpointReport | null;
  loading: boolean;
  onRefresh: () => void;
  onSaveCategories: (categories: Record<string, string>) => Promise<boolean>;
  onOpenSample: (filename: string) => void;
  onFilterStep: (step: number) => void;
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
  const pair = report && effectiveA && effectiveB ? pairInOrientation(report, effectiveA, effectiveB) : null;

  const promptIndices = useMemo(() => (report ? promptIndicesInReport(report) : []), [report]);
  const bestSteps = useMemo(
    () => (report ? highestMedianSteps(report.steps, effectiveModel, effectiveSource) : []),
    [report, effectiveModel, effectiveSource],
  );
  // Drift warnings (Phase 2d) filtered to the selected model/source; the
  // hidden count keeps the rest discoverable instead of silently dropped.
  const driftList = useMemo(
    () => (report ? driftSignals(report, effectiveModel, effectiveSource) : []),
    [report, effectiveModel, effectiveSource],
  );
  const driftHidden = useMemo(
    () => (report ? driftHiddenCount(report, effectiveModel, effectiveSource) : 0),
    [report, effectiveModel, effectiveSource],
  );
  // Retention graph (Phase 2c): weights-kind default follows what the run
  // actually saved (dual-save runs offer EMA/raw; the graph never mixes them).
  const retentionKindOptions = useMemo(() => (report ? retentionWeightsKinds(report) : []), [report]);
  const [weightsKind, setWeightsKind] = useState<RetentionWeightsKind>('ema');
  const effectiveWeights: RetentionWeightsKind = retentionKindOptions.includes(weightsKind)
    ? weightsKind
    : (retentionKindOptions[0] ?? 'ema');
  const categoryOptions = useMemo(() => (report ? retentionCategories(report) : []), [report]);
  const [category, setCategory] = useState<string | null>(null);
  const effectiveCategory = category != null && categoryOptions.includes(category) ? category : null;
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

      {/* how to read this: the intended workflow, spelled out */}
      <details className="text-[10px] text-gray-500 dark:text-gray-400">
        <summary className="cursor-pointer select-none font-medium text-gray-600 dark:text-gray-300">How to read this</summary>
        <ul className="mt-1 list-disc space-y-0.5 pl-4">
          <li>
            <span className="font-medium">▲</span> marks the checkpoint with the highest median similarity — the leader
            by this one number, not a verdict.
          </li>
          <li>
            A checkpoint is only clearly better when the paired comparison below calls a{' '}
            <span className="font-medium">real difference</span> (its 95% confidence interval excludes 0). Otherwise the
            checkpoints are tied — pick between them by looking at the samples.
          </li>
          <li>Every dot in the strips below is one sample — click it to open and judge with your own eyes.</li>
          <li>
            The dashed line is the <span className="font-medium">leave-one-out (LOO)</span> band: how similar your
            reference photos are to each other — a same-person baseline, not a ceiling. Generated faces often land
            near or above it; well below it means weak likeness.
          </li>
          <li>
            <span className="font-medium">T−E gap</span> = training-refs median − external-refs median. A gap that grows
            over steps can mean the model is memorizing the training photos instead of learning the face — external
            refs are the honest test.
          </li>
          <li>
            The <span className="font-medium">drift warnings</span> block below lists statistical warning signs
            (decline, spread growth, memorization signature, pose dependence, near-copies) with links to the samples
            involved. They are suggestions to look, never conclusions — you make the call.
          </li>
        </ul>
      </details>

      {/* drift warnings (Phase 2d): amber, statistical, never a verdict */}
      {report.drift && (
        <DriftWarnings
          report={report}
          drift={report.drift}
          signals={driftList}
          hidden={driftHidden}
          model={effectiveModel}
          source={effectiveSource}
          onOpenSample={onOpenSample}
        />
      )}

      {/* identity retention graph (Phase 2c): two models, two scales, never merged */}
      <div className="space-y-1 rounded border border-gray-300 p-1.5 dark:border-gray-600">
        <div className="flex items-center gap-1 text-xs">
          <span
            className="font-medium"
            title="Median similarity per training step — where likeness peaks as training progresses. ArcFace and SFace are drawn on separate scales and are never compared to each other."
          >
            Identity retention
          </span>
          {categoryOptions.length > 0 && (
            <select
              className={classNames(controlClass, 'ml-auto px-1 py-0.5 text-[10px]')}
              value={effectiveCategory ?? ''}
              onChange={e => setCategory(e.target.value || null)}
              title="Restrict the graph to one prompt category (tags set below)"
            >
              <option value="">all prompts</option>
              {categoryOptions.map(cat => (
                <option key={cat} value={cat}>
                  {cat}
                </option>
              ))}
            </select>
          )}
          {retentionKindOptions.length > 1 && (
            <select
              className={classNames(controlClass, 'px-1 py-0.5 text-[10px]')}
              value={effectiveWeights}
              onChange={e => setWeightsKind(e.target.value as RetentionWeightsKind)}
              title="RAW_ dual-save samples and the un-prefixed (EMA) samples are separate checkpoint series and are never mixed"
            >
              <option value="ema">EMA weights</option>
              <option value="raw">raw weights</option>
            </select>
          )}
        </div>
        <IdentityRetentionGraph
          report={report}
          source={effectiveSource}
          weightsKind={effectiveWeights}
          category={effectiveCategory}
          onFilterStep={onFilterStep}
        />
      </div>

      {/* per-checkpoint table: median (p10) + IQR tooltip; leader highlighted, but no best-checkpoint verdict */}
      <table className="w-full table-fixed text-xs tabular-nums">
        <thead>
          <tr className="text-gray-500 dark:text-gray-400">
            <th className="text-left font-medium">Checkpoint</th>
            <th
              className="font-medium"
              title="Sample counts: ok = one face, scored · amb = ambiguous, multiple faces · nf = no face · fail = error"
            >
              n ok/amb/nf/fail
            </th>
            <th
              className="font-medium"
              title={`Median ${modelName} similarity on comparable prompts — higher = closer to the reference faces. ▲ marks the highest.`}
            >
              {modelName} med
            </th>
            {report.steps.some(s => s.train_vs_external) && (
              <th
                className="font-medium"
                title="Training-refs median − external-refs median. A gap that grows over steps can indicate memorizing the training photos; external refs are the honest test."
              >
                T−E gap
              </th>
            )}
          </tr>
        </thead>
        <tbody>
          {report.steps.map(step => {
            const stats = step.stats?.[effectiveModel]?.[effectiveSource];
            const gap = step.train_vs_external?.[effectiveModel]?.gap;
            const isBest = bestSteps.includes(stepValue(step));
            return (
              <tr
                key={stepValue(step)}
                className={classNames('border-t border-gray-200 dark:border-gray-700', isBest && 'bg-blue-500/[0.07]')}
              >
                <td className="py-0.5 text-left">
                  {isBest && (
                    <span
                      title="highest median on comparable prompts — the leader by this number, not a verdict"
                      className="text-blue-500"
                    >
                      ▲
                    </span>
                  )}{' '}
                  {step.label}
                </td>
                <td className="py-0.5 text-center text-gray-500 dark:text-gray-400">{countsLabel(step.counts)}</td>
                <td
                  className={classNames('py-0.5 text-center', isBest && 'font-semibold text-gray-900 dark:text-gray-50')}
                  title={
                    stats?.iqr != null
                      ? `IQR ${fmtScore(stats.iqr)} (middle half of samples) · p25 ${fmtScore(stats.p25)} · p75 ${fmtScore(stats.p75)} · n ${stats.n}`
                      : undefined
                  }
                >
                  {fmtScore(stats?.median)}
                  {stats?.p10 != null && (
                    <span className="text-gray-400" title="p10 = 10th percentile — 1 in 10 samples scored below this">
                      {' '}
                      p10 {fmtScore(stats.p10)}
                    </span>
                  )}
                </td>
                {report.steps.some(s => s.train_vs_external) && (
                  <td className="py-0.5 text-center text-gray-500 dark:text-gray-400">{gap == null ? '–' : fmtSigned(gap)}</td>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="text-[10px] text-gray-400 dark:text-gray-500">
        ▲ = highest median — strong evidence, not a verdict. Confirm with the paired comparison before calling a
        checkpoint better.
      </p>

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
                  <span
                    title={`median of ${pair.b.label} − ${pair.a.label} on matched (prompt, seed) pairs; the brackets are a bootstrap 95% confidence interval — if it excludes 0 the difference is real, if it straddles 0 it is noise`}
                  >
                    median diff {fmtSigned(comparison.diff_median)}{' '}
                    {comparison.ci_low != null && comparison.ci_high != null && (
                      <span className="text-gray-400">
                        (95% CI {fmtSigned(comparison.ci_low)} … {fmtSigned(comparison.ci_high)})
                      </span>
                    )}{' '}
                  </span>
                  · {pair.b.label} higher on {comparison.wins_b}/{comparison.n_pairs} pairs
                  {comparison.ties > 0 && ` · ${comparison.ties} tied`}
                </div>
                <div
                  className={classNames(
                    comparison.verdict === 'higher' || comparison.verdict === 'lower'
                      ? 'font-medium text-emerald-700 dark:text-emerald-300'
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
              onClick={async () => {
                // keep the draft marked dirty until persistence succeeds, so
                // a failed request cannot clobber the edits via the sync effect
                const ok = await onSaveCategories(draft);
                if (ok) setDirty(false);
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
                {Object.keys(
                  Object.fromEntries(
                    report.steps.flatMap(step =>
                      Object.entries(step.by_category ?? {}).map(([cat]) => [cat, true] as const),
                    ),
                  ),
                ).map(cat => {
                  const cell = (step: CheckpointStepEntry) => step.by_category?.[cat]?.[effectiveModel]?.[effectiveSource];
                  const rowBest = report.steps.reduce((acc, step) => Math.max(acc, cell(step)?.median ?? -Infinity), -Infinity);
                  return (
                    <tr key={cat} className="border-t border-gray-200 dark:border-gray-700">
                      <td className="py-0.5 text-left">{cat}</td>
                      {report.steps.map(step => {
                        const stats = cell(step);
                        const isBest = stats?.median != null && stats.median === rowBest;
                        return (
                          <td key={stepValue(step)} className={classNames('py-0.5 text-center', isBest && 'bg-blue-500/10 font-semibold')}>
                            {isBest && (
                              <span title="highest median in this prompt category" className="text-blue-500">
                                ▲
                              </span>
                            )}{' '}
                            {fmtScore(stats?.median)}
                            <span className="text-gray-400"> ({stats?.n ?? 0})</span>
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
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
