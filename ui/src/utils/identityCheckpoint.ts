// Pure logic for checkpoint comparison (Phase 2b): report types shared with
// the Python builder, verdict phrasing, category-tag sanitizing and strip
// helpers. No React, no network -- ui/testing/identity-checkpoint.test.ts
// covers everything here. The report always comes precomputed from the
// compare worker; the UI only renders and phrases.
import type { IdentitySource } from './identityEval';

export interface CheckpointStats {
  n: number;
  median: number | null;
  p10: number | null;
  p25: number | null;
  p75: number | null;
  p90: number | null;
  iqr: number | null;
}

export interface CheckpointCounts {
  evaluated: number;
  ambiguous: number;
  no_face: number;
  failed: number;
  // Phase 2e: ambiguous samples resolved by a subject-face pick (folded
  // into `evaluated`; shown separately so the counts still add up visually)
  picked?: number;
}

export interface CheckpointStepEntry {
  step: number;
  is_raw: boolean;
  label: string;
  counts: CheckpointCounts;
  n_prompts: number;
  stats: Record<string, Partial<Record<IdentitySource | 'combined', CheckpointStats>>>;
  train_vs_external?: Record<string, { training: number | null; external: number | null; gap: number | null }>;
  by_category: Record<string, Record<string, Partial<Record<IdentitySource | 'combined', CheckpointStats>>>>;
  strip: Record<string, Partial<Record<IdentitySource | 'combined', { f: string; s: number }[]>>>;
  // Phase 2e: per-model per-source leakage rate (flagged / evaluable
  // multi-person samples); only steps with evaluable multi-person samples
  leakage?: Record<string, Partial<Record<IdentitySource | 'combined', { flagged: number; evaluable: number; rate: number | null }>>>;
}

export type PairVerdict = 'higher' | 'lower' | 'none' | 'insufficient';

export interface PairModelComparison {
  n_pairs: number;
  diff_median: number | null;
  ci_low: number | null;
  ci_high: number | null;
  wins_b: number;
  wins_a: number;
  ties: number;
  verdict: PairVerdict;
  excluded: { id: string; reason: string; status_a?: string; status_b?: string }[];
}

export interface PairComparison {
  a: { step: number; is_raw: boolean; label: string };
  b: { step: number; is_raw: boolean; label: string };
  models: Record<string, PairModelComparison>;
}

export interface ConvergenceSummary {
  last_step: number;
  loss_buckets_250: number[];
  late_500_change_pct: number | null;
  lr_start: number;
  lr_end: number;
  lr_floored: boolean;
  bucket_noise_pct: number;
  converged: boolean;
  source: string;
}

export type DriftSignalKind =
  | 'similarity_decline'
  | 'variance_increase'
  | 'train_external_divergence'
  | 'pose_weakness'
  | 'narrow_generalization'
  | 'near_copy';

// One Phase 2d drift warning. Always per model (never a blend), always
// statistical ("possible drift" wording lives in the message text), always
// with the numbers behind it and sample filenames for full-res review.
export interface DriftSignal {
  kind: DriftSignalKind;
  model: string;
  // null for cross-source signals (training-vs-external divergence)
  source: string | null;
  scope: { step: number; is_raw: boolean } | null;
  message: string;
  evidence: Record<string, unknown>;
  samples: string[];
}

export interface DriftReport {
  version: number;
  signals: DriftSignal[];
  notes: string[];
}

// ------------------------------------------------------- leakage (Phase 2e)

// One flagged non-target face of a multi-person sample. Levels are per
// model per source (never a blend); user_twin comes exclusively from the
// user's own mark -- the tool never asserts it.
export interface LeakageFace {
  filename: string;
  sample_path?: string | null;
  step: number | null;
  is_raw: boolean;
  subject_face: number;
  face_index: number | null;
  det_score: number | null;
  user_twin: boolean;
  levels: Record<string, Partial<Record<string, 'resembles' | 'likely_duplicate'>>>;
  scores: Record<string, Partial<Record<string, { mean: number | null; max: number | null }>>>;
  closest_ref?: Record<string, Partial<Record<string, { key: string | null; face_index: number | null; sim: number }>>> | null;
}

export interface LeakageStepRates {
  step: number;
  is_raw: boolean;
  rates: Record<string, Partial<Record<string, { flagged: number; evaluable: number; rate: number | null }>>>;
}

export interface LeakageReport {
  version: number;
  margin: number;
  thresholds: Record<string, Record<string, { loo: number; resembles_min: number }>>;
  picks_used: number;
  counts: { multi_face_samples: number; evaluable: number; unpicked_multi: number };
  faces: LeakageFace[];
  per_step: LeakageStepRates[];
  notes: string[];
}

export interface CheckpointReport {
  version: number;
  job: string;
  generated_at: string;
  results_completed: boolean;
  results_updated_at?: string;
  sources: (IdentitySource | 'combined')[];
  models: string[];
  refsets: Partial<Record<IdentitySource | 'combined', { n: number; loo: Record<string, number | null> } | null>>;
  prompt_texts: Record<string, string>;
  categories: Record<string, string>;
  steps: CheckpointStepEntry[];
  pairs: PairComparison[];
  convergence: ConvergenceSummary | null;
  drift?: DriftReport | null;
  leakage?: LeakageReport | null;
  notes: string[];
}

export type ScoreModel = 'arcface' | 'sface';
export type ScoreSource = IdentitySource | 'combined';

// ------------------------------------------------------------------ phrasing

// The spec's wording rules: a checkpoint is only called "better" when the
// paired bootstrap CI excludes zero; "tied" otherwise; never "best". The
// phrasing spells out what the CI means so the takeaway needs no stats
// background.
export function verdictPhrase(comparison: PairModelComparison, model: string, labelA: string, labelB: string): string {
  const name = model === 'arcface' ? 'ArcFace' : model === 'sface' ? 'SFace' : model;
  if (comparison.n_pairs === 0) return `no comparable (prompt, seed) pairs between ${labelA} and ${labelB}`;
  if (comparison.verdict === 'insufficient') {
    return `too few comparable pairs (${comparison.n_pairs}) to judge ${name} identity`;
  }
  if (comparison.verdict === 'higher') {
    return `${labelB} matches the reference faces better than ${labelA} on comparable prompts — a real difference (the 95% confidence interval excludes 0)`;
  }
  if (comparison.verdict === 'lower') {
    return `${labelA} matches the reference faces better than ${labelB} on comparable prompts — a real difference (the 95% confidence interval excludes 0)`;
  }
  return `no trustworthy ${name} difference between ${labelA} and ${labelB} on comparable prompts — treat them as tied`;
}

export function excludedReasonLabel(reason: string): string {
  switch (reason) {
    case 'missing_at_a':
      return 'not sampled at the earlier checkpoint';
    case 'missing_at_b':
      return 'not sampled at the later checkpoint';
    case 'not_evaluated':
      return 'not evaluable at both checkpoints (no face / ambiguous / failed)';
    default:
      return reason;
  }
}

export function countsLabel(counts: CheckpointCounts): string {
  return `${counts.evaluated}/${counts.ambiguous}/${counts.no_face}/${counts.failed}`;
}

export function fmtScore(value: number | null | undefined): string {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(3) : '–';
}

export function fmtSigned(value: number | null | undefined): string {
  return typeof value === 'number' && Number.isFinite(value) ? (value >= 0 ? '+' : '') + value.toFixed(3) : '–';
}

export function convergencePhrase(convergence: ConvergenceSummary): string {
  if (convergence.converged) {
    return 'likely converged — more training steps are unlikely to reduce loss further';
  }
  return 'possibly still gaining — more steps may help; check the loss curve';
}

// --------------------------------------------------------------- categories

// Prompt-index keys, trimmed and length-capped category strings; empty
// values drop the tag. Mirrors the compare route's sanitizer so optimistic
// UI state matches what the server persists.
export function sanitizeCategories(input: Record<string, string> | null | undefined): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [key, value] of Object.entries(input ?? {})) {
    if (!/^\d{1,4}$/.test(key)) continue;
    if (typeof value !== 'string') continue;
    const clean = value.trim().slice(0, 40);
    if (clean.length > 0) out[key] = clean;
  }
  return out;
}

export const CATEGORY_PRESETS = ['solo', 'multi', 'gate-no-trigger', 'profile'] as const;

export function promptIndicesInReport(report: CheckpointReport): number[] {
  const indices = new Set<number>();
  for (const key of Object.keys(report.prompt_texts ?? {})) {
    const idx = Number(key);
    if (Number.isInteger(idx) && idx >= 0) indices.add(idx);
  }
  for (const pair of report.pairs ?? []) {
    for (const comparison of Object.values(pair.models ?? {})) {
      for (const excluded of comparison.excluded ?? []) {
        const match = excluded.id.match(/prompt (\d+)/);
        if (match) indices.add(Number(match[1]));
      }
    }
  }
  // prompts observed in strips (the evaluated ones categories actually split)
  const stripIdx = new Set<number>();
  for (const step of report.steps ?? []) {
    for (const modelData of Object.values(step.strip ?? {})) {
      for (const entries of Object.values(modelData ?? {})) {
        for (const entry of entries ?? []) {
          // Python formats the prompt index with :02d (minimum width), so
          // prompt 100+ encodes as _100_seed-; match variable width.
          const match = entry.f.match(/_(\d{2,})_seed-/);
          if (match) stripIdx.add(Number(match[1]));
        }
      }
    }
  }
  for (const idx of stripIdx) indices.add(idx);
  return [...indices].sort((a, b) => a - b);
}

// ------------------------------------------------------------------ pairing

export function stepValue(step: { step: number; is_raw: boolean }): string {
  return `${step.step}:${step.is_raw ? 1 : 0}`;
}

export function findPair(report: CheckpointReport, aValue: string, bValue: string): PairComparison | null {
  return report.pairs.find(p => stepValue(p.a) === aValue && stepValue(p.b) === bValue) ?? null;
}

// The report builder emits each checkpoint combination once, in ascending
// order. The dropdowns allow either selection order, so a reversed selection
// is served by inverting the stored comparison: swap endpoints, negate the
// differences (CI bounds swap AND negate), swap win counts and flip the
// higher/lower verdict. Orientation-neutral fields (excluded ids) are kept,
// per-side statuses swapped.
export function invertPairComparison(pair: PairComparison): PairComparison {
  const models: PairComparison['models'] = {};
  for (const [key, value] of Object.entries(pair.models ?? {})) {
    models[key] = {
      ...value,
      diff_median: value.diff_median == null ? null : roundSigned(-value.diff_median),
      ci_low: value.ci_high == null ? null : roundSigned(-value.ci_high),
      ci_high: value.ci_low == null ? null : roundSigned(-value.ci_low),
      wins_a: value.wins_b,
      wins_b: value.wins_a,
      verdict: value.verdict === 'higher' ? 'lower' : value.verdict === 'lower' ? 'higher' : value.verdict,
      excluded: (value.excluded ?? []).map(entry => ({
        ...entry,
        status_a: entry.status_b,
        status_b: entry.status_a,
      })),
    };
  }
  return { a: pair.b, b: pair.a, models };
}

// Orientation-aware lookup: exact match first, else the stored reversed pair
// inverted to the requested direction.
export function pairInOrientation(report: CheckpointReport, aValue: string, bValue: string): PairComparison | null {
  const exact = findPair(report, aValue, bValue);
  if (exact) return exact;
  const reversed = findPair(report, bValue, aValue);
  return reversed ? invertPairComparison(reversed) : null;
}

// Step keys holding the highest median for the given model/source (exact ties
// kept). This is what the checkpoint table highlights: the leader by that one
// number — evidence to look at, never a best-checkpoint verdict.
export function highestMedianSteps(steps: CheckpointStepEntry[], model: ScoreModel, source: ScoreSource): string[] {
  let best = -Infinity;
  for (const step of steps) {
    const median = step.stats?.[model]?.[source]?.median;
    if (median != null && median > best) best = median;
  }
  if (!Number.isFinite(best)) return [];
  return steps.filter(step => step.stats?.[model]?.[source]?.median === best).map(stepValue);
}

function roundSigned(value: number): number {
  return Object.is(value, -0) ? 0 : Math.round(value * 10000) / 10000;
}

// ------------------------------------------------------- report freshness

// Pure mtime predicate behind the compare route's cache decision (-1 = file
// absent). The loss log participates because training advances it (the
// convergence panel) without touching results.json; lossWal covers WAL-mode
// SQLite, whose writes land in the -wal sidecar without updating the main
// db file's mtime until a checkpoint. selection.json participates since
// Phase 2e: subject-face picks/twin marks change the report (pick resolution
// + leakage rates) without touching results.json.
export function reportFreshByMtimes(parts: {
  results: number;
  categories: number;
  picks: number;
  loss: number;
  lossWal: number;
  report: number;
}): boolean {
  if (parts.results < 0 || parts.report < 0) return false;
  return (
    parts.report >= parts.results &&
    parts.report >= parts.categories &&
    parts.report >= parts.picks &&
    parts.report >= parts.loss &&
    parts.report >= parts.lossWal
  );
}

// --------------------------------------------------------- leakage labels

// "2/5" cell for the checkpoint table's leak column (flagged/evaluable
// multi-person samples for the selected model + source).
export function leakageRateLabel(entry: { flagged: number; evaluable: number; rate: number | null } | undefined): string {
  if (!entry) return '–';
  return `${entry.flagged}/${entry.evaluable}`;
}

export function leakLevelLabel(level: 'resembles' | 'likely_duplicate'): string {
  return level === 'likely_duplicate' ? 'possible duplicate' : 'resembles subject';
}

// ---------------------------------------------------------------- stripping

export function stripDomain(
  steps: CheckpointStepEntry[],
  model: ScoreModel,
  source: ScoreSource,
): { min: number; max: number } {
  let min = Infinity;
  let max = -Infinity;
  for (const step of steps) {
    for (const entry of step.strip?.[model]?.[source] ?? []) {
      min = Math.min(min, entry.s);
      max = Math.max(max, entry.s);
    }
  }
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) {
    const fallback = Number.isFinite(min) ? min : 0;
    return { min: fallback - 0.02, max: fallback + 0.02 };
  }
  const pad = (max - min) * 0.08;
  return { min: min - pad, max: max + pad };
}

export function stripX(score: number, domain: { min: number; max: number }, width: number): number {
  const t = (score - domain.min) / (domain.max - domain.min || 1);
  return 4 + t * (width - 8);
}

// --------------------------------------------------------- retention graph (2c)

// Chart data for the identity-retention graph: median similarity per training
// step, one series per model on its own scale (ArcFace and SFace are never
// merged onto a shared axis), with a p10-p90 band and per-point sample counts.
// Everything here derives from the precomputed checkpoint report; no network,
// no React -- ui/testing/identity-checkpoint.test.ts covers it.

export type RetentionWeightsKind = 'ema' | 'raw';

// Steps with fewer than this many evaluated samples draw as hollow points:
// too little data for the median to mean much.
export const RETENTION_SPARSE_N = 5;

export interface RetentionModelSeries {
  model: ScoreModel;
  xs: number[];
  median: (number | null)[];
  p10: (number | null)[];
  p90: (number | null)[];
  n: number[];
  // true where a median point exists but n < RETENTION_SPARSE_N (hollow)
  sparse: boolean[];
  loo: number | null;
}

// Which weights kinds actually carry steps (dual-save runs have both RAW_ and
// EMA groups; plain runs only EMA). The graph never mixes them in one line.
export function retentionWeightsKinds(report: CheckpointReport): RetentionWeightsKind[] {
  const kinds: RetentionWeightsKind[] = [];
  if (report.steps.some(s => !s.is_raw)) kinds.push('ema');
  if (report.steps.some(s => s.is_raw)) kinds.push('raw');
  return kinds;
}

// Category values that actually appear in the step breakdowns (union of
// by_category keys), sorted; the UI prepends an "all" option.
export function retentionCategories(report: CheckpointReport): string[] {
  const set = new Set<string>();
  for (const step of report.steps) {
    for (const cat of Object.keys(step.by_category ?? {})) set.add(cat);
  }
  return [...set].sort();
}

export function buildRetentionSeries(
  report: CheckpointReport,
  source: ScoreSource,
  weightsKind: RetentionWeightsKind,
  category: string | null,
): RetentionModelSeries[] {
  const models = (['arcface', 'sface'] as const).filter(model => report.models.includes(model));
  const statsOf = (step: CheckpointStepEntry, model: ScoreModel): CheckpointStats | null =>
    category == null
      ? step.stats?.[model]?.[source] ?? null
      : step.by_category?.[category]?.[model]?.[source] ?? null;
  // Steps of the selected weights kind where at least one model carries data
  // for the filter -- one shared x list keeps every model's arrays aligned
  // (uPlot requires it), and category views stay compact instead of padding
  // gaps for steps that never had that category.
  const steps = report.steps.filter(
    step => step.is_raw === (weightsKind === 'raw') && (category == null || models.some(model => statsOf(step, model) != null)),
  );
  return models.map(model => {
    const xs: number[] = [];
    const median: (number | null)[] = [];
    const p10: (number | null)[] = [];
    const p90: (number | null)[] = [];
    const n: number[] = [];
    const sparse: boolean[] = [];
    for (const step of steps) {
      xs.push(step.step);
      const stats = statsOf(step, model);
      median.push(stats?.median ?? null);
      p10.push(stats?.p10 ?? null);
      p90.push(stats?.p90 ?? null);
      n.push(stats?.n ?? 0);
      sparse.push(stats?.median != null && (stats?.n ?? 0) < RETENTION_SPARSE_N);
    }
    return { model, xs, median, p10, p90, n, sparse, loo: report.refsets?.[source]?.loo?.[model] ?? null };
  });
}

// --------------------------------------------------------- drift warnings (2d)

// The checkpoint section shows one model/source at a time (the header
// toggle), so the warnings block filters to that selection. Cross-source
// signals (training-vs-external divergence carries source: null) are always
// shown; the hidden count keeps the other model/source's warnings discoverable
// instead of silently dropped. Per-model presentation is a spec rule: ArcFace
// and SFace warnings are never merged.
export function driftSignals(report: CheckpointReport, model: string, source: string): DriftSignal[] {
  return (report.drift?.signals ?? []).filter(
    signal => signal.model === model && (signal.source === null || signal.source === source),
  );
}

export function driftHiddenCount(report: CheckpointReport, model: string, source: string): number {
  return (report.drift?.signals ?? []).length - driftSignals(report, model, source).length;
}

export function driftKindLabel(kind: DriftSignalKind | string): string {
  switch (kind) {
    case 'similarity_decline':
      return 'identity decline';
    case 'variance_increase':
      return 'score spread growth';
    case 'train_external_divergence':
      return 'training vs external divergence';
    case 'pose_weakness':
      return 'pose-dependent scoring';
    case 'narrow_generalization':
      return 'narrow generalization';
    case 'near_copy':
      return 'possible near-copies';
    default:
      return kind;
  }
}
