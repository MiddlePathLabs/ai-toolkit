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
  iqr: number | null;
}

export interface CheckpointCounts {
  evaluated: number;
  ambiguous: number;
  no_face: number;
  failed: number;
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
  notes: string[];
}

export type ScoreModel = 'arcface' | 'sface';
export type ScoreSource = IdentitySource | 'combined';

// ------------------------------------------------------------------ phrasing

// The spec's wording rules: "higher identity" only when the paired bootstrap
// CI excludes zero; "no detectable difference" otherwise; never "best".
export function verdictPhrase(comparison: PairModelComparison, model: string, labelA: string, labelB: string): string {
  const name = model === 'arcface' ? 'ArcFace' : model === 'sface' ? 'SFace' : model;
  if (comparison.n_pairs === 0) return `no comparable (prompt, seed) pairs between ${labelA} and ${labelB}`;
  if (comparison.verdict === 'insufficient') {
    return `too few comparable pairs (${comparison.n_pairs}) to judge ${name} identity`;
  }
  if (comparison.verdict === 'higher') {
    return `${labelB} shows higher ${name} identity than ${labelA} on comparable prompts (CI excludes 0)`;
  }
  if (comparison.verdict === 'lower') {
    return `${labelA} shows higher ${name} identity than ${labelB} on comparable prompts (CI excludes 0)`;
  }
  return `no detectable ${name} identity difference between ${labelA} and ${labelB} on comparable prompts`;
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

function roundSigned(value: number): number {
  return Object.is(value, -0) ? 0 : Math.round(value * 10000) / 10000;
}

// ------------------------------------------------------- report freshness

// Pure mtime predicate behind the compare route's cache decision (-1 = file
// absent). The loss log participates because training advances it (the
// convergence panel) without touching results.json; lossWal covers WAL-mode
// SQLite, whose writes land in the -wal sidecar without updating the main
// db file's mtime until a checkpoint.
export function reportFreshByMtimes(parts: {
  results: number;
  categories: number;
  loss: number;
  lossWal: number;
  report: number;
}): boolean {
  if (parts.results < 0 || parts.report < 0) return false;
  return parts.report >= parts.results && parts.report >= parts.categories && parts.report >= parts.loss && parts.report >= parts.lossWal;
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
