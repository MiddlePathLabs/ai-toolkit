// Pure logic for the identity-eval feature: result indexing, per-step
// summaries, sorting and the rank-based model-disagreement flag. No React,
// no network -- ui/testing/identity-eval.test.ts covers everything here.
import { parseSampleFilename } from './sampleImages';

export type IdentitySource = 'training' | 'external';
export type IdentitySampleStatus = 'evaluated' | 'ambiguous' | 'failed';

export interface IdentityModelScores {
  mean: number | null;
  max: number | null;
}

// scores[model][source]
export type IdentityFaceScores = Record<string, Partial<Record<IdentitySource | 'combined', IdentityModelScores>>>;

export interface IdentityFace {
  index: number;
  bbox: number[];
  // 5-point landmarks (results v2+); absent in older documents -- the Phase
  // 2d pose signal skips samples without them
  kps?: number[][] | null;
  det_score: number;
  scores: IdentityFaceScores;
  // Single nearest reference per model per source (results v3+; absent in
  // older documents -- the Phase 2e inspection view then hides the crop)
  closest_ref?: Record<string, Partial<Record<IdentitySource | 'combined', { key: string | null; face_index: number | null; sim: number }>>>;
}

export interface IdentitySampleResult {
  path: string;
  filename: string;
  step: number | null;
  prompt_idx: number | null;
  seed: number | null;
  is_raw: boolean;
  status: IdentitySampleStatus;
  face_count: number;
  error: string | null;
  primary: IdentityFaceScores | null;
  faces: IdentityFace[];
  warnings: string[];
  // Phase 2e pick-resolution markers (client-side copies only; the worker
  // never writes them)
  picked_subject?: boolean;
  subject_face_index?: number;
}

export interface IdentityRefset {
  n: number;
  hash: string;
  skipped?: Record<string, number>;
  loo: Record<string, number | null>;
}

export interface IdentityResults {
  version: number;
  job: string;
  completed: boolean;
  created_at: string;
  updated_at: string;
  sources: IdentitySource[];
  device?: { requested?: string; resolved?: string; gpu?: number; warnings?: string[] };
  models?: Record<string, { active?: boolean; available?: boolean; reason?: string | null }>;
  refsets: Partial<Record<IdentitySource | 'combined', IdentityRefset | null>>;
  samples: IdentitySampleResult[];
  warnings?: string[];
}

export interface IdentityRefImage {
  path: string;
  name: string;
  sha: string;
  status: 'ok' | 'needs_review' | 'no_face' | 'error';
  error: string | null;
  faces: { index: number; bbox: number[]; kps: number[][]; det_score: number }[];
}

export interface IdentityRefs {
  version: number;
  job: string;
  generated_at: string;
  models: Record<string, unknown>;
  training: { datasets?: unknown[]; images: IdentityRefImage[]; warnings?: string[] };
  external: { images: IdentityRefImage[] };
  warnings?: string[];
}

export interface IdentitySamplePick {
  face_index?: number | null;
  twin_confirmed?: number[];
}

export interface IdentitySelection {
  version: number;
  training_refs: Record<string, { excluded?: boolean; face_index?: number | null }>;
  external_refs: Record<string, { excluded?: boolean; face_index?: number | null }>;
  // Phase 2e: keyed by generated-sample filename
  sample_picks?: Record<string, IdentitySamplePick>;
}

export type IdentitySortMode = 'off' | 'arcface' | 'sface';

// ------------------------------------------------------------------- indexing

export function identityFilenameKey(path: string): string {
  const parts = path.replace(/\\/g, '/').split('/');
  return parts[parts.length - 1] ?? path;
}

export function buildIdentityIndex(results: { samples?: IdentitySampleResult[] } | null): Map<string, IdentitySampleResult> {
  const index = new Map<string, IdentitySampleResult>();
  if (!results?.samples) return index;
  for (const sample of results.samples) {
    index.set(sample.filename || identityFilenameKey(sample.path), sample);
  }
  return index;
}

export interface IdentityOverlayInfo {
  status: IdentitySampleStatus;
  label: string;
  tooltip: string;
  arcfaceMean: number | null;
  sfaceMean: number | null;
}

export function overlayInfo(
  sample: IdentitySampleResult | undefined,
  source: IdentitySource | 'combined',
): IdentityOverlayInfo | null {
  if (!sample) return null;
  const arcfaceMean = sample.primary?.arcface?.[source]?.mean ?? null;
  const sfaceMean = sample.primary?.sface?.[source]?.mean ?? null;
  const parts: string[] = [];
  if (arcfaceMean !== null) parts.push(`A ${arcfaceMean.toFixed(3)}`);
  if (sfaceMean !== null) parts.push(`S ${sfaceMean.toFixed(3)}`);
  const label = parts.join(' · ');
  const tooltipBits = [
    'A = ArcFace, S = SFace — face-recognition similarity to the reference faces; 1.0 = identical. The two models use different scales and are never compared to each other.',
    sample.warnings?.length ? sample.warnings.join(' ') : '',
  ].filter(Boolean);
  return {
    status: sample.status,
    label: label || (sample.status === 'ambiguous' ? 'ambiguous' : sample.status === 'failed' ? 'failed' : 'pending'),
    tooltip: tooltipBits.join(' '),
    arcfaceMean,
    sfaceMean,
  };
}

// ------------------------------------------------------------------ summaries

function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

export interface IdentityStepSummary {
  step: number;
  evaluated: number;
  failed: number;
  ambiguous: number;
  medianArcface: number | null;
  medianSface: number | null;
  // Model the highest/lowest lists are ranked by: ArcFace when it scored
  // anything for this source, else SFace (SFace-only mode), else null.
  rankModel: 'arcface' | 'sface' | null;
  lowest: IdentitySampleResult[];
  highest: IdentitySampleResult[];
}

export function groupSummariesByStep(
  samples: IdentitySampleResult[],
  source: IdentitySource | 'combined',
  lowestCount = 3,
): IdentityStepSummary[] {
  const score = (s: IdentitySampleResult, model: string) => s.primary?.[model]?.[source]?.mean ?? null;
  const model = samples.some(s => score(s, 'arcface') !== null)
    ? 'arcface'
    : samples.some(s => score(s, 'sface') !== null)
      ? 'sface'
      : null;
  const byStep = new Map<number, IdentitySampleResult[]>();
  for (const sample of samples) {
    if (sample.step === null || sample.step === undefined) continue;
    const bucket = byStep.get(sample.step) ?? [];
    bucket.push(sample);
    byStep.set(sample.step, bucket);
  }
  return [...byStep.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([step, bucket]) => {
      const evaluated = bucket.filter(s => s.status === 'evaluated');
      const arcfaceScores = evaluated.map(s => score(s, 'arcface')).filter((v): v is number => v !== null);
      const sfaceScores = evaluated.map(s => score(s, 'sface')).filter((v): v is number => v !== null);
      // Only samples the ranking model actually scored compete for the
      // highest/lowest slots -- a missing score must never surface as 0.000.
      let lowest: IdentitySampleResult[] = [];
      let highest: IdentitySampleResult[] = [];
      if (model !== null) {
        const ranked = evaluated.filter(s => score(s, model) !== null);
        const byModel = (a: IdentitySampleResult, b: IdentitySampleResult) =>
          (score(a, model) as number) - (score(b, model) as number);
        lowest = [...ranked].sort(byModel).slice(0, lowestCount);
        highest = [...ranked].sort((a, b) => byModel(b, a)).slice(0, lowestCount);
      }
      return {
        step,
        evaluated: evaluated.length,
        failed: bucket.filter(s => s.status === 'failed').length,
        ambiguous: bucket.filter(s => s.status === 'ambiguous').length,
        medianArcface: median(arcfaceScores),
        medianSface: median(sfaceScores),
        rankModel: model,
        lowest,
        highest,
      };
    });
}

// --------------------------------------------------------------- rank disagreement

// A sample is flagged when its rank under ArcFace vs SFace differs by more
// than a third of the evaluated set. Rank-based, so no cross-model cosine
// threshold is implied.
export function rankDisagreements(
  samples: IdentitySampleResult[],
  source: IdentitySource | 'combined',
): Map<string, boolean> {
  const evaluated = samples.filter(
    s => s.status === 'evaluated' &&
      typeof s.primary?.arcface?.[source]?.mean === 'number' &&
      typeof s.primary?.sface?.[source]?.mean === 'number',
  );
  const flags = new Map<string, boolean>();
  const n = evaluated.length;
  if (n < 3) {
    for (const s of evaluated) flags.set(s.filename, false);
    return flags;
  }
  const rankOf = (model: string) => {
    const sorted = [...evaluated].sort(
      (a, b) => (b.primary![model]![source]!.mean as number) - (a.primary![model]![source]!.mean as number),
    );
    const ranks = new Map<string, number>();
    sorted.forEach((s, i) => ranks.set(s.filename, i));
    return ranks;
  };
  const rankA = rankOf('arcface');
  const rankS = rankOf('sface');
  const threshold = n / 3;
  for (const s of evaluated) {
    flags.set(s.filename, Math.abs((rankA.get(s.filename) ?? 0) - (rankS.get(s.filename) ?? 0)) > threshold);
  }
  return flags;
}

// ---------------------------------------------------------------------- sorting

export function sampleScore(sample: IdentitySampleResult | undefined, model: 'arcface' | 'sface', source: IdentitySource | 'combined'): number | null {
  if (!sample || sample.status !== 'evaluated') return null;
  return sample.primary?.[model]?.[source]?.mean ?? null;
}

export function sortPathsByScore(
  paths: string[],
  index: Map<string, IdentitySampleResult>,
  model: 'arcface' | 'sface',
  source: IdentitySource | 'combined',
  direction: 'best-first' | 'worst-first' = 'worst-first',
): string[] {
  const decorated = paths.map(path => ({ path, score: sampleScore(index.get(identityFilenameKey(path)), model, source) }));
  decorated.sort((a, b) => {
    if (a.score === null && b.score === null) return a.path.localeCompare(b.path);
    if (a.score === null) return 1; // unscored last
    if (b.score === null) return -1;
    return direction === 'worst-first' ? a.score - b.score : b.score - a.score;
  });
  return decorated.map(d => d.path);
}

// ------------------------------------------------------------------- selection

export function mergeSelection(
  current: IdentitySelection,
  update: {
    trainingRefs?: IdentitySelection['training_refs'];
    externalRefs?: IdentitySelection['external_refs'];
    samplePicks?: IdentitySelection['sample_picks'];
  },
): IdentitySelection {
  return {
    version: 1,
    training_refs: { ...current.training_refs, ...sanitizeEntries(update.trainingRefs) },
    external_refs: { ...current.external_refs, ...sanitizeEntries(update.externalRefs) },
    sample_picks: { ...(current.sample_picks ?? {}), ...sanitizeSamplePicks(update.samplePicks) },
  };
}

// Mirrors the refs route's samplePicks sanitizer so optimistic UI state
// matches what the server persists (bare-filename keys only).
export function sanitizeSamplePicks(entries: IdentitySelection['sample_picks'] | undefined): IdentitySelection['sample_picks'] {
  const out: NonNullable<IdentitySelection['sample_picks']> = {};
  for (const [key, value] of Object.entries(entries ?? {})) {
    if (!/^[^\\/:*?"<>|]{1,200}$/.test(key)) continue;
    if (!value || typeof value !== 'object') continue;
    const clean: IdentitySamplePick = {};
    if (value.face_index === null || (Number.isInteger(value.face_index) && (value.face_index as number) >= 0 && (value.face_index as number) <= 32)) {
      clean.face_index = value.face_index;
    }
    if (Array.isArray(value.twin_confirmed)) {
      const marks = [...new Set(value.twin_confirmed.filter((n): n is number => Number.isInteger(n) && n >= 0 && n <= 32))].sort((a, b) => a - b);
      if (marks.length > 0) clean.twin_confirmed = marks;
    }
    if (Object.keys(clean).length > 0) out[key] = clean;
  }
  return out;
}

function sanitizeEntries(entries: IdentitySelection['training_refs'] | undefined): IdentitySelection['training_refs'] {
  const out: IdentitySelection['training_refs'] = {};
  for (const [key, value] of Object.entries(entries ?? {})) {
    if (!value || typeof value !== 'object') continue;
    const clean: { excluded?: boolean; face_index?: number | null } = {};
    if (typeof value.excluded === 'boolean') clean.excluded = value.excluded;
    if (value.face_index === null || (Number.isInteger(value.face_index) && (value.face_index as number) >= 0 && (value.face_index as number) <= 32)) {
      clean.face_index = value.face_index;
    }
    if (Object.keys(clean).length > 0) out[key] = clean;
  }
  return out;
}

export function selectionForImage(
  selection: IdentitySelection | null,
  image: { path: string; name: string },
  source: IdentitySource,
): { excluded?: boolean; face_index?: number | null } {
  if (!selection) return {};
  const map = source === 'external' ? selection.external_refs : selection.training_refs;
  const key = source === 'external' ? image.name : normalizeRefPath(image.path);
  return map[key] ?? {};
}

// Training-ref keys are the canonical selection-key contract shared with the
// worker's canonical_ref_key (normcase + normpath + forward slashes): the
// panel writes, this reader and the Python lookup all use this exact form.
export function normalizeRefPath(path: string): string {
  const forward = path.replace(/\\/g, '/').toLowerCase();
  // collapse duplicate slashes; strip trailing slash (posix normpath behaviour)
  const collapsed = forward.replace(/\/+/g, '/');
  return collapsed.length > 1 && collapsed.endsWith('/') ? collapsed.slice(0, -1) : collapsed;
}

// Parse step/prompt for identity bookkeeping (worker already stores them;
// this is the client fallback for results produced before a field existed).
export function identityStepPrompt(path: string): { step: number | null; promptIdx: number | null } {
  const parsed = parseSampleFilename(path);
  return { step: parsed?.trainingStep ?? null, promptIdx: parsed?.sampleIndex ?? null };
}

// ------------------------------------------------- multi-person leakage (2e)

export type FaceLeakLevel = 'resembles' | 'likely_duplicate';

// Mirror of toolkit.identity_eval.leakage.RESEMBLES_MARGIN. Applied to each
// model's OWN LOO band (never a shared scale); provisional pending 2a/11.7.
// identity-eval.test.ts pins this value against the Python constant.
export const LEAKAGE_RESEMBLES_MARGIN = 0.10;

// Mirror of leakage.classify_level: >= loo -> likely duplicate, within
// LEAKAGE_RESEMBLES_MARGIN below -> resembles (informational), else null.
// Per model per source -- callers pass that model's own loo and mean.
export function classifyFaceLevel(mean: number | null | undefined, loo: number | null | undefined): FaceLeakLevel | null {
  if (mean == null || loo == null || !Number.isFinite(mean) || !Number.isFinite(loo)) return null;
  if (mean >= loo) return 'likely_duplicate';
  if (mean >= loo - LEAKAGE_RESEMBLES_MARGIN) return 'resembles';
  return null;
}

// The subject-face pick of a sample, when it resolves a subject: only
// ambiguous (multi-face) samples honour a pick, and the index must point
// at a real face. Returns null for everything else (mirrors valid_pick).
export function samplePickFor(sample: IdentitySampleResult, picks: Record<string, IdentitySamplePick> | null | undefined): IdentitySamplePick | null {
  if (sample.status !== 'ambiguous' || !picks) return null;
  const pick = picks[sample.filename];
  if (!pick) return null;
  const index = pick.face_index;
  if (!Number.isInteger(index) || (index as number) < 0 || (index as number) >= sample.faces.length) return null;
  return pick;
}

// Apply subject-face picks: a picked ambiguous sample becomes evaluated
// with primary = the picked face's already-persisted scores (Phase 1 scored
// every face; no re-run needed). Pure/immutable -- mirrors the Python
// resolve_picks used by the checkpoint report so both sides agree.
export function resolveSamplePicks(
  samples: IdentitySampleResult[],
  picks: Record<string, IdentitySamplePick> | null | undefined,
): IdentitySampleResult[] {
  return samples.map(sample => {
    const pick = samplePickFor(sample, picks);
    if (!pick) return sample;
    const index = pick.face_index as number;
    return {
      ...sample,
      status: 'evaluated',
      primary: sample.faces[index].scores ?? null,
      picked_subject: true,
      subject_face_index: index,
    };
  });
}

export function twinMarked(pick: IdentitySamplePick | null, faceIndex: number): boolean {
  return Array.isArray(pick?.twin_confirmed) && (pick?.twin_confirmed as number[]).includes(faceIndex);
}
