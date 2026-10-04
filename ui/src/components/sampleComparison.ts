import type { SampleConfig } from '../types';
import { getSampleItems, getSampleMetadata, parseSampleFilename } from '../utils/sampleImages';
import type { SampleMetadata, SampleRow } from '../utils/sampleImages';

type ComparisonRow = Pick<SampleRow, 'key' | 'paths'>;

export interface ComparisonColumn {
  index: number;
  path: string | null;
  prompt: string | null;
  isRaw: boolean | null;
  metadata: SampleMetadata;
}

export function getComparisonColumns(row: ComparisonRow, config: SampleConfig | null): ComparisonColumn[] {
  const items = getSampleItems(config);
  return Array.from({ length: Math.max(row.paths.length, items.length) }, (_, index) => {
    const path = row.paths[index] ?? null;
    const info = path ? parseSampleFilename(path) : null;
    const item = items[index];
    return {
      index,
      path,
      prompt: (!path || info) && item ? item.prompt.split('--')[0].trim() : null,
      isRaw: info?.isRaw ?? (item ? Boolean(item.raw_weights) : null),
      metadata: getSampleMetadata(path ?? '', config, index),
    };
  });
}

function pairScore(left: ComparisonColumn, right: ComparisonColumn): number {
  const samePrompt = left.prompt !== null && left.prompt === right.prompt;
  const sameSeed = left.metadata.seed !== null && left.metadata.seed === right.metadata.seed;
  return (
    Number(Boolean(left.path)) * 100 + Number(Boolean(right.path)) * 100 + Number(samePrompt) * 10 + Number(sameSeed)
  );
}

export function getDefaultComparisonColumns(columns: ComparisonColumn[]): [number, number] {
  const available = columns.filter(column => column.path);
  const candidates = available.length >= 2 ? available : columns;
  const rawColumns = candidates.filter(column => column.isRaw === true);
  const normalColumns = candidates.filter(column => column.isRaw === false);
  let pair: [number, number] | null = null;
  let score = -1;
  for (const raw of rawColumns) {
    for (const normal of normalColumns) {
      const candidateScore = pairScore(raw, normal);
      if (candidateScore <= score) continue;
      pair = [raw.index, normal.index];
      score = candidateScore;
    }
  }
  if (pair) return pair;
  for (let leftIndex = 0; leftIndex < available.length; leftIndex++) {
    for (let rightIndex = leftIndex + 1; rightIndex < available.length; rightIndex++) {
      const candidateScore = pairScore(available[leftIndex], available[rightIndex]);
      if (candidateScore <= score) continue;
      pair = [available[leftIndex].index, available[rightIndex].index];
      score = candidateScore;
    }
  }
  if (pair) return pair;
  const left = available[0]?.index ?? columns[0]?.index ?? -1;
  const right = available[1]?.index ?? columns.find(column => column.index !== left)?.index ?? -1;
  return [left, right];
}

export function getComparisonNotice(left?: ComparisonColumn, right?: ComparisonColumn): string | null {
  if (!left?.path || !right?.path) return 'One or both samples are not available for this training row.';
  if (left.path === right.path) return 'The same sample is selected in both panes.';
  const notices: string[] = [];
  if (left.prompt === null || right.prompt === null) notices.push('Prompt match cannot be verified.');
  else if (left.prompt !== right.prompt) notices.push('Prompts differ.');
  if (left.metadata.seed === null || right.metadata.seed === null) notices.push('Seed match cannot be verified.');
  else if (left.metadata.seed !== right.metadata.seed) notices.push('Seeds differ.');
  return notices.length ? `${notices.join(' ')} This is not a matched prompt-and-seed comparison.` : null;
}

export function getComparisonWeightLabel(column: ComparisonColumn, hasEma: boolean): string {
  if (column.isRaw === null) return 'Weight type unknown';
  return column.isRaw ? 'Raw weights' : hasEma ? 'EMA weights' : 'Non-raw weights';
}

export function getComparisonRowLabels(rows: ComparisonRow[]): string[] {
  const counts = new Map<string, number>();
  return rows.map((row, index) => {
    const info = row.paths.map(path => (path ? parseSampleFilename(path) : null)).find(Boolean);
    const identity = /^(\d+):(\d+)$/.exec(row.key);
    const step = info?.trainingStep ?? (identity ? Number(identity[1]) : null);
    const timestamp = info?.timestamp ?? (identity ? Number(identity[2]) : null);
    if (step === null) return `Row ${index + 1} · Unknown step · ${row.key.split(/[\\/]/).pop()}`;
    const run = (counts.get(String(step)) ?? 0) + 1;
    counts.set(String(step), run);
    return `Step ${step.toLocaleString()} · Run ${run}${timestamp === null ? '' : ` · ${timestamp}`}`;
  });
}
