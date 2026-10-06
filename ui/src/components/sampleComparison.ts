import type { JobConfig, SampleConfig } from '../types';
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

export interface CrossJobPane {
  rowKey: string | null;
  column: number;
}

export interface CrossJobSelection {
  left: CrossJobPane;
  right: CrossJobPane;
}

export interface TrainingPace {
  batchSize: number | null;
  gradientAccumulation: number | null;
  legacyAccumulation: number | null;
  datasets: string[];
}

export interface NearestStepOffer {
  side: 'left' | 'right';
  selectedStep: number;
  nearestStep: number;
  rowKey: string;
}

export function getComparisonRowStep(row: ComparisonRow): number | null {
  const info = row.paths.map(path => (path ? parseSampleFilename(path) : null)).find(Boolean);
  if (info) return info.trainingStep;
  const identity = /^(\d+):(\d+)$/.exec(row.key);
  if (!identity) return null;
  const step = Number(identity[1]);
  return Number.isSafeInteger(step) ? step : null;
}

export function findExactStepRow<T extends ComparisonRow>(rows: T[], step: number): T | null {
  let match: T | null = null;
  for (const row of rows) {
    if (getComparisonRowStep(row) === step) match = row;
  }
  return match;
}

export function findNearestStepRow<T extends ComparisonRow>(rows: T[], step: number): T | null {
  let best: T | null = null;
  let bestStep: number | null = null;
  for (const row of rows) {
    const rowStep = getComparisonRowStep(row);
    if (rowStep === null) continue;
    if (best === null || bestStep === null || isNearerStep(rowStep, bestStep, step)) {
      best = row;
      bestStep = rowStep;
    }
  }
  return best;
}

function isNearerStep(candidate: number, current: number, target: number): boolean {
  const candidateDistance = Math.abs(candidate - target);
  const currentDistance = Math.abs(current - target);
  return candidateDistance < currentDistance || (candidateDistance === currentDistance && candidate >= current);
}

export function getDefaultTrainedColumn(columns: ComparisonColumn[]): number {
  const available = columns.filter(column => column.path);
  return (
    available.find(column => column.isRaw === false)?.index ??
    available[0]?.index ??
    columns.find(column => column.isRaw === false)?.index ??
    columns[0]?.index ??
    -1
  );
}

export function matchComparisonColumn(columns: ComparisonColumn[], source?: ComparisonColumn): number | null {
  if (!source?.prompt) return null;
  let best: number | null = null;
  let bestScore = -1;
  for (const column of columns) {
    if (column.prompt !== source.prompt) continue;
    const score = promptMatchScore(column, source);
    if (score > bestScore || (score === bestScore && (best === null || column.index < best))) {
      best = column.index;
      bestScore = score;
    }
  }
  return best;
}

function promptMatchScore(column: ComparisonColumn, source: ComparisonColumn): number {
  const sameWeight = source.isRaw !== null && column.isRaw !== null && column.isRaw === source.isRaw;
  const sameSeed =
    source.metadata.seed !== null && column.metadata.seed !== null && column.metadata.seed === source.metadata.seed;
  return Number(Boolean(column.path)) * 100 + Number(sameWeight) * 10 + Number(sameSeed);
}

export function createCrossJobSelection(
  leftRows: SampleRow[],
  leftConfig: SampleConfig | null,
  rightRows: SampleRow[],
  rightConfig: SampleConfig | null,
  preferredLeftRowKey: string | null,
): CrossJobSelection {
  const preferred = leftRows.find(row => row.key === preferredLeftRowKey);
  const leftRow =
    (preferred && preferred.paths.some(Boolean) ? preferred : undefined) ??
    [...leftRows]
      .reverse()
      .find(row => getComparisonColumns(row, leftConfig).some(column => column.path && column.isRaw === false)) ??
    [...leftRows].reverse().find(row => row.paths.some(Boolean)) ??
    leftRows[leftRows.length - 1];
  const leftColumns = leftRow ? getComparisonColumns(leftRow, leftConfig) : [];
  const leftColumn = getDefaultTrainedColumn(leftColumns);
  const leftStep = leftRow ? getComparisonRowStep(leftRow) : null;
  const rightRow = leftStep === null ? null : findExactStepRow(rightRows, leftStep);
  const rightColumns = getComparisonColumns(rightRow ?? { key: 'missing', paths: [] }, rightConfig);
  const rightColumn =
    matchComparisonColumn(rightColumns, leftColumns[leftColumn]) ?? getDefaultTrainedColumn(rightColumns);
  return {
    left: { rowKey: leftRow?.key ?? null, column: leftColumn },
    right: { rowKey: rightRow?.key ?? null, column: rightColumn },
  };
}

export function selectCrossJobRow(
  selection: CrossJobSelection,
  side: 'left' | 'right',
  rowKey: string,
  leftRows: SampleRow[],
  rightRows: SampleRow[],
  syncSteps: boolean,
): CrossJobSelection {
  const sourceRows = side === 'left' ? leftRows : rightRows;
  const sourceRow = sourceRows.find(row => row.key === rowKey);
  if (!sourceRow) return selection;
  const next: CrossJobSelection = { ...selection, [side]: { ...selection[side], rowKey } };
  if (!syncSteps) return next;
  const otherSide = side === 'left' ? 'right' : 'left';
  const step = getComparisonRowStep(sourceRow);
  const exact = step === null ? null : findExactStepRow(side === 'left' ? rightRows : leftRows, step);
  return { ...next, [otherSide]: { ...selection[otherSide], rowKey: exact?.key ?? null } };
}

export function selectCrossJobColumn(
  selection: CrossJobSelection,
  side: 'left' | 'right',
  column: number,
  leftColumns: ComparisonColumn[],
  rightColumns: ComparisonColumn[],
  matchPrompt: boolean,
): CrossJobSelection {
  const next: CrossJobSelection = { ...selection, [side]: { ...selection[side], column } };
  if (!matchPrompt) return next;
  const sourceColumns = side === 'left' ? leftColumns : rightColumns;
  const otherColumns = side === 'left' ? rightColumns : leftColumns;
  const matched = matchComparisonColumn(otherColumns, sourceColumns[column]);
  if (matched === null) return next;
  const otherSide = side === 'left' ? 'right' : 'left';
  return { ...next, [otherSide]: { ...selection[otherSide], column: matched } };
}

export function nearestStepOffer(
  selection: CrossJobSelection,
  leftRows: SampleRow[],
  rightRows: SampleRow[],
): NearestStepOffer | null {
  return (
    offerForMissingSide(selection, 'right', leftRows, rightRows) ??
    offerForMissingSide(selection, 'left', rightRows, leftRows)
  );
}

function offerForMissingSide(
  selection: CrossJobSelection,
  missingSide: 'left' | 'right',
  presentRows: SampleRow[],
  missingRows: SampleRow[],
): NearestStepOffer | null {
  const presentSide = missingSide === 'right' ? 'left' : 'right';
  if (selection[missingSide].rowKey || !selection[presentSide].rowKey) return null;
  const present = presentRows.find(row => row.key === selection[presentSide].rowKey);
  const selectedStep = present ? getComparisonRowStep(present) : null;
  if (selectedStep === null) return null;
  const nearest = findNearestStepRow(missingRows, selectedStep);
  const nearestStep = nearest ? getComparisonRowStep(nearest) : null;
  if (!nearest || nearestStep === null) return null;
  return { side: missingSide, selectedStep, nearestStep, rowKey: nearest.key };
}

export function applyNearestCrossJobStep(
  selection: CrossJobSelection,
  leftRows: SampleRow[],
  rightRows: SampleRow[],
): CrossJobSelection | null {
  const offer = nearestStepOffer(selection, leftRows, rightRows);
  if (!offer) return null;
  return { ...selection, [offer.side]: { ...selection[offer.side], rowKey: offer.rowKey } };
}

export function reconcileCrossJobSelection(
  selection: CrossJobSelection,
  leftRows: SampleRow[],
  rightRows: SampleRow[],
  syncSteps: boolean,
): CrossJobSelection {
  const leftKey =
    selection.left.rowKey && leftRows.some(row => row.key === selection.left.rowKey) ? selection.left.rowKey : null;
  const rightKey =
    selection.right.rowKey && rightRows.some(row => row.key === selection.right.rowKey) ? selection.right.rowKey : null;
  let nextLeft = leftKey;
  let nextRight = rightKey;
  if (syncSteps && nextLeft && !nextRight) nextRight = exactKey(rightRows, leftRows, nextLeft);
  if (syncSteps && nextRight && !nextLeft) nextLeft = exactKey(leftRows, rightRows, nextRight);
  if (nextLeft === selection.left.rowKey && nextRight === selection.right.rowKey) return selection;
  return {
    ...selection,
    left: { ...selection.left, rowKey: nextLeft },
    right: { ...selection.right, rowKey: nextRight },
  };
}

function exactKey(targetRows: SampleRow[], sourceRows: SampleRow[], sourceKey: string): string | null {
  const source = sourceRows.find(row => row.key === sourceKey);
  const step = source ? getComparisonRowStep(source) : null;
  return step === null ? null : (findExactStepRow(targetRows, step)?.key ?? null);
}

export function getCrossJobStepNotice(
  leftStep: number | null,
  rightStep: number | null,
  leftJobName: string,
  rightJobName: string,
): string | null {
  if (leftStep !== null && rightStep === null)
    return `${rightJobName} has no sample at step ${leftStep.toLocaleString()}.`;
  if (rightStep !== null && leftStep === null)
    return `${leftJobName} has no sample at step ${rightStep.toLocaleString()}.`;
  if (leftStep !== null && rightStep !== null && leftStep !== rightStep) {
    return `Training steps differ (${leftStep.toLocaleString()} vs ${rightStep.toLocaleString()}).`;
  }
  return null;
}

export function getTrainingPace(jobConfigText: string | null | undefined): TrainingPace | null {
  return paceFromProcess(readProcess(jobConfigText));
}

function readProcess(jobConfigText: string | null | undefined): JobConfig['config']['process'][number] | null {
  if (!jobConfigText) return null;
  try {
    const parsed = JSON.parse(jobConfigText) as JobConfig;
    return parsed.config?.process?.[0] ?? null;
  } catch {
    return null;
  }
}

function paceFromProcess(process: JobConfig['config']['process'][number] | null): TrainingPace | null {
  if (!process?.train) return null;
  const train = process.train as typeof process.train & { gradient_accumulation_steps?: number };
  return {
    batchSize: finiteNumber(train.batch_size),
    gradientAccumulation: finiteNumber(train.gradient_accumulation ?? 1),
    legacyAccumulation: finiteNumber(train.gradient_accumulation_steps ?? 1),
    datasets: (process.datasets ?? []).map(datasetIdentity).sort(),
  };
}

function finiteNumber(value: unknown): number | null {
  if (typeof value === 'boolean' || value === null || value === undefined || value === '') return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function datasetIdentity(dataset: { folder_path?: string; batch_size?: number; num_repeats?: number }): string {
  const raw = dataset.folder_path ?? '';
  const folder = raw.replace(/\\/g, '/').replace(/\/+$/, '');
  const key = /^[a-zA-Z]:[\\/]/.test(raw) || raw.includes('\\') ? folder.toLowerCase() : folder;
  return `${key}|${dataset.batch_size ?? ''}|${dataset.num_repeats ?? ''}`;
}

export function getTrainingPaceNotice(left: TrainingPace | null, right: TrainingPace | null): string | null {
  if (!left || !right) return 'Training settings cannot be compared.';
  const differences = paceDifferences(left, right);
  if (!differences.length) return null;
  const label = joinDifferences(differences);
  const sentence = `${label} ${differences.length === 1 ? 'differs' : 'differ'}, so the same step is not the same amount of training.`;
  return crossKeyAccumulation(left, right)
    ? `${sentence} Legacy gradient_accumulation_steps counts every batch as a step; gradient_accumulation counts a step after those batches.`
    : sentence;
}

function crossKeyAccumulation(left: TrainingPace, right: TrainingPace): boolean {
  const leftModern = left.gradientAccumulation !== 1 && left.legacyAccumulation === 1;
  const rightModern = right.gradientAccumulation !== 1 && right.legacyAccumulation === 1;
  const leftLegacy = left.legacyAccumulation !== 1 && left.gradientAccumulation === 1;
  const rightLegacy = right.legacyAccumulation !== 1 && right.gradientAccumulation === 1;
  return (leftModern && rightLegacy) || (leftLegacy && rightModern);
}

function paceDifferences(left: TrainingPace, right: TrainingPace): string[] {
  const differences: string[] = [];
  if (left.batchSize !== right.batchSize) differences.push('Batch size');
  if (
    left.gradientAccumulation !== right.gradientAccumulation ||
    left.legacyAccumulation !== right.legacyAccumulation
  ) {
    differences.push('Gradient accumulation');
  }
  if (left.datasets.join('\0') !== right.datasets.join('\0')) differences.push('Datasets');
  return differences;
}

function joinDifferences(differences: string[]): string {
  if (differences.length === 1) return differences[0];
  return differences.length === 2
    ? `${differences[0]} and ${differences[1]}`
    : `${differences.slice(0, -1).join(', ')}, and ${differences[differences.length - 1]}`;
}

export function readComparisonJob(jobConfigText: string | null | undefined): {
  sampleConfig: SampleConfig | null;
  hasEma: boolean;
  pace: TrainingPace | null;
} {
  const process = readProcess(jobConfigText);
  return {
    sampleConfig: process?.sample ?? null,
    hasEma: Boolean(process?.train?.ema_config?.use_ema),
    pace: paceFromProcess(process),
  };
}
