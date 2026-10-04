import type { SampleConfig, SampleItem } from '../types';

export interface SampleFilename {
  filename: string;
  timestamp: number;
  trainingStep: number;
  sampleIndex: number;
  isRaw: boolean;
  seed: number | null;
  inferenceSteps: number | null;
  hasMetadata: boolean;
}

export interface SampleRow {
  key: string;
  paths: (string | null)[];
  missing: ('not-generated' | 'deleted' | 'unavailable' | null)[];
}

export interface SampleMetadata {
  trainingStep: number | null;
  seed: number | null;
  inferenceSteps: number | null;
  isRaw: boolean;
}

export function parseSampleFilename(path: string): SampleFilename | null {
  const filename = path.split(/[\\/]/).pop() ?? '';
  const match = /^(RAW_)?(\d+)__(\d+)_(\d+)(?:_seed-(-?\d+)_steps-(\d+))?\.[^.]+$/.exec(filename);
  if (!match) return null;
  const values = [match[2], match[3], match[4], match[5], match[6]];
  if (values.some(value => value !== undefined && !Number.isSafeInteger(Number(value)))) return null;
  return {
    filename,
    timestamp: Number(match[2]),
    trainingStep: Number(match[3]),
    sampleIndex: Number(match[4]),
    isRaw: Boolean(match[1]),
    seed: match[5] === undefined ? null : Number(match[5]),
    inferenceSteps: match[6] === undefined ? null : Number(match[6]),
    hasMetadata: match[5] !== undefined,
  };
}

export function getSampleItems(config: SampleConfig | null): SampleItem[] {
  return config?.samples?.length ? config.samples : (config?.prompts?.map(prompt => ({ prompt })) ?? []);
}

export function buildSampleMatrix(
  paths: string[],
  config: SampleConfig | null,
  minimumColumns = 1,
  evidence?: { plannedSamples: string[]; deletedSamples: string[] },
): SampleRow[] {
  const items = getSampleItems(config);
  const columns = Math.max(items.length, minimumColumns, 1);
  const partitions = { raw: [] as number[], normal: [] as number[] };
  items.forEach((item, index) => partitions[item.raw_weights ? 'raw' : 'normal'].push(index));
  const byStem = new Map<
    string,
    { path: string; info: SampleFilename | null; missing: SampleRow['missing'][number] }
  >();
  for (const [source, missing] of [
    [evidence?.plannedSamples ?? [], 'not-generated'],
    [evidence?.deletedSamples ?? [], 'deleted'],
    [paths, null],
  ] as const) {
    for (const path of source) {
      byStem.set(path.replace(/\.[^./\\]+$/, ''), { path, info: parseSampleFilename(path), missing });
    }
  }
  const entries = [...byStem.values()];
  entries.sort((a, b) => {
    if (!a.info || !b.info) return a.info ? -1 : b.info ? 1 : a.path.localeCompare(b.path);
    return (
      a.info.trainingStep - b.info.trainingStep ||
      a.info.timestamp - b.info.timestamp ||
      a.info.sampleIndex - b.info.sampleIndex ||
      a.path.localeCompare(b.path)
    );
  });

  const iterations: (typeof entries)[] = [];
  let seen = new Set<string>();
  for (const entry of entries) {
    const previous = iterations[iterations.length - 1];
    const identity = entry.info ? `${entry.info.isRaw}:${entry.info.sampleIndex}` : entry.path;
    if (
      !entry.info ||
      !previous?.[0].info ||
      previous[0].info.trainingStep !== entry.info.trainingStep ||
      seen.has(identity)
    ) {
      iterations.push([]);
      seen = new Set();
    }
    iterations[iterations.length - 1].push(entry);
    seen.add(identity);
  }

  return iterations.flatMap(iteration => {
    const first = iteration[0];
    const hasRaw = iteration.some(entry => entry.info?.isRaw);
    const row: SampleRow = {
      key: first.info ? `${first.info.trainingStep}:${first.info.timestamp}` : first.path,
      paths: Array(columns).fill(null),
      missing: Array(columns).fill('unavailable'),
    };
    const overflow: SampleRow[] = [];
    const occupied = new Set<number>();
    for (const { path, info, missing } of iteration) {
      const partition = partitions[info?.isRaw ? 'raw' : 'normal'];
      const column = info && !info.hasMetadata && hasRaw ? partition[info.sampleIndex] : (info?.sampleIndex ?? 0);
      if (column === undefined || column >= columns || occupied.has(column)) {
        overflow.push({
          key: path.replace(/\.[^./\\]+$/, ''),
          paths: [missing === null ? path : null, ...Array(columns - 1).fill(null)],
          missing: [missing, ...Array(columns - 1).fill('unavailable')],
        });
      } else {
        row.paths[column] = missing === null ? path : null;
        row.missing[column] = missing;
        occupied.add(column);
      }
    }
    return [row, ...overflow];
  });
}

export function getAdjacentSamplePath(
  rows: SampleRow[],
  path: string,
  direction: 'up' | 'down' | 'left' | 'right',
): string | null {
  const rowIndex = rows.findIndex(row => row.paths.includes(path));
  if (rowIndex < 0) return null;
  const column = rows[rowIndex].paths.indexOf(path);
  const vertical = direction === 'up' || direction === 'down';
  const delta = direction === 'up' || direction === 'left' ? -1 : 1;
  for (
    let index = (vertical ? rowIndex : column) + delta;
    index >= 0 && index < (vertical ? rows.length : rows[rowIndex].paths.length);
    index += delta
  ) {
    const candidate = vertical ? rows[index].paths[column] : rows[rowIndex].paths[index];
    if (candidate) return candidate;
  }
  return null;
}

function promptNumber(prompt: string, flags: string[]): number | null {
  let value: number | null = null;
  for (const part of prompt.trim().split('--').slice(1)) {
    const space = part.indexOf(' ');
    if (space < 0 || !flags.includes(part.slice(0, space).trim())) continue;
    const content = part.slice(space).trim();
    if (/^-?\d+$/.test(content) && Number.isSafeInteger(Number(content))) value = Number(content);
  }
  return value;
}

export function getSampleMetadata(path: string, config: SampleConfig | null, column: number): SampleMetadata {
  const info = parseSampleFilename(path);
  let seed: number | null = null;
  let inferenceSteps: number | null = null;
  const items = getSampleItems(config);
  if (info && config && column >= 0 && column < items.length) {
    seed = config.seed ?? 0;
    for (let index = 0; index <= column; index++) {
      if (config.walk_seed) seed = (config.seed ?? 0) + index;
      seed = items[index].seed ?? seed;
    }
    seed = promptNumber(items[column].prompt, ['seed', 'd']) ?? seed;
    inferenceSteps =
      promptNumber(items[column].prompt, ['steps', 's']) ?? items[column].sample_steps ?? config.sample_steps ?? 20;
  }
  if (info?.hasMetadata) {
    seed = info.seed;
    inferenceSteps = info.inferenceSteps;
  }
  return {
    trainingStep: info?.trainingStep ?? null,
    seed: seed === -1 ? null : seed,
    inferenceSteps,
    isRaw: info?.isRaw ?? false,
  };
}
