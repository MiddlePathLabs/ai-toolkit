import fs from 'fs/promises';
import path from 'path';
import { randomUUID } from 'crypto';

const mediaExtension = /\.(png|jpg|jpeg|webp|bmp|gif|tiff|mp4|mp3|wav|flac|ogg)$/i;
const sampleExtension = /\.(png|jpg|jpeg|webp|bmp|gif|tiff|mp4|mp3|wav|flac|ogg|txt)$/i;
const stem = (filename: string) => filename.slice(0, filename.lastIndexOf('.'));
const safeFilename = (value: unknown): value is string =>
  typeof value === 'string' && !/[\\/]/.test(value) && !value.startsWith('.') && sampleExtension.test(value);
const isMissing = (error: unknown) => (error as NodeJS.ErrnoException).code === 'ENOENT';

async function filenames(folder: string): Promise<string[]> {
  try {
    return (await fs.readdir(folder, { withFileTypes: true })).filter(entry => entry.isFile()).map(entry => entry.name);
  } catch (error) {
    if (isMissing(error)) return [];
    throw error;
  }
}

export async function readSampleEvidence(samplesFolder: string) {
  const folder = path.resolve(samplesFolder);
  const names = (await filenames(folder)).filter(safeFilename);
  const deletedNames = (await filenames(path.join(folder, '.deleted'))).filter(safeFilename);
  const plansFolder = path.join(folder, '.sample-plans');
  const plannedNames: string[] = [];
  for (const name of await filenames(plansFolder)) {
    if (!name.endsWith('.json')) continue;
    try {
      const plan = JSON.parse(await fs.readFile(path.join(plansFolder, name), 'utf8'));
      if (!Array.isArray(plan?.samples)) continue;
      for (const sample of plan.samples) {
        if (safeFilename(sample?.filename)) plannedNames.push(sample.filename);
      }
    } catch (error) {
      if (error instanceof SyntaxError || isMissing(error)) continue;
      throw error;
    }
  }
  const mediaStems = new Set(
    [...names, ...plannedNames, ...deletedNames].filter(name => mediaExtension.test(name)).map(stem),
  );
  const isSample = (name: string) => mediaExtension.test(name) || !mediaStems.has(stem(name));
  const actualStems = new Set(names.filter(isSample).map(stem));
  const missingPaths = (entries: string[]) =>
    [
      ...new Map(
        entries.filter(name => isSample(name) && !actualStems.has(stem(name))).map(name => [stem(name), name]),
      ).values(),
    ]
      .map(name => path.join(folder, name))
      .sort();
  return {
    samples: names
      .filter(isSample)
      .map(name => path.join(folder, name))
      .sort(),
    plannedSamples: missingPaths(plannedNames),
    deletedSamples: missingPaths(deletedNames),
  };
}

function isContained(root: string, filename: string): boolean {
  const relative = path.relative(root, filename);
  return relative !== '' && relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative);
}

export class SampleFileError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}

export async function deleteSampleFile(imgPath: unknown, datasetsRoot: string, trainingRoot: string): Promise<void> {
  if (typeof imgPath !== 'string' || !imgPath || imgPath.includes('\0')) {
    throw new SampleFileError('Invalid image path', 400);
  }
  const filename = path.resolve(imgPath);
  const dataset = path.resolve(datasetsRoot);
  const training = path.resolve(trainingRoot);
  if (![dataset, training].some(root => isContained(root, filename))) {
    throw new SampleFileError('Invalid image path', 400);
  }
  const segments = path.relative(training, filename).split(path.sep);
  const isTrainingSample =
    isContained(training, filename) &&
    segments.length === 3 &&
    segments[1] === 'samples' &&
    !segments[0].startsWith('.');
  if (!mediaExtension.test(filename) && !(isTrainingSample && /\.txt$/i.test(filename))) {
    throw new SampleFileError('Not a sample or image', 400);
  }
  const parent = path.dirname(filename);
  let realParent: string;
  try {
    realParent = await fs.realpath(parent);
  } catch (error) {
    if (isMissing(error)) return;
    throw error;
  }
  const roots = await Promise.all(
    [dataset, training].map(async root => {
      try {
        return await fs.realpath(root);
      } catch (error) {
        if (isMissing(error)) return root;
        throw error;
      }
    }),
  );
  if (!roots.some(root => isContained(root, path.join(realParent, path.basename(filename))))) {
    throw new SampleFileError('Invalid image path', 400);
  }
  // Sample evidence must remain in the exact job samples directory, not a symlink target elsewhere.
  if (isTrainingSample && path.relative(roots[1], realParent) !== path.join(segments[0], 'samples')) {
    throw new SampleFileError('Invalid sample path', 400);
  }
  if (isTrainingSample && /\.txt$/i.test(filename)) {
    const evidence = await readSampleEvidence(parent);
    if (
      [...evidence.samples, ...evidence.plannedSamples, ...evidence.deletedSamples].some(
        sample => mediaExtension.test(sample) && stem(sample) === stem(filename),
      )
    ) {
      throw new SampleFileError('Text is a media caption, not a sample', 400);
    }
  }
  const markers = path.join(parent, '.deleted');
  let pendingMarker: string | null = null;
  const markerPath = path.join(markers, path.basename(filename));
  if (isTrainingSample) {
    await fs.mkdir(markers, { recursive: true });
    if ((await fs.realpath(markers)) !== path.join(realParent, '.deleted')) {
      throw new SampleFileError('Invalid deletion marker path', 400);
    }
    try {
      if (!(await fs.lstat(markerPath)).isFile()) {
        throw new SampleFileError('Invalid deletion marker file', 400);
      }
    } catch (error) {
      if (!isMissing(error)) throw error;
    }
    pendingMarker = path.join(markers, `.${randomUUID()}.tmp`);
    await fs.writeFile(pendingMarker, '', { flag: 'wx' });
  }
  try {
    try {
      await fs.unlink(filename);
    } catch (error) {
      if (isMissing(error)) return;
      throw error;
    }
    if (pendingMarker) await fs.rename(pendingMarker, markerPath);
  } finally {
    if (pendingMarker) {
      try {
        await fs.unlink(pendingMarker);
      } catch (error) {
        if (!isMissing(error)) throw error;
      }
    }
  }
  if (!/\.txt$/i.test(filename)) {
    try {
      await fs.unlink(stem(filename) + '.txt');
    } catch (error) {
      if (!isMissing(error)) throw error;
    }
  }
}
