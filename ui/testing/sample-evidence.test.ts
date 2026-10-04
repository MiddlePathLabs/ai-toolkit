import assert from 'node:assert/strict';
import test, { type TestContext } from 'node:test';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { deleteSampleFile, readSampleEvidence, SampleFileError } from '../src/server/sampleEvidence';
import { buildSampleMatrix } from '../src/utils/sampleImages';

const filename = (index: number, ext = 'png') => `1234500__000000042_${index}_seed-100_steps-4.${ext}`;

async function fixture(t: TestContext) {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'sample-evidence-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const training = path.join(root, 'training');
  const datasets = path.join(root, 'datasets');
  const samples = path.join(training, 'job', 'samples');
  await fs.mkdir(samples, { recursive: true });
  await fs.mkdir(datasets);
  return { root, training, datasets, samples };
}

async function plan(samples: string, names: string[]) {
  const folder = path.join(samples, '.sample-plans');
  await fs.mkdir(folder, { recursive: true });
  await fs.writeFile(
    path.join(folder, '1234500_42.json'),
    JSON.stringify({
      timestamp: 1234500,
      trainingStep: 42,
      samples: names.map((name, index) => ({ filename: name, index, seed: 100, steps: 4 })),
    }),
  );
}

test('missing sample folder returns all evidence arrays empty', async t => {
  const { samples } = await fixture(t);
  assert.deepEqual(await readSampleEvidence(path.join(samples, 'absent')), {
    samples: [],
    plannedSamples: [],
    deletedSamples: [],
  });
});

test('successful concurrent media and text deletions survive a fresh read and retain a fully deleted row', async t => {
  const { samples, training, datasets } = await fixture(t);
  const names = [filename(0), filename(1, 'mp4'), filename(2, 'txt')];
  await plan(samples, names);
  await Promise.all(names.map(name => fs.writeFile(path.join(samples, name), 'sample')));
  await fs.writeFile(path.join(samples, filename(0, 'txt')), 'caption');
  await fs.writeFile(path.join(samples, filename(1, 'txt')), 'caption');
  await Promise.all(names.map(name => deleteSampleFile(path.join(samples, name), datasets, training)));
  const evidence = await readSampleEvidence(samples);
  const expected = names.map(name => path.join(samples, name)).sort();
  assert.deepEqual(evidence, { samples: [], plannedSamples: expected, deletedSamples: expected });
  assert.deepEqual((await fs.readdir(path.join(samples, '.deleted'))).sort(), [...names].sort());
  assert.equal(await fs.readFile(path.join(samples, '.deleted', names[0]), 'utf8'), '');
  await assert.rejects(fs.access(path.join(samples, filename(0, 'txt'))), { code: 'ENOENT' });
  await assert.rejects(fs.access(path.join(samples, filename(1, 'txt'))), { code: 'ENOENT' });
  const rows = buildSampleMatrix(evidence.samples, null, 3, evidence);
  assert.deepEqual(rows, [
    { key: '42:1234500', paths: [null, null, null], missing: ['deleted', 'deleted', 'deleted'] },
  ]);
  await Promise.all(names.map(name => deleteSampleFile(path.join(samples, name), datasets, training)));
  assert.deepEqual(await readSampleEvidence(samples), evidence);
});

test('same-file concurrent deletion is idempotent with one durable marker', async t => {
  const { samples, training, datasets } = await fixture(t);
  const image = path.join(samples, filename(0));
  await fs.writeFile(image, 'sample');
  await Promise.all(Array.from({ length: 8 }, () => deleteSampleFile(image, datasets, training)));
  assert.deepEqual(await fs.readdir(path.join(samples, '.deleted')), [filename(0)]);
  assert.deepEqual((await readSampleEvidence(samples)).deletedSamples, [image]);
});

test('failed unlink and never-existing files never create deleted evidence', async t => {
  const { samples, training, datasets } = await fixture(t);
  const directory = path.join(samples, filename(0));
  await fs.mkdir(directory);
  await assert.rejects(deleteSampleFile(directory, datasets, training));
  await deleteSampleFile(path.join(samples, filename(1)), datasets, training);
  assert.deepEqual((await readSampleEvidence(samples)).deletedSamples, []);
  assert.deepEqual(await fs.readdir(path.join(samples, '.deleted')), []);
  assert.equal((await fs.stat(directory)).isDirectory(), true);
});

test('plans, tombstones and media use shared stems across extension changes and exclude captions', async t => {
  const { samples } = await fixture(t);
  await plan(samples, [filename(0), filename(1), filename(2, 'txt'), '../outside.png', 'nested/file.png']);
  await fs.mkdir(path.join(samples, '.deleted'));
  await fs.writeFile(path.join(samples, '.deleted', filename(0, 'webp')), '');
  await fs.writeFile(path.join(samples, '.deleted', filename(1, 'mp4')), '');
  await fs.writeFile(path.join(samples, filename(0, 'mp4')), 'actual media');
  await fs.writeFile(path.join(samples, filename(0, 'txt')), 'caption');
  await fs.writeFile(path.join(samples, filename(1, 'txt')), 'orphan caption');
  await fs.writeFile(path.join(samples, filename(2, 'txt')), 'text sample');
  await fs.mkdir(path.join(samples, 'directory.png'));
  await fs.mkdir(path.join(samples, '.thumbs'));
  await fs.writeFile(path.join(samples, '.thumbs', filename(3)), 'thumbnail');
  const evidence = await readSampleEvidence(samples);
  assert.deepEqual(evidence.samples, [path.join(samples, filename(0, 'mp4')), path.join(samples, filename(2, 'txt'))]);
  assert.deepEqual(evidence.plannedSamples, [path.join(samples, filename(1))]);
  assert.deepEqual(evidence.deletedSamples, [path.join(samples, filename(1, 'mp4'))]);
  const rows = buildSampleMatrix(evidence.samples, null, 3, evidence);
  assert.deepEqual(rows[0].missing, [null, 'deleted', null]);
  assert.equal(rows[0].paths[1], null);
});

test('only complete JSON plans supply evidence and malformed records do not hide real samples', async t => {
  const { samples } = await fixture(t);
  await plan(samples, [filename(1)]);
  const plans = path.join(samples, '.sample-plans');
  await fs.writeFile(path.join(plans, 'null.json'), 'null');
  await fs.writeFile(path.join(plans, 'broken.json'), '{');
  await fs.writeFile(path.join(plans, 'uncommitted.tmp'), JSON.stringify({ samples: [{ filename: filename(2) }] }));
  await fs.writeFile(path.join(samples, filename(0)), 'actual media');
  const evidence = await readSampleEvidence(samples);
  assert.deepEqual(evidence.plannedSamples, [path.join(samples, filename(1))]);
  assert.deepEqual(buildSampleMatrix(evidence.samples, null, 3, evidence)[0].missing, [
    null,
    'not-generated',
    'unavailable',
  ]);
});

test('dataset and other training media deletions keep caption cleanup without sample markers', async t => {
  const { datasets, training } = await fixture(t);
  for (const parent of [path.join(datasets, 'set'), path.join(training, 'job', 'other')]) {
    await fs.mkdir(parent, { recursive: true });
    const image = path.join(parent, filename(0));
    const caption = path.join(parent, filename(0, 'txt'));
    await fs.writeFile(image, 'image');
    await fs.writeFile(caption, 'caption');
    await deleteSampleFile(image, datasets, training);
    await assert.rejects(fs.access(image), { code: 'ENOENT' });
    await assert.rejects(fs.access(caption), { code: 'ENOENT' });
    await assert.rejects(fs.access(path.join(parent, '.deleted')), { code: 'ENOENT' });
  }
  const datasetText = path.join(datasets, 'set', filename(1, 'txt'));
  await fs.writeFile(datasetText, 'dataset caption');
  await assert.rejects(deleteSampleFile(datasetText, datasets, training), SampleFileError);
  assert.equal(await fs.readFile(datasetText, 'utf8'), 'dataset caption');
});

test('normalized containment rejects sibling prefix and traversal paths without touching files', async t => {
  const { root, datasets, training, samples } = await fixture(t);
  const outside = path.join(root, 'training-sibling', filename(0));
  await fs.mkdir(path.dirname(outside));
  await fs.writeFile(outside, 'outside');
  await assert.rejects(deleteSampleFile(outside, datasets, training), SampleFileError);
  await assert.rejects(
    deleteSampleFile(path.join(samples, '..', '..', '..', '..', 'training-sibling', filename(0)), datasets, training),
    SampleFileError,
  );
  assert.equal(await fs.readFile(outside, 'utf8'), 'outside');
  const normalized = path.join(samples, filename(1));
  await fs.writeFile(normalized, 'sample');
  await deleteSampleFile(path.join(samples, '..', 'samples', filename(1)), datasets, training);
  assert.deepEqual((await readSampleEvidence(samples)).deletedSamples, [normalized]);
});

test('text media captions cannot be deleted as standalone training samples', async t => {
  const { samples, training, datasets } = await fixture(t);
  await plan(samples, [filename(0)]);
  const caption = path.join(samples, filename(0, 'txt'));
  await fs.writeFile(caption, 'caption');
  await assert.rejects(deleteSampleFile(caption, datasets, training), SampleFileError);
  assert.equal(await fs.readFile(caption, 'utf8'), 'caption');
  assert.deepEqual((await readSampleEvidence(samples)).deletedSamples, []);
});

test('sample directory symlink escapes cannot delete or mark files in another directory', async t => {
  const { root, samples, training, datasets } = await fixture(t);
  const target = path.join(root, 'outside');
  await fs.mkdir(target);
  const image = path.join(target, filename(0));
  await fs.writeFile(image, 'outside');
  const linkedSamples = path.join(training, 'linked-job', 'samples');
  await fs.mkdir(path.dirname(linkedSamples));
  await fs.symlink(target, linkedSamples, process.platform === 'win32' ? 'junction' : 'dir');
  await assert.rejects(deleteSampleFile(path.join(linkedSamples, filename(0)), datasets, training), SampleFileError);
  assert.equal(await fs.readFile(image, 'utf8'), 'outside');
  await assert.rejects(fs.access(path.join(target, '.deleted')), { code: 'ENOENT' });
  assert.deepEqual((await readSampleEvidence(samples)).deletedSamples, []);
});

test('an unusable marker path fails before deleting real media', async t => {
  const { samples, training, datasets } = await fixture(t);
  const image = path.join(samples, filename(0));
  await fs.writeFile(image, 'sample');
  await fs.mkdir(path.join(samples, '.deleted', filename(0)), { recursive: true });
  await assert.rejects(deleteSampleFile(image, datasets, training), SampleFileError);
  assert.equal(await fs.readFile(image, 'utf8'), 'sample');
  assert.deepEqual((await readSampleEvidence(samples)).deletedSamples, []);
});
