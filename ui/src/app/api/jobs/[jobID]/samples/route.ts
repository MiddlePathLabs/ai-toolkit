import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import path from 'path';
import fs from 'fs/promises';
import { getTrainingFolder } from '@/server/settings';
import { readSampleEvidence } from '@/server/sampleEvidence';

export async function GET(request: NextRequest, { params }: { params: Promise<{ jobID: string }> }) {
  const { jobID } = await params;

  const job = await prisma.job.findUnique({
    where: { id: jobID },
  });

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  const trainingFolder = await getTrainingFolder();
  const samplesFolder = path.resolve(trainingFolder, job.name, 'samples');
  const relative = path.relative(path.resolve(trainingFolder), samplesFolder);
  if (relative !== path.join(job.name, 'samples') || /[\\/]/.test(job.name) || job.name === '..') {
    return NextResponse.json({ error: 'Invalid samples path' }, { status: 400 });
  }
  try {
    const [realTraining, realSamples] = await Promise.all([fs.realpath(trainingFolder), fs.realpath(samplesFolder)]);
    if (path.relative(realTraining, realSamples) !== relative) {
      return NextResponse.json({ error: 'Invalid samples path' }, { status: 400 });
    }
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
  }
  return NextResponse.json(await readSampleEvidence(samplesFolder));
}
