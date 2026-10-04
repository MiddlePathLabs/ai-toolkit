import { NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import { isMac } from '@/helpers/basic';
import { cached } from '@/server/apiCache';
import { parseRawYaml, runAdmission, shouldRunTrainingAdmission, AdmissionUnavailableError } from '@/server/admission';

const invalidRawYamlResponse = (message: string, line: number) =>
  NextResponse.json(
    {
      valid: false,
      diagnostics: [
        {
          rule_id: 'admission.raw_yaml_invalid',
          severity: 'error',
          fields: ['config'],
          reason: `The submitted YAML is invalid: ${message} (line ${line}).`,
          remedy: 'Fix the highlighted YAML syntax before saving or starting this job.',
          phase: 'static',
        },
      ],
      deferred: [],
      source: 'api',
    },
    { status: 422 },
  );
export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const id = searchParams.get('id');
  const job_ref = searchParams.get('job_ref');
  const job_type = searchParams.get('job_type');
  const only_active = searchParams.get('only_active');

  try {
    if (id) {
      const job = await prisma.job.findUnique({
        where: { id },
      });
      return NextResponse.json(job);
    }
    if (job_ref) {
      const job = await prisma.job.findFirst({
        where: { job_ref },
        orderBy: { updated_at: 'desc' },
      });
      return NextResponse.json(job);
    }

    const where: any = {};
    if (job_type) {
      where.job_type = job_type;
    }
    if (only_active === 'true') {
      where.status = { in: ['running', 'queued', 'stopping'] };
      const jobs = await cached(
        'jobs-active',
        () =>
          prisma.job.findMany({
            where,
            orderBy: { created_at: 'desc' },
          }),
        5000,
        { job_type },
      );
      return NextResponse.json({ jobs: jobs });
    }

    const jobs = await prisma.job.findMany({
      where,
      orderBy: { created_at: 'desc' },
    });
    return NextResponse.json({ jobs: jobs });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to fetch training data' }, { status: 500 });
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { id, name, job_config } = body;
    let configToPersist = job_config;

    if (typeof body.raw_yaml === 'string') {
      const parsed = parseRawYaml(body.raw_yaml);
      if (!parsed.valid) return invalidRawYamlResponse(parsed.message, parsed.line);
      configToPersist = parsed.value;
    }

    if (shouldRunTrainingAdmission(configToPersist)) {
      const admission = await runAdmission(configToPersist, 'api');
      if (!admission.valid) return NextResponse.json(admission, { status: 422 });
    }

    let gpu_ids: string = body.gpu_ids;
    if (isMac()) gpu_ids = 'mps';

    const extra: any = {};
    if ('job_ref' in body) extra['job_ref'] = body.job_ref;
    if ('job_type' in body) extra['job_type'] = body.job_type;

    if (id) {
      const training = await prisma.job.update({
        where: { id },
        data: {
          name,
          gpu_ids,
          job_config: JSON.stringify(configToPersist),
          ...extra,
        },
      });
      return NextResponse.json(training);
    }

    const highestQueuePosition = await prisma.job.aggregate({
      _max: {
        queue_position: true,
      },
    });
    const newQueuePosition = (highestQueuePosition._max.queue_position || 0) + 1000;
    const training = await prisma.job.create({
      data: {
        name,
        gpu_ids,
        job_config: JSON.stringify(configToPersist),
        queue_position: newQueuePosition,
        ...extra,
      },
    });
    return NextResponse.json(training);
  } catch (error: unknown) {
    if (error instanceof AdmissionUnavailableError) {
      return NextResponse.json(
        {
          error: 'Canonical admission is unavailable. The job was not saved.',
          code: 'admission_unavailable',
        },
        { status: 503 },
      );
    }
    if (typeof error === 'object' && error !== null && 'code' in error && error.code === 'P2002') {
      return NextResponse.json({ error: 'Job name already exists' }, { status: 409 });
    }
    console.error(error);
    return NextResponse.json({ error: 'Failed to save training data' }, { status: 500 });
  }
}
