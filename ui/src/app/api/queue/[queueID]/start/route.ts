import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import { validateStoredConfigBeforeMutation } from '@/server/admission';
export async function GET(_request: NextRequest, { params }: { params: Promise<{ queueID: string }> }) {
  const { queueID } = await params;
  const queuedJobs = await prisma.job.findMany({
    where: {
      gpu_ids: queueID,
      status: { in: ['queued', 'running', 'stopping'] },
    },
    orderBy: { queue_position: 'asc' },
  });
  for (const job of queuedJobs) {
    const failure = await validateStoredConfigBeforeMutation(job.job_config, {
      source: 'api',
      context: { job_id: job.id, job_name: job.name },
    });
    if (failure) return NextResponse.json(failure.body, { status: failure.status });
  }


  const queue = await prisma.queue.findUnique({
    where: { gpu_ids: queueID },
  });

  if (!queue) {
    // create it if it doesn't exist
    const newQueue = await prisma.queue.create({
      data: { gpu_ids: queueID, is_running: true },
    });
    return NextResponse.json(newQueue);
  }

  const updatedQueue = await prisma.queue.update({
    where: { id: queue.id },
    data: { is_running: true },
  });

  return NextResponse.json(updatedQueue);
}
