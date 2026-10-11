import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';

export async function GET(request: NextRequest, { params }: { params: Promise<{ jobID: string }> }) {
  const { jobID } = await params;

  const job = await prisma.job.update({
    where: { id: jobID },
    data: {
      stop_after_save: true,
      info: 'Saving and stopping on next step...',
    },
  });

  console.log(`Job ${jobID} marked to save and stop on next step`);

  return NextResponse.json(job);
}
