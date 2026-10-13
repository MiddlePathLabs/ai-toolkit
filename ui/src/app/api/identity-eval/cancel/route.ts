// POST /api/identity-eval/cancel -- kill the job's running evaluation
// worker. Partial results already written to results.json stay valid.
import { NextResponse } from 'next/server';
import { getIdentityJobContext, readRunState, clearRunState, pidAlive } from '@/server/identityEval';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function POST(request: Request) {
  let jobId: string | undefined;
  try {
    ({ jobId } = (await request.json()) as { jobId?: string });
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 });
  }
  if (!jobId) return NextResponse.json({ error: 'jobId is required' }, { status: 400 });
  const context = await getIdentityJobContext(jobId);
  if (!context) return NextResponse.json({ error: 'Job not found' }, { status: 404 });

  const state = await readRunState(context.identityDir);
  if (!state || !pidAlive(state.pid)) {
    await clearRunState(context.identityDir);
    return NextResponse.json({ canceled: false, running: false });
  }
  try {
    process.kill(state.pid);
  } catch (error) {
    return NextResponse.json(
      { error: `Failed to kill worker pid ${state.pid}: ${error instanceof Error ? error.message : String(error)}` },
      { status: 500 },
    );
  }
  await clearRunState(context.identityDir);
  return NextResponse.json({ canceled: true, pid: state.pid });
}
