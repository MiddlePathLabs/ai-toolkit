// GET /api/identity-eval/status?jobId= -- worker liveness, live progress and
// results (complete or partial). The UI polls this while a run is active.
import { NextRequest, NextResponse } from 'next/server';
import path from 'path';
import { getIdentityJobContext, readJsonFile, readRunState, pidAlive } from '@/server/identityEval';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET(request: NextRequest) {
  const jobId = request.nextUrl.searchParams.get('jobId');
  if (!jobId) return NextResponse.json({ error: 'jobId is required' }, { status: 400 });
  const context = await getIdentityJobContext(jobId);
  if (!context) return NextResponse.json({ error: 'Job not found' }, { status: 404 });

  const state = await readRunState(context.identityDir);
  const running = Boolean(state && pidAlive(state.pid));
  const [progress, results, refs, selection] = await Promise.all([
    readJsonFile<Record<string, unknown>>(context.progressPath),
    readJsonFile<Record<string, unknown>>(context.resultsPath),
    readJsonFile<Record<string, unknown>>(context.refsPath),
    readJsonFile<Record<string, unknown>>(path.join(context.identityDir, 'selection.json')),
  ]);
  return NextResponse.json({
    running,
    state,
    progress,
    results,
    models: (refs as { models?: unknown } | null)?.models ?? null,
    selection: selection ?? { version: 1, training_refs: {}, external_refs: {} },
  });
}
