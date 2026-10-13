// POST /api/identity-eval/clear -- delete results.json (and optionally the
// embedding cache). References, external images and the selection survive.
import { NextResponse } from 'next/server';
import { getIdentityJobContext, readRunState, pidAlive } from '@/server/identityEval';
import fsp from 'fs/promises';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function POST(request: Request) {
  let jobId: string | undefined;
  let includeCache = false;
  try {
    const body = (await request.json()) as { jobId?: string; includeCache?: boolean };
    jobId = body.jobId;
    includeCache = Boolean(body.includeCache);
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 });
  }
  if (!jobId) return NextResponse.json({ error: 'jobId is required' }, { status: 400 });
  const context = await getIdentityJobContext(jobId);
  if (!context) return NextResponse.json({ error: 'Job not found' }, { status: 404 });

  const state = await readRunState(context.identityDir);
  if (state && pidAlive(state.pid)) {
    return NextResponse.json({ error: 'An evaluation is running; cancel it first' }, { status: 409 });
  }

  await fsp.rm(context.resultsPath, { force: true });
  await fsp.rm(context.progressPath, { force: true });
  await fsp.rm(context.runStatePath, { force: true });
  if (includeCache) {
    await fsp.rm(context.cacheDir, { recursive: true, force: true });
  }
  return NextResponse.json({ cleared: true, cacheCleared: includeCache });
}
