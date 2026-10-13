// POST /api/identity-eval/run -- start an evaluation worker for a job.
// One active evaluation per job: a second request returns the running one.
// The worker is detached (survives the request); progress is polled via
// /status and cancellation is a process kill (partial results stay valid).
import { NextResponse } from 'next/server';
import path from 'path';
import {
  getIdentityJobContext,
  getIdentityModelsDir,
  readJsonFile,
  readRunState,
  runIdentityWorker,
  spawnIdentityWorkerDetached,
  writeJsonFile,
  writeRunState,
  pidAlive,
  isUnder,
} from '@/server/identityEval';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export const maxDuration = 1200;

interface RunBody {
  jobId: string;
  sources?: string[];
  deviceMode?: 'auto' | 'gpu' | 'cpu';
  samples?: string[];
}

export async function POST(request: Request) {
  let body: RunBody;
  try {
    body = (await request.json()) as RunBody;
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 });
  }
  if (!body?.jobId) return NextResponse.json({ error: 'jobId is required' }, { status: 400 });
  const context = await getIdentityJobContext(body.jobId);
  if (!context) return NextResponse.json({ error: 'Job not found' }, { status: 404 });

  // one active evaluation per job
  const existing = await readRunState(context.identityDir);
  if (existing && pidAlive(existing.pid)) {
    return NextResponse.json({ alreadyRunning: true, state: existing }, { status: 200 });
  }

  const sources = (body.sources ?? ['training']).filter(s => s === 'training' || s === 'external');
  if (sources.length === 0) return NextResponse.json({ error: 'sources must include training and/or external' }, { status: 400 });
  const deviceMode = ['auto', 'gpu', 'cpu'].includes(body.deviceMode ?? 'auto') ? (body.deviceMode as string) : 'auto';

  // samples must live inside this job's samples folder
  const samples = (body.samples ?? []).filter(p => typeof p === 'string' && isUnder(context.samplesDir, p));
  if (samples.length === 0) return NextResponse.json({ error: 'No valid sample paths provided' }, { status: 400 });

  // references must be discovered first (also refreshes the selection prune)
  if (!(await readJsonFile(context.refsPath))) {
    const modelsDir = await getIdentityModelsDir();
    const discovery = await runIdentityWorker([
      '--command', 'refs',
      '--job-name', context.job.name,
      '--output-root', context.trainingFolder,
      '--device-mode', 'cpu',
      '--models-dir', modelsDir,
    ]);
    if (!discovery.ok) {
      return NextResponse.json(
        { error: `reference discovery failed: ${String(discovery.result ?? discovery.stderr).slice(0, 500)}` },
        { status: 500 },
      );
    }
  }

  const argsPath = path.join(context.identityDir, 'run_args.json');
  await writeJsonFile(argsPath, { samples });

  const modelsDir = await getIdentityModelsDir();
  const logPath = path.join(context.identityDir, 'run.log');
  try {
    const run = await spawnIdentityWorkerDetached(
      [
        '--command', 'run',
        '--job-name', context.job.name,
        '--output-root', context.trainingFolder,
        '--device-mode', deviceMode,
        '--sources', sources.join(','),
        '--models-dir', modelsDir,
        '--args-file', argsPath,
      ],
      logPath,
    );
    const state = {
      pid: run.pid,
      started_at: new Date().toISOString(),
      command: 'run',
      sources,
      device_mode: deviceMode,
      sample_count: samples.length,
    };
    await writeRunState(context.identityDir, state);
    return NextResponse.json({ started: true, state });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : String(error) }, { status: 500 });
  }
}
