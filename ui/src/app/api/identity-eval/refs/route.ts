// GET  /api/identity-eval/refs?jobId=  -- reference state (discovered refs,
//       per-image face status, models, saved selection). Runs discovery on
//       first open and when ?refresh=1; otherwise serves the cached refs.json.
// POST /api/identity-eval/refs        -- multipart: add external reference
//       images (content-addressed, deduped). JSON: save selection updates
//       (exclusions, face picks) or remove external refs; removal re-runs
//       discovery so statuses stay in sync.
import { NextRequest, NextResponse } from 'next/server';
import fsp from 'fs/promises';
import path from 'path';
import {
  getIdentityJobContext,
  getIdentityModelsDir,
  readJsonFile,
  runIdentityWorker,
  writeJsonFile,
  sha256Hex,
  type IdentityJobContext,
} from '@/server/identityEval';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export const maxDuration = 1200;

const REF_EXTENSIONS = new Set(['.jpg', '.jpeg', '.png', '.webp']);

async function runDiscovery(context: IdentityJobContext) {
  const modelsDir = await getIdentityModelsDir();
  const result = await runIdentityWorker([
    '--command', 'refs',
    '--job-name', context.job.name,
    '--output-root', context.trainingFolder,
    '--device-mode', 'cpu', // discovery is light and must never fight a training run
    '--models-dir', modelsDir,
  ]);
  if (!result.ok) {
    const message = result.timedOut
      ? 'reference discovery timed out'
      : `reference discovery failed: ${String(result.result ?? result.stderr).slice(0, 500)}`;
    return { error: message as string };
  }
  return { error: null };
}

async function loadState(context: IdentityJobContext) {
  const refs = await readJsonFile<Record<string, unknown>>(context.refsPath);
  const selection = await readJsonFile<Record<string, unknown>>(path.join(context.identityDir, 'selection.json'));
  return { refs, selection: selection ?? { version: 1, training_refs: {}, external_refs: {} } };
}

export async function GET(request: NextRequest) {
  const jobId = request.nextUrl.searchParams.get('jobId');
  const refresh = request.nextUrl.searchParams.get('refresh') === '1';
  if (!jobId) return NextResponse.json({ error: 'jobId is required' }, { status: 400 });
  const context = await getIdentityJobContext(jobId);
  if (!context) return NextResponse.json({ error: 'Job not found' }, { status: 404 });

  let state = await loadState(context);
  if (refresh || !state.refs) {
    const { error } = await runDiscovery(context);
    if (error) {
      // serve any previous state along with the error so the panel stays usable
      const previous = state.refs ? state : await loadState(context);
      return NextResponse.json({ error, ...previous }, { status: 500 });
    }
    state = await loadState(context);
  }
  return NextResponse.json(state);
}

interface SelectionUpdate {
  jobId: string;
  trainingRefs?: Record<string, { excluded?: boolean; face_index?: number | null }>;
  externalRefs?: Record<string, { excluded?: boolean; face_index?: number | null }>;
  // Phase 2e: per-generated-sample subject-face picks + user twin marks,
  // keyed by sample filename (never a path -- no traversal surface)
  samplePicks?: Record<string, { face_index?: number | null; twin_confirmed?: number[] }>;
  removeExternal?: string[];
}

export async function POST(request: NextRequest) {
  const contentType = request.headers.get('content-type') ?? '';

  if (contentType.includes('multipart/form-data')) {
    const form = await request.formData();
    const jobId = String(form.get('jobId') ?? '');
    const context = await getIdentityJobContext(jobId);
    if (!context) return NextResponse.json({ error: 'Job not found' }, { status: 404 });
    const files = form.getAll('files').filter((f): f is File => f instanceof File);
    if (files.length === 0) return NextResponse.json({ error: 'No files provided' }, { status: 400 });
    await fsp.mkdir(context.externalDir, { recursive: true });
    const saved: string[] = [];
    const duplicates: string[] = [];
    for (const file of files) {
      const buffer = Buffer.from(await file.arrayBuffer());
      if (buffer.length === 0) continue;
      const ext = path.extname(file.name).toLowerCase();
      const name = `${sha256Hex(buffer)}${REF_EXTENSIONS.has(ext) ? ext : '.jpg'}`;
      const target = path.join(context.externalDir, name);
      try {
        await fsp.access(target);
        duplicates.push(name);
      } catch {
        await fsp.writeFile(target, buffer);
      }
      saved.push(name);
    }
    const { error } = await runDiscovery(context);
    const state = await loadState(context);
    if (error) return NextResponse.json({ error, ...state }, { status: 500 });
    return NextResponse.json({ added: saved, duplicates, ...state });
  }

  let body: SelectionUpdate;
  try {
    body = (await request.json()) as SelectionUpdate;
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 });
  }
  if (!body?.jobId) return NextResponse.json({ error: 'jobId is required' }, { status: 400 });
  const context = await getIdentityJobContext(body.jobId);
  if (!context) return NextResponse.json({ error: 'Job not found' }, { status: 404 });

  if (Array.isArray(body.removeExternal) && body.removeExternal.length > 0) {
    const removed: string[] = [];
    for (const name of body.removeExternal) {
      if (!/^[a-f0-9]{64}\.(jpg|jpeg|png|webp)$/i.test(name)) continue; // sha256 names only, no traversal
      try {
        await fsp.rm(path.join(context.externalDir, name));
        removed.push(name);
      } catch {
        // already gone
      }
    }
    const { error } = await runDiscovery(context);
    const state = await loadState(context);
    if (error) return NextResponse.json({ error, ...state }, { status: 500 });
    return NextResponse.json({ removed, ...state });
  }

  // selection update: merge over the current selection file, then persist
  const selectionPath = path.join(context.identityDir, 'selection.json');
  const current = (await readJsonFile<{
    version: number;
    training_refs: Record<string, unknown>;
    external_refs: Record<string, unknown>;
    sample_picks?: Record<string, unknown>;
  }>(selectionPath)) ?? { version: 1, training_refs: {}, external_refs: {} };

  const sanitize = (entries: Record<string, { excluded?: boolean; face_index?: number | null }> | undefined) => {
    const out: Record<string, { excluded?: boolean; face_index?: number | null }> = {};
    for (const [key, value] of Object.entries(entries ?? {})) {
      if (!value || typeof value !== 'object') continue;
      const clean: { excluded?: boolean; face_index?: number | null } = {};
      if (typeof value.excluded === 'boolean') clean.excluded = value.excluded;
      if (value.face_index === null || (Number.isInteger(value.face_index) && (value.face_index as number) >= 0 && (value.face_index as number) <= 32)) {
        clean.face_index = value.face_index as number | null;
      }
      if (Object.keys(clean).length > 0) out[key] = clean;
    }
    return out;
  };

  // Sample picks are keyed by bare filename; reject anything carrying a
  // path (the UI only sends basename keys, but the API must not trust that).
  const sanitizeSamplePicks = (entries: SelectionUpdate['samplePicks'] | undefined) => {
    const out: NonNullable<SelectionUpdate['samplePicks']> = {};
    for (const [key, value] of Object.entries(entries ?? {})) {
      if (!/^[^\\/:*?"<>|]{1,200}$/.test(key)) continue;
      if (!value || typeof value !== 'object') continue;
      const clean: { face_index?: number | null; twin_confirmed?: number[] } = {};
      if (value.face_index === null || (Number.isInteger(value.face_index) && (value.face_index as number) >= 0 && (value.face_index as number) <= 32)) {
        clean.face_index = value.face_index as number | null;
      }
      if (Array.isArray(value.twin_confirmed)) {
        const marks = [...new Set(value.twin_confirmed.filter((n): n is number => Number.isInteger(n) && n >= 0 && n <= 32))].sort((a, b) => a - b);
        if (marks.length > 0) clean.twin_confirmed = marks;
      }
      if (Object.keys(clean).length > 0) out[key] = clean;
    }
    return out;
  };

  current.version = 1;
  current.training_refs = { ...current.training_refs, ...sanitize(body.trainingRefs) };
  current.external_refs = { ...current.external_refs, ...sanitize(body.externalRefs) };
  current.sample_picks = { ...(current.sample_picks ?? {}), ...sanitizeSamplePicks(body.samplePicks) };
  await writeJsonFile(selectionPath, { ...current, updated_at: new Date().toISOString() });
  return NextResponse.json({ selection: current });
}
