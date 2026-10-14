// GET  /api/identity-eval/compare?jobId= -- checkpoint comparison report
//       (Phase 2b). Serves the cached checkpoint_report.json while fresh
//       (results.json / prompt_categories.json not touched since it was
//       built); otherwise runs the compare worker (pure post-processing,
//       no face models, safe during training) and returns the new report.
// POST /api/identity-eval/compare        -- save prompt-index category tags
//       ({ "0": "solo", "1": "multi" }), then return the recomputed report.
import { NextRequest, NextResponse } from 'next/server';
import fsp from 'fs/promises';
import path from 'path';
import {
  getIdentityJobContext,
  readJsonFile,
  runIdentityWorker,
  writeJsonFile,
  type IdentityJobContext,
} from '@/server/identityEval';
import { reportFreshByMtimes } from '@/utils/identityCheckpoint';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export const maxDuration = 300;

async function mtime(filePath: string): Promise<number> {
  try {
    return (await fsp.stat(filePath)).mtimeMs;
  } catch {
    return -1;
  }
}

interface CompareReport {
  version?: number;
  steps?: unknown[];
  pairs?: unknown[];
}

// The report is stale when results, category tags, sample picks or the loss
// log changed after it was written (see reportFreshByMtimes for the WAL
// rationale), or when its version predates the current builder (v2 added
// p90 for the retention graph; v3 added the Phase 2d drift warnings; v4
// added Phase 2e leakage + subject-face pick resolution -- a cached older
// report must be rebuilt, not served).
const REPORT_VERSION = 4;

async function reportIsFresh(context: IdentityJobContext, report: CompareReport | null): Promise<boolean> {
  if (!report || report.version !== REPORT_VERSION) return false;
  const [resultsMtime, categoriesMtime, picksMtime, reportMtime, lossMtime, lossWalMtime] = await Promise.all([
    mtime(context.resultsPath),
    mtime(context.categoriesPath),
    mtime(path.join(context.identityDir, 'selection.json')),
    mtime(context.reportPath),
    mtime(path.join(context.jobDir, 'loss_log.db')),
    mtime(path.join(context.jobDir, 'loss_log.db-wal')),
  ]);
  return reportFreshByMtimes({
    results: resultsMtime,
    categories: categoriesMtime,
    picks: picksMtime,
    loss: lossMtime,
    lossWal: lossWalMtime,
    report: reportMtime,
  });
}

async function buildReport(context: IdentityJobContext): Promise<{ error: string | null }> {
  const result = await runIdentityWorker(
    ['--command', 'compare', '--job-name', context.job.name, '--output-root', context.trainingFolder],
    120 * 1000,
  );
  if (!result.ok) {
    return {
      error: result.timedOut
        ? 'checkpoint comparison timed out'
        : `checkpoint comparison failed: ${String(result.result ?? result.stderr).slice(0, 500)}`,
    };
  }
  return { error: null };
}

async function loadCategories(context: IdentityJobContext): Promise<Record<string, string>> {
  const document = await readJsonFile<{ categories?: Record<string, string> }>(context.categoriesPath);
  const raw = document?.categories;
  return raw && typeof raw === 'object' ? raw : {};
}

export async function GET(request: NextRequest) {
  const jobId = request.nextUrl.searchParams.get('jobId');
  if (!jobId) return NextResponse.json({ error: 'jobId is required' }, { status: 400 });
  const context = await getIdentityJobContext(jobId);
  if (!context) return NextResponse.json({ error: 'Job not found' }, { status: 400 });

  const noResults = (await mtime(context.resultsPath)) < 0;
  if (noResults) {
    // no evaluation yet: nothing to compare, but categories may still be edited
    return NextResponse.json({ report: null, categories: await loadCategories(context) });
  }

  let report = await readJsonFile<CompareReport>(context.reportPath);
  let error: string | null = null;
  if (!(await reportIsFresh(context, report))) {
    const built = await buildReport(context);
    error = built.error;
    report = error ? report : await readJsonFile<CompareReport>(context.reportPath);
  }
  if (!report && error) return NextResponse.json({ error }, { status: 500 });
  return NextResponse.json({ report, categories: await loadCategories(context), error });
}

interface CategoryUpdate {
  jobId: string;
  categories: Record<string, string>;
}

// Prompt-index keys, category strings trimmed and length-capped; empty
// values drop the tag. Mirrors what the compare worker reads back.
function sanitizeCategories(input: Record<string, string> | undefined): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [key, value] of Object.entries(input ?? {})) {
    if (!/^\d{1,4}$/.test(key)) continue;
    if (typeof value !== 'string') continue;
    const clean = value.trim().slice(0, 40);
    if (clean.length > 0) out[key] = clean;
  }
  return out;
}

export async function POST(request: Request) {
  let body: CategoryUpdate;
  try {
    body = (await request.json()) as CategoryUpdate;
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 });
  }
  if (!body?.jobId) return NextResponse.json({ error: 'jobId is required' }, { status: 400 });
  const context = await getIdentityJobContext(body.jobId);
  if (!context) return NextResponse.json({ error: 'Job not found' }, { status: 404 });

  const categories = sanitizeCategories(body.categories);
  await writeJsonFile(context.categoriesPath, {
    version: 1,
    updated_at: new Date().toISOString(),
    categories,
  });

  if ((await mtime(context.resultsPath)) < 0) {
    return NextResponse.json({ categories, report: null });
  }
  const built = await buildReport(context);
  const report = await readJsonFile<CompareReport>(context.reportPath);
  if (built.error) return NextResponse.json({ error: built.error, categories, report }, { status: 500 });
  return NextResponse.json({ categories, report });
}
