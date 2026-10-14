// Shared server-side plumbing for the identity-eval API routes: job path
// resolution, worker spawning (buffered for refs discovery, detached for
// evaluation runs), and the tiny JSON state files the worker and UI share
// under output/<job>/identity_eval/.
import { spawn } from 'child_process';
import { createHash } from 'crypto';
import path from 'path';
import fs from 'fs';
import fsp from 'fs/promises';
import prisma from './prisma';
import { TOOLKIT_ROOT } from '@/paths';
import { resolvePythonPath } from '../../cron/pythonPath';
import { getTrainingFolder } from './settings';
import { chainOn } from '@/utils/promiseChain';
import type { Job } from '@prisma/client';

export const WORKER_SCRIPT = path.join(TOOLKIT_ROOT, 'ui_scripts', 'identity_eval.py');
export const IDENTITY_MODELS_DIR_KEY = 'IDENTITY_MODELS_DIR';

export interface IdentityJobContext {
  job: Job;
  trainingFolder: string;
  jobDir: string;
  samplesDir: string;
  identityDir: string;
  externalDir: string;
  cacheDir: string;
  refsPath: string;
  resultsPath: string;
  progressPath: string;
  runStatePath: string;
  reportPath: string;
  categoriesPath: string;
}

export async function getIdentityJobContext(jobId: string): Promise<IdentityJobContext | null> {
  const job = await prisma.job.findUnique({ where: { id: jobId } });
  if (!job) return null;
  // Job names are unique and drive the folder layout; separators or ".."
  // would escape it (same guard as the samples route).
  if (/[\\/]/.test(job.name) || job.name === '..' || job.name === '.') return null;
  const trainingFolder = await getTrainingFolder();
  const jobDir = path.resolve(trainingFolder, job.name);
  const identityDir = path.join(jobDir, 'identity_eval');
  return {
    job,
    trainingFolder,
    jobDir,
    samplesDir: path.join(jobDir, 'samples'),
    identityDir,
    externalDir: path.join(identityDir, 'external_refs'),
    cacheDir: path.join(identityDir, 'emb_cache'),
    refsPath: path.join(identityDir, 'refs.json'),
    resultsPath: path.join(identityDir, 'results.json'),
    progressPath: path.join(identityDir, 'progress.json'),
    runStatePath: path.join(identityDir, 'run_state.json'),
    reportPath: path.join(identityDir, 'checkpoint_report.json'),
    categoriesPath: path.join(identityDir, 'prompt_categories.json'),
  };
}

export async function getIdentityModelsDir(): Promise<string> {
  const row = await prisma.settings.findFirst({ where: { key: IDENTITY_MODELS_DIR_KEY } });
  if (row?.value && row.value.trim() !== '') return row.value;
  return path.join(TOOLKIT_ROOT, 'models', 'face');
}

export interface WorkerRunResult {
  ok: boolean;
  exitCode: number | null;
  stdout: string;
  stderr: string;
  result: unknown;
  timedOut: boolean;
}

function parseWorkerResult(stdout: string): unknown {
  const lines = stdout.trimEnd().split(/\r?\n/);
  for (let i = lines.length - 1; i >= 0; i--) {
    const line = lines[i].trim();
    if (!line) continue;
    if (line.startsWith('{')) {
      try {
        return JSON.parse(line);
      } catch {
        return null;
      }
    }
    return null;
  }
  return null;
}

// Buffered worker run (refs discovery). Same spawn discipline as the
// scripts route: no shell, embedded python, unbuffered utf-8 stdout.
export function runIdentityWorker(args: string[], timeoutMs = 15 * 60 * 1000): Promise<WorkerRunResult> {
  return new Promise(resolve => {
    const child = spawn(resolvePythonPath(), ['-u', WORKER_SCRIPT, ...args], {
      cwd: TOOLKIT_ROOT,
      env: { ...process.env, PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8' },
      windowsHide: true,
    });
    let stdout = '';
    let stderr = '';
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      child.kill('SIGKILL');
    }, timeoutMs);
    child.stdout.on('data', (chunk: Buffer) => {
      stdout += chunk.toString('utf-8');
    });
    child.stderr.on('data', (chunk: Buffer) => {
      stderr += chunk.toString('utf-8');
    });
    child.on('error', err => {
      clearTimeout(timer);
      resolve({ ok: false, exitCode: null, stdout, stderr: stderr + String(err), result: null, timedOut });
    });
    child.on('close', code => {
      clearTimeout(timer);
      resolve({ ok: !timedOut && code === 0, exitCode: code, stdout, stderr, result: parseWorkerResult(stdout), timedOut });
    });
  });
}

export interface DetachedRun {
  pid: number;
  logPath: string;
}

// Long-lived evaluation run: detached so it survives the HTTP request and
// the panel being closed; stdout/stderr append to run.log for debugging.
// Cancel is a process kill; results.json is written incrementally by the
// worker so partial results stay valid. On Windows `detached: true` would
// give the child its own console window (a visible flash), and Windows
// children outlive their parent regardless -- so detach off-Windows only,
// keeping windowsHide everywhere.
export async function spawnIdentityWorkerDetached(args: string[], logPath: string): Promise<DetachedRun> {
  await fsp.mkdir(path.dirname(logPath), { recursive: true });
  const logFd = await fsp.open(logPath, 'a');
  const child = spawn(resolvePythonPath(), ['-u', WORKER_SCRIPT, ...args], {
    cwd: TOOLKIT_ROOT,
    env: { ...process.env, PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8' },
    windowsHide: true,
    detached: process.platform !== 'win32',
    stdio: ['ignore', logFd.fd, logFd.fd],
  });
  child.unref();
  // give the OS a beat, then surface spawn failures (bad python path etc.)
  await new Promise(resolve => setTimeout(resolve, 250));
  if (child.exitCode !== null) {
    await logFd.close();
    throw new Error(`identity eval worker exited immediately with code ${child.exitCode}; see ${logPath}`);
  }
  // close the fd in the parent; the child keeps its own handle
  logFd.close().catch(() => {});
  return { pid: child.pid ?? -1, logPath };
}

// ------------------------------------------------------------------ state IO

export interface IdentityRunState {
  pid: number;
  started_at: string;
  command: string;
  sources?: string[];
  device_mode?: string;
  sample_count?: number;
}

export async function readRunState(identityDir: string): Promise<IdentityRunState | null> {
  return readJsonFile<IdentityRunState>(path.join(identityDir, 'run_state.json'));
}

export async function writeRunState(identityDir: string, state: IdentityRunState): Promise<void> {
  await writeJsonFile(path.join(identityDir, 'run_state.json'), state);
}

export async function clearRunState(identityDir: string): Promise<void> {
  await fsp.rm(path.join(identityDir, 'run_state.json'), { force: true });
}

export function pidAlive(pid: number | null | undefined): boolean {
  if (!pid || pid <= 0) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch (err) {
    return (err as NodeJS.ErrnoException).code === 'EPERM';
  }
}

export async function readJsonFile<T>(filePath: string): Promise<T | null> {
  try {
    return JSON.parse(await fsp.readFile(filePath, 'utf-8')) as T;
  } catch {
    return null;
  }
}

export async function writeJsonFile(filePath: string, payload: unknown): Promise<void> {
  await fsp.mkdir(path.dirname(filePath), { recursive: true });
  const tmp = filePath + '.tmp';
  await fsp.writeFile(tmp, JSON.stringify(payload), 'utf-8');
  await fsp.rename(tmp, filePath);
}

// ------------------------------------------------------------------ selection

export interface SelectionDocument {
  version: number;
  training_refs: Record<string, unknown>;
  external_refs: Record<string, unknown>;
  sample_picks?: Record<string, unknown>;
}

// One chain per selection file: concurrent saves (e.g. picking subject faces
// on two samples in quick succession) must not interleave their
// read-merge-write -- without serialization both read the same prior state
// and the later write silently erases the earlier pick (atomic writes do not
// help; the lost update is logical). Server-side only; the refs worker also
// rewrites selection.json during discovery, which is a much rarer,
// user-triggered window (refresh) than pick saves.
const selectionTransactions = new Map<string, Promise<unknown>>();

export function mutateSelectionFile(
  selectionPath: string,
  mutate: (current: SelectionDocument) => SelectionDocument,
): Promise<SelectionDocument> {
  return chainOn(selectionTransactions, selectionPath, async () => {
    const current =
      (await readJsonFile<SelectionDocument>(selectionPath)) ??
      ({ version: 1, training_refs: {}, external_refs: {} } satisfies SelectionDocument);
    const next = mutate(current);
    await writeJsonFile(selectionPath, { ...next, updated_at: new Date().toISOString() });
    return next;
  });
}

export function isUnder(folder: string, target: string): boolean {
  const rel = path.relative(path.resolve(folder), path.resolve(target));
  return rel !== '' && !rel.startsWith('..') && !path.isAbsolute(rel);
}

export function sha256Hex(buffer: Buffer): string {
  return createHash('sha256').update(buffer).digest('hex');
}
