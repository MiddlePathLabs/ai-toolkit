import { NextRequest, NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';
import { resolvePythonPath } from '@/server/admission';

export const runtime = 'nodejs';
export const maxDuration = 125;

type BridgeResponse = { ok: true; result: Record<string, unknown> } | { ok: false; error: string };

function runBridge(payload: unknown): Promise<BridgeResponse> {
  const python = resolvePythonPath();
  const root = path.resolve('@', '..', '..');
  let resolve!: (result: BridgeResponse) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<BridgeResponse>((onResolve, onReject) => {
    resolve = onResolve;
    reject = onReject;
  });
  const child = spawn(python, [path.join(root, 'ui_scripts', 'identity_isolation.py')], {
    cwd: root,
    stdio: ['pipe', 'pipe', 'pipe'],
    windowsHide: true,
    shell: false,
  });
  let stdout = '';
  let stderr = '';
  let settled = false;
  const finish = (error?: Error, result?: BridgeResponse) => {
    if (settled) return;
    settled = true;
    clearTimeout(timer);
    if (error) reject(error);
    else resolve(result!);
  };
  const timer = setTimeout(() => {
    child.kill();
    finish(new Error('Identity Isolation inspection timed out after 120 seconds. No training was started.'));
  }, 120000);
  child.stdout.on('data', chunk => {
    stdout += chunk.toString();
  });
  child.stderr.on('data', chunk => {
    stderr += chunk.toString();
  });
  child.once('error', error => finish(error));
  child.stdin.on('error', error => finish(error));
  child.once('close', code => {
    if (settled) return;
    try {
      const parsed = JSON.parse(stdout);
      if (
        !parsed ||
        typeof parsed !== 'object' ||
        !(
          (parsed.ok === true && parsed.result && typeof parsed.result === 'object') ||
          (parsed.ok === false && typeof parsed.error === 'string')
        )
      ) {
        throw new Error('Identity Isolation returned an invalid response.');
      }
      if (code !== 0 && parsed.ok) throw new Error(stderr.trim() || `Identity Isolation exited with ${code}.`);
      finish(undefined, parsed);
    } catch (error) {
      finish(
        new Error(stderr.trim() || (error instanceof Error ? error.message : 'Identity Isolation returned no JSON.')),
      );
    }
  });
  child.stdin.end(JSON.stringify(payload));
  return promise;
}

export async function POST(request: NextRequest) {
  let payload: unknown;
  try {
    payload = await request.json();
  } catch {
    return NextResponse.json({ ok: false, error: 'Request must be valid JSON.' }, { status: 400 });
  }
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    return NextResponse.json({ ok: false, error: 'Request must contain action and process.' }, { status: 400 });
  }
  const body = payload as Record<string, unknown>;
  if (
    !['caption', 'inspect', 'prepare'].includes(String(body.action)) ||
    !body.process ||
    typeof body.process !== 'object' ||
    Array.isArray(body.process) ||
    (body.caption !== undefined && typeof body.caption !== 'string')
  ) {
    return NextResponse.json(
      {
        ok: false,
        error: 'Use action caption, inspect, or prepare; process must be an object and caption must be text.',
      },
      { status: 400 },
    );
  }
  try {
    const result = await runBridge({ action: body.action, process: body.process, caption: body.caption });
    return NextResponse.json(result, { status: result.ok ? 200 : 422 });
  } catch (error) {
    return NextResponse.json(
      {
        ok: false,
        error: `Identity Isolation is unavailable: ${error instanceof Error ? error.message : String(error)}`,
      },
      { status: 503 },
    );
  }
}
