import { NextRequest, NextResponse } from 'next/server';

// Lists the models an OpenAI-compatible endpoint serves (GET {base}/models), so the
// caption modal can offer them. Done server side: local servers rarely send CORS headers.
export async function POST(request: NextRequest) {
  let body: any;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: 'invalid body' }, { status: 400 });
  }
  let base = String(body?.api_base_url || '').trim().replace(/\/+$/, '');
  if (!base) {
    return NextResponse.json({ error: 'api_base_url is required' }, { status: 400 });
  }
  if (!/^https?:\/\//i.test(base)) base = `http://${base}`;
  base = base.replace(/\/chat\/completions$/, '');
  let url: URL;
  try {
    url = new URL(base);
  } catch {
    return NextResponse.json({ error: 'invalid api_base_url' }, { status: 400 });
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    return NextResponse.json({ error: 'only http and https URLs are allowed' }, { status: 400 });
  }
  if (url.pathname === '' || url.pathname === '/') base = `${base}/v1`;

  // Only a key the caller typed is sent. The server's AITK_CAPTION_API_KEY is never
  // used here, so this route cannot forward it to an arbitrary host.
  const headers: Record<string, string> = {};
  const apiKey = String(body?.api_key || '').trim();
  if (apiKey) headers['Authorization'] = `Bearer ${apiKey}`;
  try {
    const res = await fetch(`${base}/models`, {
      headers,
      redirect: 'manual',
      signal: AbortSignal.timeout(10000),
    });
    if (res.status >= 300 && res.status < 400) {
      return NextResponse.json({ error: `endpoint redirected (HTTP ${res.status}); use the final URL` }, { status: 502 });
    }
    if (!res.ok) {
      return NextResponse.json({ error: `endpoint returned HTTP ${res.status}` }, { status: 502 });
    }
    const data = await res.json();
    const list = Array.isArray(data?.data) ? data.data : Array.isArray(data?.models) ? data.models : [];
    const models: string[] = list
      .map((m: any) => (typeof m === 'string' ? m : m?.id || m?.name || m?.model))
      .filter((m: any) => typeof m === 'string' && m.length > 0);
    return NextResponse.json({ models });
  } catch (e: any) {
    const reason = e?.name === 'TimeoutError' ? 'timed out' : 'could not connect or response was not JSON';
    return NextResponse.json({ error: reason }, { status: 502 });
  }
}
