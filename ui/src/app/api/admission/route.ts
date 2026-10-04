import { NextResponse } from 'next/server';
import { AdmissionUnavailableError, rawYamlDiagnostic, runAdmission } from '@/server/admission';

export const runtime = 'nodejs';

export async function POST(request: Request) {
  let body: { config?: unknown; raw_yaml?: string };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: 'Request body must be JSON.' }, { status: 400 });
  }

  if (typeof body.raw_yaml === 'string') {
    const syntax = rawYamlDiagnostic(body.raw_yaml);
    if (syntax) return NextResponse.json(syntax, { status: 422 });
  }

  try {
    const result = await runAdmission(body.config, 'api');
    return NextResponse.json(result, { status: result.valid ? 200 : 422 });
  } catch (error) {
    if (error instanceof AdmissionUnavailableError) {
      return NextResponse.json(
        {
          error: 'Canonical admission is unavailable. The configuration was not validated.',
          code: 'admission_unavailable',
        },
        { status: 503 },
      );
    }
    console.error(error);
    return NextResponse.json({ error: 'Canonical admission failed.' }, { status: 500 });
  }
}
