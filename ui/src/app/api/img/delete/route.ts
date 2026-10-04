import { NextResponse } from 'next/server';
import { getDatasetsRoot, getTrainingFolder } from '@/server/settings';

import { deleteSampleFile, SampleFileError } from '@/server/sampleEvidence';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { imgPath } = body;
    await deleteSampleFile(imgPath, await getDatasetsRoot(), await getTrainingFolder());

    return NextResponse.json({ success: true });
  } catch (error) {
    const status = error instanceof SampleFileError ? error.status : 500;
    const message = error instanceof SampleFileError ? error.message : 'Failed to delete image';
    return NextResponse.json({ error: message }, { status });
  }
}
