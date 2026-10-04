import type { SampleMetadata } from '@/utils/sampleImages';

export default function SampleMetadataOverlay({ metadata }: { metadata: SampleMetadata }) {
  return (
    <div className="pointer-events-none absolute bottom-2 left-2 max-w-[calc(100%-1rem)] rounded bg-gray-950/90 px-2 py-1 text-xs leading-5 text-gray-100">
      {metadata.isRaw && <div className="font-semibold">Raw</div>}
      <div>
        <span className="text-gray-300">Training step:</span> {metadata.trainingStep?.toLocaleString() ?? 'Unknown'}
      </div>
      <div className="break-all">
        <span className="text-gray-300">Seed:</span> {metadata.seed ?? 'Unknown'}
      </div>
      <div>
        <span className="text-gray-300">Steps:</span> {metadata.inferenceSteps ?? 'Unknown'}
      </div>
    </div>
  );
}
