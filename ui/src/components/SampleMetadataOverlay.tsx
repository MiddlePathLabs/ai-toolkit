import type { SampleMetadata } from '@/utils/sampleImages';

// Bottom scrim instead of a boxed panel: text sits on a soft gradient so small thumbnails stay
// readable without a dark block covering the subject. Font size follows the card width (cqw,
// clamped) and the fields wrap rather than truncate. Cards opt into hide-on-hover via the
// `group/sample` class so the full image can be inspected.
export default function SampleMetadataOverlay({ metadata }: { metadata: SampleMetadata }) {
  return (
    <div
      className="pointer-events-none absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/70 via-black/35 to-transparent px-[clamp(6px,4cqw,10px)] pb-[clamp(4px,3cqw,8px)] pt-[clamp(14px,10cqw,28px)] text-[clamp(10px,6.5cqw,12px)] leading-tight text-white/90 transition-opacity duration-150 [text-shadow:0_1px_2px_rgb(0_0_0/0.7)] group-hover/sample:opacity-0"
    >
      <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5 tabular-nums">
        {metadata.isRaw && (
          <span className="rounded-sm bg-white/20 px-1 text-[0.85em] font-semibold uppercase tracking-wide">Raw</span>
        )}
        <span>
          <span className="text-white/60">Step</span> {metadata.trainingStep?.toLocaleString() ?? '?'}
        </span>
        <span className="break-all">
          <span className="text-white/60">Seed</span> {metadata.seed ?? '?'}
        </span>
        <span>
          <span className="text-white/60">Infer</span> {metadata.inferenceSteps ?? '?'}
        </span>
      </div>
    </div>
  );
}
