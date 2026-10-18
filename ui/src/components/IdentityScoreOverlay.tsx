// One compact score line over a sample card: `A 0.420 · S 0.381` plus a
// status chip and the rank-disagreement flag. Rendered inside the card's
// image area (absolute, pointer-events-none) so it adds no card height.
import classNames from 'classnames';
import { overlayInfo, type IdentitySampleResult, type IdentitySource } from '@/utils/identityEval';

const STATUS_STYLES: Record<string, { dot: string; text: string }> = {
  evaluated: { dot: 'bg-emerald-500', text: 'text-white/90' },
  ambiguous: { dot: 'bg-amber-500', text: 'text-amber-100' },
  failed: { dot: 'bg-red-500', text: 'text-red-100' },
  subject_absent: { dot: 'bg-sky-500', text: 'text-sky-100' },
};

interface IdentityScoreOverlayProps {
  sample?: IdentitySampleResult;
  source: IdentitySource | 'combined';
  disagree?: boolean;
  running?: boolean;
}

export default function IdentityScoreOverlay({ sample, source, disagree, running }: IdentityScoreOverlayProps) {
  if (!sample) {
    if (!running) return null;
    return (
      <div className="pointer-events-none absolute inset-x-0 top-0 flex items-center gap-1 bg-black/45 px-[clamp(6px,4cqw,10px)] py-[clamp(2px,1.5cqw,4px)] text-[clamp(10px,6cqw,12px)] leading-tight text-white/80">
        <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-blue-400" />
        <span>pending</span>
      </div>
    );
  }
  const info = overlayInfo(sample, source);
  if (!info) return null;
  const style = STATUS_STYLES[sample.status] ?? STATUS_STYLES.evaluated;
  return (
    <div
      title={info.tooltip}
      className="pointer-events-none absolute inset-x-0 top-0 z-[1] flex items-center gap-1.5 bg-gradient-to-b from-black/60 to-black/10 px-[clamp(6px,4cqw,10px)] py-[clamp(2px,1.5cqw,4px)] text-[clamp(10px,6cqw,12px)] leading-tight [text-shadow:0_1px_2px_rgb(0_0_0/0.7)]"
    >
      <span className={classNames('inline-block h-1.5 w-1.5 shrink-0 rounded-full', style.dot)} />
      <span className={classNames('truncate font-medium tabular-nums', style.text)}>{info.label}</span>
      {disagree && (
        <span
          title="ArcFace and SFace rank this sample more than a third of the set apart"
          className="shrink-0 rounded-sm bg-purple-500/40 px-1 text-[0.85em] font-semibold uppercase tracking-wide text-purple-100"
        >
          A≠S
        </span>
      )}
    </div>
  );
}
