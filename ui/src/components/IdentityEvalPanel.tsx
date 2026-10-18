// Identity evaluation side panel (fork-only feature).
//
// Source toggle, reference management (exclude / face picks / external
// upload), device selector, run/cancel with progress, models indicator and
// the per-step summary. Opened from the Samples toolbar; does not change
// the sample grid when unused.
import { useEffect, useMemo, useRef, useState } from 'react';
import classNames from 'classnames';
import { LuLoader, LuRefreshCw, LuX } from 'react-icons/lu';
import { FaPlay, FaStop, FaTrash } from 'react-icons/fa';
import type { Job } from '@prisma/client';
import type { IdentityEvalApi } from '@/hooks/useIdentityEval';
import {
  absentPickFor,
  absenceVerdict,
  classifyFaceLevel,
  groupSummariesByStep,
  normalizeRefPath,
  samplePickFor,
  selectionForImage,
  twinMarked,
  type IdentityFace,
  type IdentityRefImage,
  type IdentitySampleResult,
  type IdentitySelection,
  type IdentitySource,
} from '@/utils/identityEval';
import { leakLevelLabel } from '@/utils/identityCheckpoint';
import { encodeFilePathForUrl } from '@/utils/basic';
import IdentityCheckpointSection from './IdentityCheckpointSection';
import { openConfirm } from './ConfirmModal';

const controlClass =
  'rounded border border-gray-300 bg-white px-2 py-1 text-sm text-gray-900 dark:border-gray-600 dark:bg-gray-900 dark:text-gray-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-blue-500';

const STATUS_BADGE: Record<string, string> = {
  ok: 'border-emerald-400/50 text-emerald-700 dark:text-emerald-300',
  needs_review: 'border-amber-400/60 text-amber-700 dark:text-amber-300',
  no_face: 'border-gray-400/50 text-gray-600 dark:text-gray-400',
  error: 'border-red-400/60 text-red-600 dark:text-red-400',
};

function imageUrl(path: string): string {
  return `/api/img/${encodeFilePathForUrl(path)}`;
}

// CSS face crop: the image is scaled so the bbox (plus a little margin)
// covers the box and is centred on the bbox centre -- a real crop view
// without server-side thumbnails. `max-w-none` is load-bearing: Tailwind's
// preflight clamps imgs to 100% of the container, which squeezed the
// scaled-up image horizontally into the box (the "squished face" bug).
// A "no preview" state covers images that fail to load.
function FaceCropThumb({ src, bbox, size = 72, margin = 0.25 }: { src: string; bbox: number[]; size?: number; margin?: number }) {
  const [natural, setNatural] = useState<{ w: number; h: number } | null>(null);
  const [failed, setFailed] = useState(false);
  const [x1, y1, x2, y2] = bbox;
  const w = Math.max(x2 - x1, 1);
  const h = Math.max(y2 - y1, 1);
  // cover-fill with a zoom cap so tiny distant faces don't dissolve into
  // pixels (a capped face letterboxes a hair instead of over-zooming)
  const scale = natural ? Math.min(Math.max(size / w, size / h) * (1 + margin), 4) : 1;
  return (
    <div
      className="relative overflow-hidden rounded border border-gray-300 bg-gray-900 dark:border-gray-600"
      style={{ width: size, height: size }}
    >
      {failed ? (
        <span className="absolute inset-0 flex items-center justify-center px-1 text-center text-[9px] leading-tight text-gray-500 dark:text-gray-400">
          no preview
        </span>
      ) : (
        natural && (
          <img
            src={imageUrl(src)}
            alt="face crop"
            className="absolute max-w-none origin-top-left"
            style={{
              width: natural.w * scale,
              height: natural.h * scale,
              left: size / 2 - (x1 + w / 2) * scale,
              top: size / 2 - (y1 + h / 2) * scale,
            }}
          />
        )
      )}
      <img
        src={imageUrl(src)}
        alt=""
        className="hidden"
        onLoad={e => {
          const img = e.target as HTMLImageElement;
          setNatural({ w: img.naturalWidth, h: img.naturalHeight });
        }}
        onError={() => setFailed(true)}
      />
    </div>
  );
}

// Clickable ArcFace score chip for the highest/lowest sample links: opens the
// sample, tooltip carries both models' scores.
function SampleScoreChip({
  sample,
  source,
  tone,
  onOpen,
}: {
  sample: IdentitySampleResult;
  source: IdentitySource | 'combined';
  tone: 'high' | 'low';
  onOpen: (path: string) => void;
}) {
  const arcface = sample.primary?.arcface?.[source]?.mean;
  const sface = sample.primary?.sface?.[source]?.mean;
  const scoreBits = [
    arcface != null ? `ArcFace ${arcface.toFixed(3)}` : null,
    sface != null ? `SFace ${sface.toFixed(3)}` : null,
  ].filter(Boolean);
  return (
    <button
      type="button"
      className={classNames(
        'rounded border px-1.5 py-0.5 tabular-nums hover:bg-gray-100 dark:hover:bg-gray-800',
        tone === 'high'
          ? 'border-emerald-400/60 text-emerald-700 dark:text-emerald-300'
          : 'border-red-400/50 text-red-600 dark:text-red-400',
      )}
      onClick={() => onOpen(sample.path)}
      title={`Step ${sample.step?.toLocaleString()} · prompt ${sample.prompt_idx}\n${scoreBits.join(' · ')}\nclick to open`}
    >
      {arcface != null ? arcface.toFixed(3) : sface != null ? `S ${sface.toFixed(3)}` : '–'}
    </button>
  );
}

function RefThumb({ image, selected, onClick }: { image: IdentityRefImage; selected: boolean; onClick: () => void }) {
  const label =
    image.status === 'ok'
      ? 'single face — included by default'
      : image.status === 'needs_review'
        ? 'multiple faces — needs a face pick before it can be used'
        : image.status === 'no_face'
          ? 'no face detected — excluded'
          : image.error ?? 'unreadable';
  return (
    <button
      type="button"
      onClick={onClick}
      title={`${image.name}\n${label}`}
      className={classNames(
        'relative block h-14 w-14 shrink-0 overflow-hidden rounded border-2 transition',
        selected ? 'border-blue-500' : 'border-transparent opacity-100',
        image.status === 'needs_review' && !selected && 'ring-2 ring-amber-500/70',
        image.status === 'no_face' || image.status === 'error' ? 'grayscale' : '',
      )}
    >
      <img src={imageUrl(image.path)} alt={image.name} className="h-full w-full object-cover" />
    </button>
  );
}

function RefSection({
  title,
  images,
  source,
  selection,
  onToggleExclude,
  onPickFace,
  onRemove,
}: {
  title: string;
  images: IdentityRefImage[];
  source: IdentitySource;
  selection: IdentityEvalApi['selection'];
  onToggleExclude: (image: IdentityRefImage, excluded: boolean) => void;
  onPickFace: (image: IdentityRefImage, faceIndex: number) => void;
  onRemove?: (image: IdentityRefImage) => void;
}) {
  const [openImage, setOpenImage] = useState<IdentityRefImage | null>(null);
  const counts = useMemo(() => {
    const acc = { ok: 0, needs_review: 0, no_face: 0, error: 0 };
    for (const image of images) acc[image.status] = (acc[image.status] ?? 0) + 1;
    return acc;
  }, [images]);
  if (images.length === 0) {
    return (
      <div className="rounded border border-dashed border-gray-300 p-2 text-xs text-gray-500 dark:border-gray-600 dark:text-gray-400">
        {title}: none
      </div>
    );
  }
  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between text-xs">
        <span className="font-semibold text-gray-900 dark:text-gray-100">{title}</span>
        <span className="text-gray-500 dark:text-gray-400">
          {counts.ok} ok · {counts.needs_review} need review · {counts.no_face + counts.error} excluded
        </span>
      </div>
      <div className="flex flex-wrap gap-1">
        {images.map(image => (
          <RefThumb key={image.path} image={image} selected={openImage?.path === image.path} onClick={() => setOpenImage(openImage?.path === image.path ? null : image)} />
        ))}
      </div>
      {openImage && (
        <div className="rounded border border-gray-300 bg-gray-50 p-2 text-xs dark:border-gray-600 dark:bg-gray-900">
          <div className="mb-2 break-all text-gray-600 dark:text-gray-300">{openImage.name}</div>
          {openImage.status === 'needs_review' && (
            <div className="mb-2 rounded bg-amber-500/10 p-1.5 text-amber-700 dark:text-amber-300">
              Multiple faces: pick the subject&apos;s face. Until then this reference is excluded — the largest face is
              never auto-picked.
            </div>
          )}
          <div className="flex flex-wrap items-center gap-2">
            {openImage.faces.map(face => {
              const current = selectionForImage(selection ?? null, openImage, source);
              const picked =
                current.face_index === face.index ||
                (current.face_index == null && openImage.faces.length === 1 && face.index === 0);
              return (
                <button
                  key={face.index}
                  type="button"
                  onClick={() => onPickFace(openImage, face.index)}
                  className={classNames('rounded p-0 transition', picked ? 'ring-2 ring-blue-500' : 'opacity-80 hover:opacity-100')}
                  title={`Use face ${face.index + 1} (det ${face.det_score.toFixed(2)})`}
                >
                  <FaceCropThumb src={openImage.path} bbox={face.bbox} />
                </button>
              );
            })}
            {openImage.faces.length === 0 && (
              <span className="text-gray-500 dark:text-gray-400">{openImage.error ?? 'No detectable face.'}</span>
            )}
          </div>
          {openImage.faces.length > 0 && (
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <label className="flex items-center gap-1.5">
                <input
                  type="checkbox"
                  className="h-3.5 w-3.5 accent-blue-500"
                  checked={Boolean(selectionForImage(selection ?? null, openImage, source).excluded)}
                  onChange={e => onToggleExclude(openImage, e.target.checked)}
                />
                Exclude this reference
              </label>
              {onRemove && (
                <button type="button" className={controlClass} onClick={() => onRemove(openImage)}>
                  <FaTrash className="mr-1 inline" size={10} /> Remove
                </button>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// Phase 2e: multi-person sample picker + leakage inspection. Picking the
// subject face resolves an "ambiguous" sample instantly (per-face scores
// already exist); non-subject faces show their leakage level per model
// against that model's own LOO band. Levels are informational — the tool's
// strongest wording is "possible duplicate"; "user-marked twin" is your own
// mark, never the tool's claim.
function MultiPersonSamples({
  samples,
  picks,
  trainingRefs,
  externalRefs,
  results,
  source,
  onSavePick,
  onMarkAbsent,
  onToggleTwin,
  onOpenSample,
}: {
  samples: IdentitySampleResult[];
  picks: IdentitySelection['sample_picks'] | null | undefined;
  trainingRefs: IdentityRefImage[];
  externalRefs: IdentityRefImage[];
  results: IdentityEvalApi['results'];
  source: IdentitySource | 'combined';
  onSavePick: (sample: IdentitySampleResult, faceIndex: number | null) => void;
  onMarkAbsent: (sample: IdentitySampleResult, absent: boolean) => void;
  onToggleTwin: (sample: IdentitySampleResult, faceIndex: number) => void;
  onOpenSample: (path: string) => void;
}) {
  const multi = useMemo(
    () =>
      samples
        .filter(s => s.faces.length >= 2)
        .sort((a, b) => (b.step ?? 0) - (a.step ?? 0) || a.filename.localeCompare(b.filename)),
    [samples],
  );
  const [openFile, setOpenFile] = useState<string | null>(null);
  // reference lookup for the closest-reference crop: training keys are the
  // canonical path form (normalizeRefPath), external keys are content names
  const refByKey = useMemo(() => {
    const map = new Map<string, IdentityRefImage>();
    for (const image of trainingRefs) map.set(normalizeRefPath(image.path), image);
    for (const image of externalRefs) map.set(image.name, image);
    return map;
  }, [trainingRefs, externalRefs]);
  if (multi.length === 0) return null;

  const loo = (model: string): number | null => results?.refsets?.[source]?.loo?.[model] ?? null;
  // per-model leakage chips for one non-subject face (current source only)
  const levelsFor = (face: IdentityFace): { model: 'arcface' | 'sface'; level: 'resembles' | 'likely_duplicate'; mean: number; band: number }[] => {
    const out: { model: 'arcface' | 'sface'; level: 'resembles' | 'likely_duplicate'; mean: number; band: number }[] = [];
    for (const model of ['arcface', 'sface'] as const) {
      const mean = face.scores?.[model]?.[source]?.mean ?? null;
      const band = loo(model);
      const level = classifyFaceLevel(mean, band);
      if (level && mean != null && band != null) out.push({ model, level, mean, band });
    }
    return out;
  };
  // the closest reference crop next to a non-subject face (results v3+;
  // hidden for older evaluations)
  const closestRefOf = (face: IdentityFace): { image: IdentityRefImage; bbox: number[]; sim: number } | null => {
    for (const model of ['arcface', 'sface'] as const) {
      const entry = face.closest_ref?.[model]?.[source];
      if (entry?.key) {
        const image = refByKey.get(entry.key);
        const refFace = image?.faces[entry.face_index ?? 0];
        if (image && refFace) return { image, bbox: refFace.bbox, sim: entry.sim };
      }
    }
    return null;
  };

  return (
    <div className="space-y-2 rounded border border-gray-300 p-2 dark:border-gray-600">
      <div className="flex items-baseline justify-between">
        <span className="text-xs font-semibold" title="Multi-face samples: pick which face is the intended subject — the sample then counts as evaluated (its picked face joins the summaries). Other faces are checked against the reference band for duplication of the subject.">
          Multi-person samples
        </span>
        <span className="text-xs text-gray-500 dark:text-gray-400">
          {multi.length} · {multi.filter(s => samplePickFor(s, picks)).length} picked
        </span>
      </div>
      <p className="text-[10px] text-gray-500 dark:text-gray-400">
        Click the intended subject&apos;s face. &quot;possible duplicate&quot; = that face scores at or above the LOO band (as close to
        your references as they are to each other); &quot;resembles subject&quot; = within {0.1.toFixed(1)} below it — informational. Levels
        are per model; mark a twin yourself when you see one at full resolution.
      </p>
      <div className="space-y-1.5">
        {multi.map(sample => {
          const pick = samplePickFor(sample, picks);
          const absent = absentPickFor(sample, picks) !== null;
          const subject = absent ? null : (pick?.face_index ?? null);
          const open = openFile === sample.filename;
          // "flagged" keeps the checkpoint rate's meaning (possible duplicate
          // or user twin mark); resembles-level faces get their own count
          const nonSubject = subject == null ? sample.faces : sample.faces.filter(f => f.index !== subject);
          const flagged = nonSubject.filter(
            f => levelsFor(f).some(l => l.level === 'likely_duplicate') || (pick ? twinMarked(pick, f.index) : false),
          ).length;
          const resembles = nonSubject.filter(
            f => !levelsFor(f).some(l => l.level === 'likely_duplicate') && !(pick ? twinMarked(pick, f.index) : false) && levelsFor(f).length > 0,
          ).length;
          return (
            <div key={sample.filename} className="rounded border border-gray-200 dark:border-gray-700">
              <button
                type="button"
                className="flex w-full items-center gap-2 px-1.5 py-1 text-left text-xs hover:bg-gray-100 dark:hover:bg-gray-800"
                onClick={() => setOpenFile(open ? null : sample.filename)}
              >
                {sample.faces[0] && <FaceCropThumb src={sample.path} bbox={sample.faces[0].bbox} size={32} />}
                <span className="min-w-0 flex-1 truncate text-gray-700 dark:text-gray-200" title={sample.filename}>
                  step {sample.step?.toLocaleString() ?? '?'} · prompt {sample.prompt_idx ?? '?'}
                  {sample.is_raw ? ' · raw' : ''}
                </span>
                {absent ? (
                  <span
                    className="shrink-0 rounded border border-sky-400/60 px-1 text-[10px] text-sky-700 dark:text-sky-300"
                    title="You marked the subject as not in this picture — the sample is excluded from likeness summaries and checked for trigger adherence instead"
                  >
                    no subject · excluded
                  </span>
                ) : subject == null ? (
                  <span className="shrink-0 rounded border border-amber-400/60 px-1 text-[10px] text-amber-700 dark:text-amber-300">needs pick</span>
                ) : (
                  <span className="shrink-0 rounded border border-emerald-400/50 px-1 text-[10px] text-emerald-700 dark:text-emerald-300">
                    picked{flagged > 0 ? ` · ${flagged} flagged` : resembles > 0 ? ` · ${resembles} resembles` : ''}
                  </span>
                )}
              </button>
              {open && (
                <div className="space-y-1.5 border-t border-gray-200 p-1.5 dark:border-gray-700">
                  <p className="text-[10px] text-gray-400 dark:text-gray-500">
                    {absent
                      ? 'marked as subject-absent — click a face to pick a subject after all'
                      : 'click a face to set it as the subject — the other faces are then checked for duplication'}
                  </p>
                  {sample.faces.map(face => {
                    const isSubject = face.index === subject;
                    const levels = isSubject ? [] : levelsFor(face);
                    const twin = !isSubject && pick ? twinMarked(pick, face.index) : false;
                    const closest = !isSubject ? closestRefOf(face) : null;
                    const scoreBits = (['arcface', 'sface'] as const)
                      .map(model => {
                        const entry = face.scores?.[model]?.[source];
                        return entry?.mean != null ? `${model === 'arcface' ? 'ArcFace' : 'SFace'} ${entry.mean.toFixed(3)}` : null;
                      })
                      .filter(Boolean)
                      .join(' · ');
                    return (
                      <div
                        key={face.index}
                        className={classNames(
                          'flex items-center gap-2 rounded border p-1.5',
                          isSubject ? 'border-blue-400/60 bg-blue-500/5 dark:border-blue-500/40' : twin ? 'border-purple-400/50 dark:border-purple-500/40' : 'border-gray-200 dark:border-gray-700',
                        )}
                      >
                        <button
                          type="button"
                          onClick={() => onSavePick(sample, isSubject ? null : face.index)}
                          className={classNames(
                            'block shrink-0 rounded p-0 transition',
                            isSubject ? 'ring-2 ring-blue-500' : twin ? 'ring-2 ring-purple-500' : 'opacity-90 hover:opacity-100',
                          )}
                          title={`${isSubject ? 'subject face — click to un-pick' : 'click to make this the subject face'}\n${scoreBits}`}
                        >
                          <FaceCropThumb src={sample.path} bbox={face.bbox} size={64} />
                        </button>
                        <div className="min-w-0 flex-1 space-y-0.5">
                          <div className="flex items-center justify-between gap-2">
                            <span className={classNames('text-xs', isSubject ? 'font-semibold text-blue-600 dark:text-blue-400' : 'font-medium text-gray-700 dark:text-gray-200')}>
                              {isSubject ? 'Subject' : `Face ${face.index + 1}`}
                            </span>
                            {!isSubject && (
                              <button
                                type="button"
                                className={classNames(
                                  'rounded border px-1.5 py-0.5 text-[10px]',
                                  twin
                                    ? 'border-purple-400/70 bg-purple-500/10 text-purple-700 dark:text-purple-300'
                                    : 'border-gray-300 text-gray-500 hover:bg-gray-100 dark:border-gray-600 dark:text-gray-400 dark:hover:bg-gray-800',
                                )}
                                disabled={subject == null}
                                title={subject == null ? 'set the subject face first' : twin ? 'your own mark — the tool never asserts a twin. Click to remove' : 'mark this face as a duplicated subject yourself'}
                                onClick={() => onToggleTwin(sample, face.index)}
                              >
                                {twin ? '✓ your twin mark' : 'Mark as twin'}
                              </button>
                            )}
                          </div>
                          {isSubject ? (
                            <p className="text-[10px] text-gray-500 dark:text-gray-400">
                              the intended person — its scores represent the sample
                            </p>
                          ) : (
                            (['arcface', 'sface'] as const).map(model => {
                              const entry = levels.find(l => l.model === model);
                              const mean = face.scores?.[model]?.[source]?.mean ?? null;
                              return (
                                <p
                                  key={model}
                                  className={classNames(
                                    'truncate text-[11px] tabular-nums',
                                    entry?.level === 'likely_duplicate'
                                      ? 'text-red-600 dark:text-red-400'
                                      : entry?.level === 'resembles'
                                        ? 'text-amber-700 dark:text-amber-300'
                                        : 'text-gray-500 dark:text-gray-400',
                                  )}
                                  title={
                                    mean != null && entry
                                      ? `mean ${mean.toFixed(3)} vs the ${model === 'arcface' ? 'ArcFace' : 'SFace'} LOO band ${entry.band.toFixed(3)} (${source} refs) — per model, never averaged`
                                      : 'no score for this model/source'
                                  }
                                >
                                  {model === 'arcface' ? 'ArcFace' : 'SFace'}{' '}
                                  {mean != null ? mean.toFixed(3) : '–'} — {entry ? leakLevelLabel(entry.level) : 'below the band'}
                                </p>
                              );
                            })
                          )}
                        </div>
                        {closest && (
                          <div className="shrink-0 text-center" title={`the single nearest reference photo by cosine (${closest.sim.toFixed(3)}) — compare at full resolution`}>
                            <FaceCropThumb src={closest.image.path} bbox={closest.bbox} size={48} />
                            <div className="mt-0.5 text-[9px] text-gray-400 dark:text-gray-500">closest ref {closest.sim.toFixed(2)}</div>
                          </div>
                        )}
                      </div>
                    );
                  })}
                  <div className="flex flex-wrap items-center gap-2 pt-0.5">
                    <button
                      type="button"
                      className={classNames(controlClass, 'shrink-0 whitespace-nowrap')}
                      title="The crops above are for orientation only — judge possible duplicates from the full-resolution image"
                      onClick={() => onOpenSample(sample.path)}
                    >
                      Open full resolution
                    </button>
                    <button
                      type="button"
                      className={classNames(
                        'shrink-0 whitespace-nowrap rounded border px-2 py-1 text-sm',
                        absent
                          ? 'border-sky-400/70 bg-sky-500/10 text-sky-700 dark:text-sky-300'
                          : 'border-gray-300 text-gray-600 hover:bg-gray-100 dark:border-gray-600 dark:text-gray-300 dark:hover:bg-gray-800',
                      )}
                      title={
                        absent
                          ? 'Click to un-mark — the sample returns to needing a subject pick'
                          : 'Mark that none of these faces is your subject (e.g. a prompt without the trigger). The sample leaves the likeness summaries and joins the trigger-adherence check instead.'
                      }
                      onClick={() => onMarkAbsent(sample, !absent)}
                    >
                      {absent ? '✓ no one is the subject' : 'No one is the subject'}
                    </button>
                    <span className="text-[10px] text-gray-400 dark:text-gray-500">judge duplicates from the full image, not the crops</span>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// Samples you marked "the subject is not in this picture" (no-trigger /
// trigger-adherence prompts), plus the picker for marking more. Verdicts
// reuse the leakage anchoring per model per source: any face at or above
// the LOO band means the subject likely appeared anyway (leaked); in the
// resembles band is borderline; all scored faces below is clean.
function TriggerAdherence({
  samples,
  picks,
  results,
  source,
  onMarkAbsent,
  onOpenSample,
}: {
  samples: IdentitySampleResult[];
  picks: IdentitySelection['sample_picks'] | null | undefined;
  results: { refsets?: Partial<Record<IdentitySource | 'combined', { loo?: Record<string, number | null> } | null>> };
  source: IdentitySource | 'combined';
  onMarkAbsent: (sample: IdentitySampleResult, absent: boolean) => void;
  onOpenSample: (path: string) => void;
}) {
  const [filter, setFilter] = useState('');
  // raw statuses: absentPickFor wants evaluated/ambiguous (pre-resolution)
  const eligible = useMemo(
    () => samples.filter(s => s.faces.length > 0 && (s.status === 'evaluated' || s.status === 'ambiguous')),
    [samples],
  );
  const marked = useMemo(() => eligible.filter(s => absentPickFor(s, picks)), [eligible, picks]);
  const promptFilter = useMemo(() => {
    const ids = new Set<number>();
    for (const match of filter.matchAll(/\d+/g)) {
      const value = Number(match[0]);
      if (Number.isInteger(value)) ids.add(value);
    }
    return ids;
  }, [filter]);
  const candidates = useMemo(() => {
    if (promptFilter.size === 0) return [];
    return eligible
      .filter(s => s.prompt_idx != null && promptFilter.has(s.prompt_idx) && absentPickFor(s, picks) === null)
      .sort((a, b) => (b.step ?? 0) - (a.step ?? 0) || a.filename.localeCompare(b.filename))
      .slice(0, 40);
  }, [eligible, picks, promptFilter]);

  const loo = (model: string) => results?.refsets?.[source]?.loo?.[model] ?? null;
  const verdictChip = (sample: IdentitySampleResult, model: 'arcface' | 'sface') => {
    const verdict = absenceVerdict(sample, model, source, loo(model));
    if (verdict == null) return <span key={model} className="text-gray-400 dark:text-gray-500">{model === 'arcface' ? 'ArcFace' : 'SFace'} –</span>;
    const tone =
      verdict === 'leaked'
        ? 'text-red-600 dark:text-red-400'
        : verdict === 'borderline'
          ? 'text-amber-700 dark:text-amber-300'
          : 'text-emerald-700 dark:text-emerald-300';
    return (
      <span key={model} className={classNames('tabular-nums', tone)} title={`best face vs the ${model === 'arcface' ? 'ArcFace' : 'SFace'} LOO band ${loo(model)?.toFixed(3) ?? '?'} — leaked = a face reached the band despite no trigger`}>
        {model === 'arcface' ? 'ArcFace' : 'SFace'} {verdict}
      </span>
    );
  };
  const summaryFor = (model: 'arcface' | 'sface') => {
    const counts = { leaked: 0, borderline: 0, clean: 0, unscored: 0 };
    for (const sample of marked) {
      const verdict = absenceVerdict(sample, model, source, loo(model));
      if (verdict == null) counts.unscored += 1;
      else counts[verdict] += 1;
    }
    return counts;
  };

  const row = (sample: IdentitySampleResult, isMarked: boolean) => (
    <div key={sample.filename} className="flex items-center gap-2 rounded border border-gray-200 p-1.5 dark:border-gray-700">
      {sample.faces[0] && <FaceCropThumb src={sample.path} bbox={sample.faces[0].bbox} size={32} />}
      <button
        type="button"
        className="min-w-0 flex-1 truncate text-left text-xs text-gray-700 hover:underline dark:text-gray-200"
        title={`${sample.filename}\nstep ${sample.step?.toLocaleString() ?? '?'} · prompt ${sample.prompt_idx ?? '?'} · ${sample.faces.length} face(s) — click to open`}
        onClick={() => onOpenSample(sample.path)}
      >
        step {sample.step?.toLocaleString() ?? '?'} · prompt {sample.prompt_idx ?? '?'} · {sample.faces.length} face{sample.faces.length > 1 ? 's' : ''}
      </button>
      {isMarked && (
        <span className="flex shrink-0 items-center gap-2 text-[10px]">
          {verdictChip(sample, 'arcface')}
          {verdictChip(sample, 'sface')}
        </span>
      )}
      <button
        type="button"
        className={classNames(
          'shrink-0 whitespace-nowrap rounded border px-1.5 py-0.5 text-[10px]',
          isMarked
            ? 'border-sky-400/70 bg-sky-500/10 text-sky-700 dark:text-sky-300'
            : 'border-gray-300 text-gray-500 hover:bg-gray-100 dark:border-gray-600 dark:text-gray-400 dark:hover:bg-gray-800',
        )}
        title={isMarked ? 'Un-mark — the sample returns to the likeness summaries' : 'Mark: the subject is not in this picture'}
        onClick={() => onMarkAbsent(sample, !isMarked)}
      >
        {isMarked ? '✓ absent' : 'mark absent'}
      </button>
    </div>
  );

  return (
    <details className="space-y-2 rounded border border-gray-300 p-2 dark:border-gray-600">
      <summary className="cursor-pointer select-none text-xs font-semibold" title="Samples where the prompt does not ask for your subject (e.g. no-trigger controls) — a correct render contains no match. Marking one excludes it from likeness summaries and checks whether the subject appeared anyway.">
        Trigger adherence {marked.length > 0 ? `· ${marked.length} marked` : ''}
      </summary>
      <p className="text-[10px] text-gray-500 dark:text-gray-400">
        Mark samples where <span className="font-medium">no one in the picture is your subject</span> (a prompt without the trigger should render the
        base model&apos;s person, not yours). Marked samples leave the likeness summaries; their faces are then checked against the LOO band —
        <span className="text-red-600 dark:text-red-400"> leaked</span> = a face reached the band anyway,{' '}
        <span className="text-amber-700 dark:text-amber-300">borderline</span> = within 0.1 below it,{' '}
        <span className="text-emerald-700 dark:text-emerald-300">clean</span> = all faces clearly strangers. Per model, never averaged.
      </p>
      {marked.length > 0 ? (
        <div className="space-y-3">
          <div className="space-y-1">
            <div className="flex flex-wrap items-baseline justify-between gap-2 text-[10px] text-gray-500 dark:text-gray-400">
              <span className="font-medium text-gray-600 dark:text-gray-300">Marked samples</span>
              {(['arcface', 'sface'] as const).map(model => {
                const counts = summaryFor(model);
                const scored = counts.leaked + counts.borderline + counts.clean;
                return scored > 0 ? (
                  <span key={model}>
                    {model === 'arcface' ? 'ArcFace' : 'SFace'}: {counts.leaked} leaked · {counts.borderline} borderline · {counts.clean} clean
                    {counts.unscored > 0 ? ` · ${counts.unscored} unscored` : ''}
                  </span>
                ) : null;
              })}
            </div>
            {marked.map(sample => row(sample, true))}
          </div>
        </div>
      ) : (
        <p className="text-[10px] text-gray-400 dark:text-gray-500">nothing marked yet</p>
      )}
      <div className="space-y-1 border-t border-gray-200 pt-1.5 dark:border-gray-700">
        <div className="flex items-center gap-2">
          <input
            type="text"
            value={filter}
            onChange={e => setFilter(e.target.value)}
            placeholder="prompt #s to mark, e.g. 9 10 11"
            className={classNames(controlClass, 'min-w-0 flex-1 text-xs')}
            title="List the prompt indices whose samples are no-trigger controls; every face-bearing sample of those prompts is listed for marking"
          />
        </div>
        {promptFilter.size > 0 && (
          <div className="space-y-1">
            <p className="text-[10px] text-gray-400 dark:text-gray-500">
              {candidates.length > 0
                ? `unmarked samples of prompt ${[...promptFilter].sort((a, b) => a - b).join(', ')} (latest first, up to 40)`
                : `no unmarked face-bearing samples for prompt ${[...promptFilter].sort((a, b) => a - b).join(', ')}`}
            </p>
            {candidates.map(sample => row(sample, false))}
          </div>
        )}
      </div>
    </details>
  );
}

interface IdentityEvalPanelProps {
  job: Job;
  identity: IdentityEvalApi;
  samplePaths: string[];
  onClose: () => void;
  onOpenSample: (path: string) => void;
  onFilterStep: (step: number) => void;
}

export default function IdentityEvalPanel({ job, identity, samplePaths, onClose, onOpenSample, onFilterStep }: IdentityEvalPanelProps) {
  const { refs, selection, results, resolvedSamples, progress, running, loading, error, source, report, reportLoading } = identity;
  const [deviceMode, setDeviceMode] = useState<'auto' | 'gpu' | 'cpu'>('auto');
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    identity.loadRefs(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job.id]);

  const trainingImages = refs?.training?.images ?? [];
  const externalImages = refs?.external?.images ?? [];
  const models = (refs?.models ?? results?.models ?? {}) as Record<string, { active?: boolean; available?: boolean; reason?: string | null }>;
  const arcfaceOn = Boolean(models.arcface?.active);
  const sfaceOn = Boolean(models.sface?.active);
  const activeSource: IdentitySource | 'combined' = source === 'both' ? 'combined' : source;

  // Selection keys for training refs use the shared canonical form
  // (normalizeRefPath) -- exactly what selectionForImage reads and the
  // worker looks up, so exclusions/face picks survive the round-trip.
  const refKey = (image: IdentityRefImage, src: 'training' | 'external') =>
    src === 'external' ? image.name : normalizeRefPath(image.path);

  const handleToggleExclude = (src: 'training' | 'external') => (image: IdentityRefImage, excluded: boolean) => {
    const key = refKey(image, src);
    const update = { [key]: { excluded, face_index: selectionForImage(selection, image, src).face_index ?? null } };
    identity.saveSelection(src === 'external' ? { externalRefs: update } : { trainingRefs: update });
  };

  const handlePickFace = (src: 'training' | 'external') => (image: IdentityRefImage, faceIndex: number) => {
    const key = refKey(image, src);
    const update = { [key]: { excluded: false, face_index: faceIndex } };
    identity.saveSelection(src === 'external' ? { externalRefs: update } : { trainingRefs: update });
  };

  // Phase 2e: subject-face picks + twin marks per generated sample
  const samplePicks = selection?.sample_picks;
  const handleSaveSamplePick = (sample: IdentitySampleResult, faceIndex: number | null) => {
    const existing = samplePicks?.[sample.filename];
    const marks = faceIndex == null ? [] : (existing?.twin_confirmed ?? []).filter(i => i !== faceIndex);
    identity.saveSelection({
      samplePicks: { [sample.filename]: { face_index: faceIndex, ...(marks.length > 0 ? { twin_confirmed: marks } : {}) } },
    });
  };
  const handleToggleTwin = (sample: IdentitySampleResult, faceIndex: number) => {
    const existing = samplePicks?.[sample.filename];
    if (existing?.face_index == null) return; // a twin mark needs a subject first
    const marks = new Set(existing.twin_confirmed ?? []);
    if (marks.has(faceIndex)) marks.delete(faceIndex);
    else marks.add(faceIndex);
    const sorted = [...marks].sort((a, b) => a - b);
    identity.saveSelection({
      samplePicks: { [sample.filename]: { face_index: existing.face_index, ...(sorted.length > 0 ? { twin_confirmed: sorted } : {}) } },
    });
  };
  // "the subject is not in this picture" — replaces any face pick; un-marking
  // returns the sample to pending (face_index null), never to a stale pick
  const handleMarkAbsent = (sample: IdentitySampleResult, absent: boolean) => {
    identity.saveSelection({
      samplePicks: { [sample.filename]: absent ? { subject_absent: true } : { face_index: null } },
    });
  };

  const handleRun = () => {
    const doStart = () => identity.startRun({ samples: samplePaths, deviceMode });
    if (deviceMode === 'gpu' && job.status === 'running') {
      openConfirm({
        title: 'Force GPU while training?',
        message:
          'A training job is running on this GPU. Running the evaluation on it as well can starve training of VRAM ' +
          '(on Windows it may silently spill into system RAM). Continue anyway?',
        type: 'warning',
        confirmText: 'Run on GPU',
        onConfirm: doStart,
      });
      return;
    }
    doStart();
  };

  const progressTotal = progress?.total ?? 0;
  const progressDone = progress?.done ?? 0;
  // summaries see pick-resolved samples: a picked multi-face sample counts
  // as evaluated with its picked face's scores
  const summaries = groupSummariesByStep(resolvedSamples, activeSource);
  const looArcface = results?.refsets?.[activeSource]?.loo?.arcface ?? null;
  const looSface = results?.refsets?.[activeSource]?.loo?.sface ?? null;
  const latestSummary = summaries.length > 0 ? summaries[summaries.length - 1] : null;
  const maxMedianArcface = summaries.reduce((acc, row) => Math.max(acc, row.medianArcface ?? -Infinity), -Infinity);
  const maxMedianSface = summaries.reduce((acc, row) => Math.max(acc, row.medianSface ?? -Infinity), -Infinity);

  return (
    <div className="absolute inset-y-0 right-0 z-[15] flex w-[420px] max-w-[92vw] flex-col border-l border-gray-300 bg-gray-50 text-sm text-gray-900 dark:border-gray-700 dark:bg-gray-900 dark:text-gray-100">
      <div className="flex items-center justify-between border-b border-gray-300 px-3 py-2 dark:border-gray-700">
        <h3 className="text-sm font-semibold">Identity Evaluation</h3>
        <div className="flex items-center gap-2">
          <button type="button" className={controlClass} title="Re-discover references" onClick={() => identity.loadRefs(true)}>
            <LuRefreshCw className={classNames('inline', loading && 'animate-spin')} />
          </button>
          <button type="button" className={controlClass} onClick={onClose} aria-label="Close identity panel">
            <LuX className="inline" />
          </button>
        </div>
      </div>

      <div className="flex-1 space-y-4 overflow-y-auto p-3">
        <p className="text-xs text-gray-500 dark:text-gray-400">
          Scores every sample against your reference photos with two face-recognition models — higher means closer to
          the reference face (1.0 = identical). Use it to see where likeness peaks and to compare checkpoints.
        </p>
        <details className="text-xs text-gray-500 dark:text-gray-400">
          <summary className="cursor-pointer select-none font-medium text-gray-600 dark:text-gray-300">
            How to read these numbers
          </summary>
          <ul className="mt-1 list-disc space-y-0.5 pl-4">
            <li>
              <span className="font-medium">ArcFace / SFace</span> are two independent face-recognition models. Their
              scales differ — never compare an ArcFace number to an SFace number.
            </li>
            <li>
              The <span className="font-medium">leave-one-out (LOO) band</span> shown in parentheses is how similar your
              reference photos are to <em>each other</em> — a same-person baseline. Strong generated faces often land
              near or slightly above it; well below it means weak likeness.
            </li>
            <li>
              Sample counts: <span className="font-medium">ok</span> = one face, scored ·{' '}
              <span className="font-medium">amb</span> = ambiguous, multiple faces found ·{' '}
              <span className="font-medium">abs</span> = marked subject-absent (no-trigger prompt) ·{' '}
              <span className="font-medium">fail</span> = no face or error.
            </li>
            <li>
              <span className="font-medium">▲</span> marks the highest median — where similarity peaks. When checkpoints
              are close, trust the “Checkpoint comparison” paired verdicts (a difference is only real when its 95%
              confidence interval excludes 0), then judge the samples by eye.
            </li>
          </ul>
        </details>
        {error && (
          <div className="rounded border border-red-400/60 bg-red-500/10 p-2 text-xs text-red-700 dark:text-red-300">{error}</div>
        )}
        {progress?.error && (
          <div className="rounded border border-red-400/60 bg-red-500/10 p-2 text-xs text-red-700 dark:text-red-300">
            Evaluation worker failed: {progress.error}{' '}
            <span className="text-gray-500 dark:text-gray-400">(details in output/{job.name}/identity_eval/run.log)</span>
          </div>
        )}

        {/* models */}
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <span
            className={classNames(
              'rounded-full border px-2 py-0.5',
              arcfaceOn ? STATUS_BADGE.ok : STATUS_BADGE.error,
            )}
            title={models.arcface?.reason ?? '~/.insightface/models/buffalo_l'}
          >
            ArcFace {arcfaceOn ? 'active' : 'missing'}
          </span>
          <span
            className={classNames('rounded-full border px-2 py-0.5', sfaceOn ? STATUS_BADGE.ok : STATUS_BADGE.no_face)}
            title={models.sface?.reason ?? 'models/face/face_recognition_sface_2021dec.onnx (Apache-2.0, user-supplied)'}
          >
            SFace {sfaceOn ? 'active' : 'not installed'}
          </span>
          {!sfaceOn && (
            <span className="text-gray-500 dark:text-gray-400">
              permissive SFace-only mode available once installed (<code>docs/identity_eval.md</code>)
            </span>
          )}
        </div>

        {/* source + device + run */}
        <div className="space-y-2 rounded border border-gray-300 p-2 dark:border-gray-600">
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500 dark:text-gray-400">References</span>
            <div className="flex overflow-hidden rounded border border-gray-300 text-xs dark:border-gray-600">
              {(['training', 'external', 'both'] as const).map(mode => (
                <button
                  key={mode}
                  type="button"
                  onClick={() => identity.setSource(mode)}
                  className={classNames(
                    'px-2 py-1 capitalize',
                    source === mode ? 'bg-blue-500 text-white' : 'bg-white text-gray-700 hover:bg-gray-100 dark:bg-gray-900 dark:text-gray-300 dark:hover:bg-gray-800',
                  )}
                >
                  {mode}
                </button>
              ))}
            </div>
            <label className="ml-auto flex items-center gap-1 text-xs">
              Device
              <select value={deviceMode} onChange={e => setDeviceMode(e.target.value as 'auto' | 'gpu' | 'cpu')} className={controlClass}>
                <option value="auto">Auto</option>
                <option value="gpu">GPU</option>
                <option value="cpu">CPU</option>
              </select>
            </label>
          </div>
          <div className="flex items-center gap-2">
            {running ? (
              <button type="button" className={classNames(controlClass, 'flex items-center gap-1')} onClick={() => identity.cancel()}>
                <FaStop className="inline" /> Cancel
              </button>
            ) : (
              <button
                type="button"
                className={classNames(controlClass, 'flex items-center gap-1')}
                onClick={handleRun}
                disabled={samplePaths.length === 0}
                title="Score all current samples against the selected reference sources"
              >
                <FaPlay className="inline" /> Evaluate {samplePaths.length} samples
              </button>
            )}
            {results && !running && (
              <span className={classNames('text-xs', results.completed ? 'text-emerald-600 dark:text-emerald-400' : 'text-amber-600 dark:text-amber-400')}>
                {results.completed ? 'complete' : 'partial'} · {results.samples.length} scored
              </span>
            )}
          </div>
          {(running || progress) && (
            <div>
              <div className="mb-1 flex justify-between text-xs text-gray-500 dark:text-gray-400">
                <span>{progress?.phase === 'refs' ? 'detecting reference faces' : 'scoring samples'}</span>
                <span>
                  {progressDone}/{progressTotal || '…'}
                </span>
              </div>
              <div className="h-2 w-full overflow-hidden rounded bg-gray-300 dark:bg-gray-700">
                <div
                  className={classNames('h-full transition-all duration-200', running ? 'bg-blue-500' : 'bg-emerald-500')}
                  style={{ width: `${progressTotal > 0 ? Math.min(100, (progressDone / progressTotal) * 100) : 0}%` }}
                />
              </div>
            </div>
          )}
        </div>

        {/* references */}
        <div className="space-y-3">
          <RefSection
            title="Training references"
            source="training"
            images={trainingImages}
            selection={selection}
            onToggleExclude={handleToggleExclude('training')}
            onPickFace={handlePickFace('training')}
          />
          <div className="space-y-2">
            <RefSection
              title="External references"
              source="external"
              images={externalImages}
              selection={selection}
              onToggleExclude={handleToggleExclude('external')}
              onPickFace={handlePickFace('external')}
              onRemove={image => identity.removeExternal([image.name])}
            />
            <div>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/jpeg,image/png,image/webp"
                multiple
                className="hidden"
                onChange={e => {
                  const files = Array.from(e.target.files ?? []);
                  if (files.length > 0) identity.addExternalFiles(files);
                  e.target.value = '';
                }}
              />
              <button type="button" className={controlClass} onClick={() => fileInputRef.current?.click()}>
                Add images…
              </button>
              <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                External refs test generalization on unseen photos; training-ref similarity partly measures memorization.
                Stored content-addressed under <code>output/{job.name}/identity_eval/external_refs</code>.
              </p>
            </div>
          </div>
        </div>

        {/* multi-person samples: subject picker + leakage inspection (Phase 2e) */}
        {results && results.samples.length > 0 && (
          <MultiPersonSamples
            samples={results.samples}
            picks={samplePicks}
            trainingRefs={trainingImages}
            externalRefs={externalImages}
            results={results}
            source={activeSource}
            onSavePick={handleSaveSamplePick}
            onMarkAbsent={handleMarkAbsent}
            onToggleTwin={handleToggleTwin}
            onOpenSample={onOpenSample}
          />
        )}

        {/* trigger adherence: samples where the subject is deliberately absent */}
        {results && results.samples.length > 0 && (
          <TriggerAdherence
            samples={results.samples}
            picks={samplePicks}
            results={results}
            source={activeSource}
            onMarkAbsent={handleMarkAbsent}
            onOpenSample={onOpenSample}
          />
        )}

        {/* summary */}
        {results && results.samples.length > 0 && (
          <div className="space-y-2 rounded border border-gray-300 p-2 dark:border-gray-600">
            <div className="flex items-baseline justify-between">
              <span className="text-xs font-semibold">Summary by step</span>
              <span className="text-xs text-gray-500 dark:text-gray-400">
                {activeSource === 'combined' ? 'training + external' : activeSource} refs
              </span>
            </div>
            <table className="w-full table-fixed text-xs tabular-nums">
              <thead>
                <tr className="text-gray-500 dark:text-gray-400">
                  <th className="text-left font-medium">Step</th>
                  <th
                    className="font-medium"
                    title="Median ArcFace similarity to the reference faces — higher = closer. ArcFace and SFace use different scales; never compare across models."
                  >
                    ArcFace med
                  </th>
                  <th className="font-medium" title="Median SFace similarity to the reference faces — higher = closer">
                    SFace med
                  </th>
                  <th
                    className="font-medium"
                    title="Sample counts: ok = one face, scored · amb = ambiguous, multiple faces found · abs = you marked the subject absent (no-trigger prompt) · fail = no face or error"
                  >
                    ok/amb/abs/fail
                  </th>
                </tr>
              </thead>
              <tbody>
                {summaries.map(row => {
                  const bestA = row.medianArcface != null && row.medianArcface === maxMedianArcface;
                  const bestS = row.medianSface != null && row.medianSface === maxMedianSface;
                  return (
                    <tr key={row.step} className="border-t border-gray-200 dark:border-gray-700">
                      <td className="py-0.5 text-left">{row.step.toLocaleString()}</td>
                      <td
                        className={classNames(
                          'py-0.5 text-center',
                          bestA && 'bg-blue-500/10 font-semibold text-gray-900 dark:text-gray-50',
                        )}
                      >
                        {bestA && (
                          <span title="highest median across steps" className="text-blue-500">
                            ▲
                          </span>
                        )}{' '}
                        {row.medianArcface?.toFixed(3) ?? '–'}
                        {looArcface !== null && <span className="text-gray-400"> ({looArcface.toFixed(3)})</span>}
                      </td>
                      <td
                        className={classNames(
                          'py-0.5 text-center',
                          bestS && 'bg-blue-500/10 font-semibold text-gray-900 dark:text-gray-50',
                        )}
                      >
                        {bestS && (
                          <span title="highest median across steps" className="text-blue-500">
                            ▲
                          </span>
                        )}{' '}
                        {row.medianSface?.toFixed(3) ?? '–'}
                        {looSface !== null && <span className="text-gray-400"> ({looSface.toFixed(3)})</span>}
                      </td>
                      <td className="py-0.5 text-center text-gray-500 dark:text-gray-400">
                        {row.evaluated}/{row.ambiguous}
                        <span className={row.absent > 0 ? 'text-sky-600 dark:text-sky-400' : ''}>/{row.absent}</span>/{row.failed}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Numbers in parentheses are the leave-one-out (LOO) reference band: each reference photo scored against
              the others — how similar real photos of this person are to each other. It is a same-person baseline, not
              an upper bound: a generated face that captures the shared identity can score above it. ▲ marks the
              highest median; medians exclude ambiguous, failed and subject-absent samples.
            </p>
            {latestSummary && latestSummary.highest.length > 0 && (
              <div className="flex flex-wrap items-center gap-1 text-xs">
                <span className="text-gray-500 dark:text-gray-400">
                  highest at step {latestSummary.step.toLocaleString()}:
                </span>
                {latestSummary.highest.map(sample => (
                  <SampleScoreChip key={sample.filename} sample={sample} source={activeSource} tone="high" onOpen={onOpenSample} />
                ))}
              </div>
            )}
            {latestSummary && latestSummary.lowest.length > 0 && (
              <div className="flex flex-wrap items-center gap-1 text-xs">
                <span className="text-gray-500 dark:text-gray-400">
                  lowest at step {latestSummary.step.toLocaleString()}:
                </span>
                {latestSummary.lowest.map(sample => (
                  <SampleScoreChip key={sample.filename} sample={sample} source={activeSource} tone="low" onOpen={onOpenSample} />
                ))}
              </div>
            )}
          </div>
        )}

        {/* checkpoint comparison (Phase 2b) */}
        {results && results.samples.length > 0 && (
          <IdentityCheckpointSection
            report={report}
            loading={reportLoading}
            onRefresh={identity.loadReport}
            onSaveCategories={identity.saveCategories}
            onOpenSample={filename => {
              const sample = identity.index.get(filename);
              if (sample) onOpenSample(sample.path);
            }}
            onFilterStep={onFilterStep}
          />
        )}

        {/* footer actions */}
        <div className="flex items-center gap-2 border-t border-gray-300 pt-2 text-xs dark:border-gray-700">
          <button
            type="button"
            className={controlClass}
            disabled={running}
            onClick={() =>
              openConfirm({
                title: 'Clear results',
                message: 'Delete the evaluation results for this job? References and cache are kept.',
                type: 'warning',
                confirmText: 'Clear',
                onConfirm: () => identity.clear(false),
              })
            }
          >
            Clear results
          </button>
          <button
            type="button"
            className={controlClass}
            disabled={running}
            onClick={() =>
              openConfirm({
                title: 'Clear results and cache',
                message: 'Delete results and the embedding cache? References re-detect on the next run.',
                type: 'warning',
                confirmText: 'Clear all',
                onConfirm: () => identity.clear(true),
              })
            }
          >
            Clear + cache
          </button>
          {running && <LuLoader className="inline animate-spin text-blue-500" />}
        </div>
      </div>
    </div>
  );
}
