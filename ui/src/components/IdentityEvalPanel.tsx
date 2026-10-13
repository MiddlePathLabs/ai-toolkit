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
  groupSummariesByStep,
  normalizeRefPath,
  selectionForImage,
  type IdentityRefImage,
  type IdentitySource,
} from '@/utils/identityEval';
import { encodeFilePathForUrl } from '@/utils/basic';
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

// CSS face crop: the image is scaled and positioned so the bbox fills the
// box -- a real crop view without server-side thumbnails.
function FaceCropThumb({ src, bbox, size = 72 }: { src: string; bbox: number[]; size?: number }) {
  const [natural, setNatural] = useState<{ w: number; h: number } | null>(null);
  const [x1, y1, x2, y2] = bbox;
  const scale = natural ? Math.min(size / Math.max(x2 - x1, 1), size / Math.max(y2 - y1, 1), 2.5) : 1;
  return (
    <div
      className="relative overflow-hidden rounded border border-gray-300 bg-gray-900 dark:border-gray-600"
      style={{ width: size, height: size }}
    >
      {natural && (
        <img
          src={imageUrl(src)}
          alt="reference face"
          className="absolute origin-top-left"
          style={{
            width: natural.w * scale,
            height: natural.h * scale,
            left: -x1 * scale,
            top: -y1 * scale,
          }}
        />
      )}
      <img
        src={imageUrl(src)}
        alt=""
        className="hidden"
        onLoad={e => {
          const img = e.target as HTMLImageElement;
          setNatural({ w: img.naturalWidth, h: img.naturalHeight });
        }}
      />
    </div>
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

interface IdentityEvalPanelProps {
  job: Job;
  identity: IdentityEvalApi;
  samplePaths: string[];
  onClose: () => void;
  onOpenSample: (path: string) => void;
}

export default function IdentityEvalPanel({ job, identity, samplePaths, onClose, onOpenSample }: IdentityEvalPanelProps) {
  const { refs, selection, results, progress, running, loading, error, source } = identity;
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
  const summaries = groupSummariesByStep(results?.samples ?? [], activeSource);
  const looArcface = results?.refsets?.[activeSource]?.loo?.arcface ?? null;
  const looSface = results?.refsets?.[activeSource]?.loo?.sface ?? null;

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
        {error && (
          <div className="rounded border border-red-400/60 bg-red-500/10 p-2 text-xs text-red-700 dark:text-red-300">{error}</div>
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
                  <th className="font-medium">A med</th>
                  <th className="font-medium">S med</th>
                  <th className="font-medium">ok/amb/fail</th>
                </tr>
              </thead>
              <tbody>
                {summaries.map(row => (
                  <tr key={row.step} className="border-t border-gray-200 dark:border-gray-700">
                    <td className="py-0.5 text-left">{row.step.toLocaleString()}</td>
                    <td className="py-0.5 text-center">
                      {row.medianArcface?.toFixed(3) ?? '–'}
                      {looArcface !== null && <span className="text-gray-400"> ({looArcface.toFixed(3)})</span>}
                    </td>
                    <td className="py-0.5 text-center">
                      {row.medianSface?.toFixed(3) ?? '–'}
                      {looSface !== null && <span className="text-gray-400"> ({looSface.toFixed(3)})</span>}
                    </td>
                    <td className="py-0.5 text-center text-gray-500 dark:text-gray-400">
                      {row.evaluated}/{row.ambiguous}/{row.failed}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Values in parentheses are the reference LOO self-similarity band (the natural ceiling). Medians exclude
              ambiguous and failed samples.
            </p>
            {summaries.length > 0 && summaries[summaries.length - 1].lowest.length > 0 && (
              <div className="flex flex-wrap items-center gap-1 text-xs">
                <span className="text-gray-500 dark:text-gray-400">lowest:</span>
                {summaries[summaries.length - 1].lowest.map(sample => (
                  <button
                    key={sample.filename}
                    type="button"
                    className="rounded border border-gray-300 px-1.5 py-0.5 hover:bg-gray-100 dark:border-gray-600 dark:hover:bg-gray-800"
                    onClick={() => onOpenSample(sample.path)}
                    title={`Step ${sample.step?.toLocaleString()} · prompt ${sample.prompt_idx}`}
                  >
                    {(sample.primary?.arcface?.[activeSource]?.mean ?? 0).toFixed(3)}
                  </button>
                ))}
              </div>
            )}
          </div>
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
