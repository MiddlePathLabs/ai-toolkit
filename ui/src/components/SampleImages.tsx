import { useMemo, useState, useRef, useCallback, useEffect } from 'react';
import { Virtuoso, VirtuosoHandle } from 'react-virtuoso';
import useSampleImages from '@/hooks/useSampleImages';
import SampleImageCard from './SampleImageCard';
import { Job } from '@prisma/client';
import { JobConfig } from '@/types';
import { LuImageOff, LuLoader, LuBan } from 'react-icons/lu';
import { Button, Dialog, DialogBackdrop, DialogPanel, DialogTitle } from '@headlessui/react';
import { FaDownload } from 'react-icons/fa';
import { apiClient } from '@/utils/api';
import { encodeFilePathForUrl } from '@/utils/basic';
import classNames from 'classnames';
import { FaCaretDown, FaCaretUp } from 'react-icons/fa';
import SampleImageViewer from './SampleImageViewer';
import { openConfirm } from './ConfirmModal';
import { buildSampleMatrix, getSampleItems, getSampleMetadata, parseSampleFilename } from '@/utils/sampleImages';
import SampleComparisonViewer from './SampleComparisonViewer';

interface SampleImagesMenuProps {
  job?: Job | null;
}

export const SampleImagesMenu = ({ job }: SampleImagesMenuProps) => {
  const [isZipping, setIsZipping] = useState(false);

  const downloadZip = async () => {
    if (isZipping) return;
    setIsZipping(true);

    try {
      const res = await apiClient.post('/api/zip', {
        zipTarget: 'samples',
        jobName: job?.name,
      });

      const zipPath = res.data.zipPath; // e.g. /mnt/Train2/out/ui/.../samples.zip
      if (!zipPath) throw new Error('No zipPath in response');

      const downloadPath = `/api/files/${encodeFilePathForUrl(zipPath)}`;
      const a = document.createElement('a');
      a.href = downloadPath;
      // optional: suggest filename (browser may ignore if server sets Content-Disposition)
      a.download = 'samples.zip';
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch (err) {
      console.error('Error downloading zip:', err);
    } finally {
      setIsZipping(false);
    }
  };
  return (
    <Button
      onClick={downloadZip}
      className={classNames(
        `flex-1 sm:flex-initial justify-center px-2 sm:px-4 py-1 h-8 hover:bg-gray-200 dark:hover:bg-gray-700 flex items-center`,
        {
          'opacity-50 cursor-not-allowed': isZipping,
        },
      )}
    >
      {isZipping ? (
        <LuLoader className="animate-spin inline-block sm:mr-2" />
      ) : (
        <FaDownload className="inline-block sm:mr-2" />
      )}
      <span className="hidden sm:inline">{isZipping ? 'Preparing' : 'Download'}</span>
    </Button>
  );
};

interface SampleImagesProps {
  job: Job;
}

export default function SampleImages({ job }: SampleImagesProps) {
  const { sampleImages, plannedSamples, deletedSamples, status, refreshSampleImages } = useSampleImages(job.id, 5000);
  const [selectedSamplePath, setSelectedSamplePath] = useState<string | null>(null);
  // multi-select for bulk delete: shift-click ranges from the anchor, ctrl/cmd-click toggles
  const [selectedSet, setSelectedSet] = useState<Set<string>>(() => new Set());
  const anchorIdxRef = useRef<number | null>(null);
  // selection as it was when the anchor was set; shift-click ranges are rebuilt on top of it, not accumulated
  const baseSetRef = useRef<Set<string>>(new Set());
  const [deleteProgress, setDeleteProgress] = useState<{ done: number; total: number } | null>(null);
  const [scrollParent, setScrollParent] = useState<HTMLDivElement | null>(null);
  const scrollParentCallback = useCallback((el: HTMLDivElement | null) => setScrollParent(el), []);
  const virtuosoRef = useRef<VirtuosoHandle>(null);
  const headerRef = useRef<HTMLDivElement>(null);
  const [thumbnailSize, setThumbnailSize] = useState('fit');
  const [showMetadata, setShowMetadata] = useState(true);
  const [stepFilter, setStepFilter] = useState('all');
  const [jumpTarget, setJumpTarget] = useState('');
  const [pendingJump, setPendingJump] = useState<string | null>(null);
  const [comparisonOpen, setComparisonOpen] = useState(false);
  const hasEma = useMemo(() => {
    if (!job.job_config) return false;
    const config = JSON.parse(job.job_config) as JobConfig;
    return Boolean(config.config.process[0].train.ema_config?.use_ema);
  }, [job.job_config]);
  const sampleConfig = useMemo(() => {
    if (job?.job_config) {
      const jobConfig = JSON.parse(job.job_config) as JobConfig;
      return jobConfig.config.process[0].sample;
    }
    return null;
  }, [job]);
  const sampleItems = useMemo(() => getSampleItems(sampleConfig), [sampleConfig]);
  const numSamples = Math.max(sampleItems.length, 1);
  const rows = useMemo(
    () => buildSampleMatrix(sampleImages, sampleConfig, numSamples, { plannedSamples, deletedSamples }),
    [sampleImages, sampleConfig, numSamples, plannedSamples, deletedSamples],
  );
  const rowSteps = useMemo(
    () =>
      new Map(
        rows.map(row => {
          const path = row.paths.find(path => path !== null);
          const step = path ? (parseSampleFilename(path)?.trainingStep ?? null) : Number(row.key.split(':')[0]);
          return [row.key, step !== null && Number.isSafeInteger(step) ? step : null];
        }),
      ),
    [rows],
  );
  const steps = useMemo(
    () => [...new Set([...rowSteps.values()].filter((step): step is number => step !== null && Number.isFinite(step)))],
    [rowSteps],
  );
  const latestStep = steps[steps.length - 1];
  const visibleRows = useMemo(
    () =>
      rows.filter(
        row =>
          stepFilter === 'all' || rowSteps.get(row.key) === (stepFilter === 'latest' ? latestStep : Number(stepFilter)),
      ),
    [rows, rowSteps, stepFilter, latestStep],
  );
  const sortedSampleImages = useMemo(
    () => visibleRows.flatMap(row => row.paths.filter((path): path is string => path !== null)),
    [visibleRows],
  );
  useEffect(() => {
    anchorIdxRef.current = null;
    baseSetRef.current = new Set();
    setSelectedSet(new Set());
  }, [stepFilter]);
  useEffect(() => {
    if (!pendingJump || !scrollParent) return;
    const index = visibleRows.findIndex(row => row.key === pendingJump);
    if (index < 0) return;
    const frame = requestAnimationFrame(() => {
      virtuosoRef.current?.scrollToIndex({ index, align: 'start', offset: -(headerRef.current?.clientHeight ?? 0) });
      setPendingJump(null);
    });
    return () => cancelAnimationFrame(frame);
  }, [pendingJump, visibleRows, scrollParent]);

  const handleCardClick = useCallback(
    (sample: string, e: React.MouseEvent) => {
      const isRange = e.shiftKey;
      const isToggle = e.ctrlKey || e.metaKey;
      if (!isRange && !isToggle) {
        setSelectedSet(new Set());
        anchorIdxRef.current = null;
        baseSetRef.current = new Set();
        setSelectedSamplePath(sample);
        return;
      }
      e.preventDefault();
      const idx = sortedSampleImages.indexOf(sample);
      if (idx === -1) return;
      setSelectedSet(prev => {
        const anchor = anchorIdxRef.current;
        if (isRange && anchor !== null && anchor < sortedSampleImages.length) {
          const next = new Set(baseSetRef.current);
          const [lo, hi] = anchor < idx ? [anchor, idx] : [idx, anchor];
          for (let i = lo; i <= hi; i++) next.add(sortedSampleImages[i]);
          return next;
        }
        const next = new Set(prev);
        if (isToggle && next.has(sample)) {
          next.delete(sample);
        } else {
          next.add(sample);
        }
        anchorIdxRef.current = idx;
        baseSetRef.current = new Set(next);
        return next;
      });
    },
    [sortedSampleImages],
  );

  const deleteSelected = useCallback(() => {
    const paths = Array.from(selectedSet);
    if (paths.length === 0) return;
    openConfirm({
      title: 'Delete Samples',
      message: `Are you sure you want to delete ${paths.length} sample${paths.length === 1 ? '' : 's'}? This action cannot be undone.`,
      type: 'warning',
      confirmText: 'Delete',
      onConfirm: async () => {
        setDeleteProgress({ done: 0, total: paths.length });
        // bounded concurrency so the progress counter advances and the server isn't flooded
        const CONCURRENCY = 4;
        let cursor = 0;
        let done = 0;
        const worker = async () => {
          while (cursor < paths.length) {
            const imgPath = paths[cursor++];
            try {
              await apiClient.post('/api/img/delete', { imgPath });
            } catch (error) {
              console.error('Error deleting sample:', imgPath, error);
            }
            done++;
            setDeleteProgress({ done, total: paths.length });
          }
        };
        await Promise.all(Array.from({ length: Math.min(CONCURRENCY, paths.length) }, worker));
        setDeleteProgress(null);
        setSelectedSet(new Set());
        anchorIdxRef.current = null;
        baseSetRef.current = new Set();
        refreshSampleImages();
      },
    });
  }, [selectedSet, refreshSampleImages]);

  // Delete/Backspace deletes the selection, Escape clears it; ignored while the viewer is open or typing in a field
  useEffect(() => {
    if (selectedSet.size === 0) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (selectedSamplePath || comparisonOpen) return;
      const target = e.target as HTMLElement | null;
      if (target?.closest('input, textarea, select, button, [contenteditable="true"]')) return;
      if (e.key === 'Delete' || e.key === 'Backspace') {
        e.preventDefault();
        deleteSelected();
      } else if (e.key === 'Escape') {
        setSelectedSet(new Set());
        anchorIdxRef.current = null;
        baseSetRef.current = new Set();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [selectedSet.size, selectedSamplePath, comparisonOpen, deleteSelected]);

  const scrollToBottom = () => {
    virtuosoRef.current?.scrollToIndex({ index: 'LAST', align: 'end' });
  };

  const scrollToTop = () => {
    virtuosoRef.current?.scrollToIndex({ index: 0, align: 'start', offset: -(headerRef.current?.clientHeight ?? 0) });
  };

  const PageInfoContent = useMemo(() => {
    let icon = null;
    let text = '';
    let subtitle = '';
    let showIt = false;
    let bgColor = '';
    let textColor = '';
    let iconColor = '';

    if (rows.length > 0) return null;

    if (status == 'loading') {
      icon = <LuLoader className="animate-spin w-8 h-8" />;
      text = 'Loading Samples';
      subtitle = 'Please wait while we fetch your samples...';
      showIt = true;
      bgColor = 'bg-gray-50 dark:bg-gray-800/50';
      textColor = 'text-gray-900 dark:text-gray-100';
      iconColor = 'text-gray-500 dark:text-gray-400';
    }
    if (status == 'error') {
      icon = <LuBan className="w-8 h-8" />;
      text = 'Error Loading Samples';
      subtitle = 'There was a problem fetching the samples.';
      showIt = true;
      bgColor = 'bg-red-50 dark:bg-red-950/20';
      textColor = 'text-red-900 dark:text-red-100';
      iconColor = 'text-red-600 dark:text-red-400';
    }
    if (status == 'success' && sampleImages.length === 0) {
      icon = <LuImageOff className="w-8 h-8" />;
      text = 'No Samples Found';
      subtitle = 'No samples have been generated yet';
      showIt = true;
      bgColor = 'bg-gray-50 dark:bg-gray-800/50';
      textColor = 'text-gray-900 dark:text-gray-100';
      iconColor = 'text-gray-500 dark:text-gray-400';
    }

    if (!showIt) return null;

    return (
      <div
        className={`mt-10 flex flex-col items-center justify-center py-16 px-8 rounded-xl border-2 border-gray-700 border-dashed ${bgColor} ${textColor} mx-auto max-w-md text-center`}
      >
        <div className={`${iconColor} mb-4`}>{icon}</div>
        <h3 className="text-lg font-semibold mb-2">{text}</h3>
        <p className="text-sm opacity-75 leading-relaxed">{subtitle}</p>
      </div>
    );
  }, [status, sampleImages.length, rows.length]);

  const gridStyle = { gridTemplateColumns: `repeat(${numSamples}, minmax(0, 1fr))` };
  const matrixStyle = {
    width: thumbnailSize === 'fit' ? '100%' : `${numSamples * Number(thumbnailSize) + (numSamples - 1) * 4}px`,
    minWidth: `${numSamples * 128 + (numSamples - 1) * 4}px`,
  };
  const controlClass =
    'min-h-9 max-sm:min-h-11 rounded border border-gray-300 bg-white px-2 py-1 text-sm text-gray-900 dark:border-gray-600 dark:bg-gray-900 dark:text-gray-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-blue-500';

  return (
    <div className="absolute top-[80px] left-0 right-0 bottom-0 flex flex-col">
      <div className="flex flex-wrap items-center gap-3 border-b border-gray-300 bg-gray-100 px-3 py-2 text-sm text-gray-900 dark:border-gray-700 dark:bg-gray-800 dark:text-gray-100">
        <label className="flex items-center gap-2">
          Thumbnail size
          <select
            aria-label="Thumbnail size"
            value={thumbnailSize}
            onChange={e => setThumbnailSize(e.target.value)}
            className={controlClass}
          >
            <option value="fit">Fit</option>
            <option value="144">Small</option>
            <option value="224">Medium</option>
            <option value="320">Large</option>
          </select>
        </label>
        <label className="flex items-center gap-2 max-sm:min-h-11">
          <input
            type="checkbox"
            checked={showMetadata}
            onChange={e => setShowMetadata(e.target.checked)}
            className="h-4 w-4 accent-blue-500"
          />
          Show metadata
        </label>
        <label className="flex items-center gap-2">
          Training step
          <select
            aria-label="Training step filter"
            value={stepFilter}
            onChange={e => setStepFilter(e.target.value)}
            className={controlClass}
          >
            <option value="all">All steps</option>
            <option value="latest">Latest step</option>
            {steps.map(step => (
              <option key={step} value={step}>
                {step.toLocaleString()}
              </option>
            ))}
          </select>
        </label>
        <form
          className="flex items-center gap-2"
          onSubmit={e => {
            e.preventDefault();
            if (!jumpTarget) return;
            setStepFilter('all');
            setPendingJump(jumpTarget);
          }}
        >
          <select
            aria-label="Jump to training step"
            value={jumpTarget}
            onChange={e => setJumpTarget(e.target.value)}
            className={controlClass}
          >
            <option value="">Jump to step</option>
            {rows.map((row, index) => (
              <option key={row.key} value={row.key}>
                {rowSteps.get(row.key)?.toLocaleString() ?? 'Unknown step'} · row {index + 1}
              </option>
            ))}
          </select>
          <button type="submit" disabled={!jumpTarget} className={`${controlClass} disabled:opacity-50`}>
            Go
          </button>
        </form>
        <button
          type="button"
          disabled={!rows.some(row => row.paths.some(Boolean))}
          onClick={() => setComparisonOpen(true)}
          className={`${controlClass} disabled:opacity-50`}
        >
          Compare samples
        </button>
        {selectedSet.size > 0 && (
          <button type="button" onClick={deleteSelected} className={controlClass}>
            Delete selected ({selectedSet.size})
          </button>
        )}
        <button
          type="button"
          onClick={scrollToTop}
          disabled={visibleRows.length === 0}
          className={`${controlClass} disabled:opacity-50`}
          aria-label="Scroll to first sample row"
          title="First row"
        >
          <FaCaretUp />
        </button>
        <button
          type="button"
          onClick={scrollToBottom}
          disabled={visibleRows.length === 0}
          className={`${controlClass} disabled:opacity-50`}
          aria-label="Scroll to last sample row"
          title="Last row"
        >
          <FaCaretDown />
        </button>
      </div>
      <div ref={scrollParentCallback} className="min-h-0 flex-1 overflow-auto">
        <div className="pb-4" style={matrixStyle}>
          {PageInfoContent}
          {rows.length > 0 && (
            <div
              ref={headerRef}
              className="sticky top-0 z-[5] grid gap-1 border-b border-gray-300 bg-gray-100 dark:border-gray-700 dark:bg-gray-800"
              style={gridStyle}
            >
              {Array.from({ length: numSamples }, (_, column) => {
                const item = sampleItems[column];
                const weightType = hasEma ? (item?.raw_weights ? 'Raw' : 'EMA') : 'Training';
                return (
                  <div
                    key={column}
                    className="min-w-0 px-2 py-2 text-xs text-gray-700 dark:text-gray-300"
                    title={item?.prompt}
                  >
                    <div className="font-semibold text-gray-900 dark:text-gray-100">
                      Sample {column + 1} · {weightType}
                    </div>
                    <div className="line-clamp-2 mt-1 break-words">
                      {item?.prompt?.replace(/^integrated_multimodal_description:\s*(?:\[Shot\s+\d+\]\s*)?/i, '') ||
                        'Prompt unavailable'}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
          {visibleRows.length > 0 && scrollParent && (
            <Virtuoso
              key={stepFilter}
              ref={virtuosoRef}
              customScrollParent={scrollParent}
              totalCount={visibleRows.length}
              initialTopMostItemIndex={visibleRows.length - 1}
              followOutput={stepFilter === 'all' || stepFilter === 'latest' ? 'auto' : false}
              increaseViewportBy={400}
              computeItemKey={index => visibleRows[index].key}
              itemContent={index => {
                const row = visibleRows[index];
                return (
                  <div className="pb-1">
                    <div className="px-2 py-1 text-xs text-gray-600 dark:text-gray-400">
                      Training step {rowSteps.get(row.key)?.toLocaleString() ?? 'Unknown'}
                    </div>
                    <div className="grid gap-1" style={gridStyle}>
                      {row.paths.map((sample, column) =>
                        sample ? (
                          <SampleImageCard
                            key={sample}
                            imageUrl={sample}
                            metadata={getSampleMetadata(sample, sampleConfig, column)}
                            showMetadata={showMetadata}
                            alt={`Sample ${column + 1}`}
                            onClick={e => handleCardClick(sample, e)}
                            selected={selectedSet.has(sample) || selectedSamplePath === sample}
                            observerRoot={scrollParent}
                          />
                        ) : (
                          <div
                            key={`empty-${column}`}
                            className="flex aspect-square items-center justify-center rounded border border-dashed border-gray-300 bg-gray-50 p-2 text-center text-sm text-gray-600 dark:border-gray-700 dark:bg-gray-900 dark:text-gray-400"
                            aria-label={`Sample ${column + 1}: ${row.missing[column] === 'deleted' ? 'Deleted' : row.missing[column] === 'not-generated' ? 'Not generated' : 'Not available'}`}
                          >
                            {row.missing[column] === 'deleted'
                              ? 'Deleted'
                              : row.missing[column] === 'not-generated'
                                ? 'Not generated'
                                : 'Not available'}
                          </div>
                        ),
                      )}
                    </div>
                  </div>
                );
              }}
            />
          )}
          {rows.length > 0 && visibleRows.length === 0 && (
            <p className="p-4 text-sm text-gray-600 dark:text-gray-400">No samples at this training step.</p>
          )}
        </div>
      </div>
      <Dialog open={deleteProgress !== null} onClose={() => {}} className="relative z-20">
        <DialogBackdrop className="fixed inset-0 bg-gray-900/75" />
        <div className="fixed inset-0 z-10 flex items-center justify-center p-4">
          <DialogPanel className="w-full max-w-sm rounded-lg bg-gray-800 p-6 shadow-xl text-gray-200">
            <DialogTitle as="h3" className="text-base font-semibold flex items-center gap-2">
              <LuLoader className="animate-spin" />
              Deleting Samples
            </DialogTitle>
            <p className="mt-2 text-sm text-gray-400">
              {deleteProgress?.done ?? 0} / {deleteProgress?.total ?? 0}
            </p>
            <div className="mt-3 h-2 w-full rounded bg-gray-700 overflow-hidden">
              <div
                className="h-full bg-blue-500 transition-all duration-150"
                style={{
                  width: `${deleteProgress && deleteProgress.total > 0 ? (deleteProgress.done / deleteProgress.total) * 100 : 0}%`,
                }}
              />
            </div>
          </DialogPanel>
        </div>
      </Dialog>
      <SampleImageViewer
        imgPath={selectedSamplePath}
        numSamples={numSamples}
        sampleImages={sortedSampleImages}
        onChange={setPath => setSelectedSamplePath(setPath)}
        sampleConfig={sampleConfig}
        refreshSampleImages={refreshSampleImages}
        matrixRows={visibleRows}
        showMetadata={showMetadata}
      />
      <SampleComparisonViewer
        open={comparisonOpen}
        onClose={() => setComparisonOpen(false)}
        jobId={job.id}
        jobName={job.name}
        jobConfig={job.job_config}
        rows={rows}
        sampleConfig={sampleConfig}
        showMetadata={showMetadata}
        hasEma={hasEma}
      />
    </div>
  );
}
