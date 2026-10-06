'use client';

import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react';
import type { RefObject } from 'react';
import { Dialog, DialogBackdrop, DialogDescription, DialogPanel, DialogTitle } from '@headlessui/react';
import type { SampleConfig } from '@/types';
import { apiClient } from '@/utils/api';
import { encodeFilePathForUrl, isAudio, isText, isVideo } from '@/utils/basic';
import { buildSampleMatrix, getSampleItems } from '@/utils/sampleImages';
import type { SampleRow } from '@/utils/sampleImages';
import usePollLoop from '@/hooks/usePollLoop';
import SampleMetadataOverlay from './SampleMetadataOverlay';
import {
  applyNearestCrossJobStep,
  createCrossJobSelection,
  getComparisonColumns,
  getComparisonNotice,
  getComparisonRowLabels,
  getComparisonRowStep,
  getComparisonWeightLabel,
  getCrossJobStepNotice,
  getDefaultComparisonColumns,
  getTrainingPace,
  getTrainingPaceNotice,
  nearestStepOffer,
  readComparisonJob,
  reconcileCrossJobSelection,
  selectCrossJobColumn,
  selectCrossJobRow,
} from './sampleComparison';
import type { ComparisonColumn, CrossJobSelection } from './sampleComparison';
import { createComparisonPlayback, EMPTY_COMPARISON_PLAYBACK } from './sampleComparisonPlayback';
import type { ComparisonPlayback, ComparisonPlaybackState } from './sampleComparisonPlayback';

interface ListedJob {
  id: string;
  name: string;
  status: string;
  job_config: string;
}

interface SampleEvidence {
  samples: string[];
  plannedSamples?: string[];
  deletedSamples?: string[];
}

interface OtherLoad {
  jobId: string;
  status: 'loading' | 'error' | 'success';
  evidence: SampleEvidence | null;
}

interface Props {
  open: boolean;
  onClose: () => void;
  jobId: string;
  jobName: string;
  jobConfig: string;
  rows: SampleRow[];
  sampleConfig: SampleConfig | null;
  showMetadata: boolean;
  hasEma: boolean;
}

const FIELD_CLASS =
  'min-h-11 w-full rounded-md border border-gray-600 bg-gray-900 px-3 py-2 text-sm text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500';
const BUTTON_CLASS =
  'min-h-11 rounded-md border border-gray-600 px-3 py-2 text-sm text-gray-100 hover:bg-gray-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 disabled:cursor-not-allowed disabled:opacity-50';

function formatTime(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`;
}

interface PaneProps {
  column?: ComparisonColumn;
  side: 'Left' | 'Right';
  hasEma: boolean;
  showMetadata: boolean;
  synchronized: boolean;
  synchronizedPlaying: boolean;
  videoRef: RefObject<HTMLVideoElement | null>;
  onVideoMount: () => void;
  missing?: SampleRow['missing'][number];
  jobLabel?: string;
  emptyMessage?: string;
}

function ComparisonPane({
  column,
  side,
  hasEma,
  showMetadata,
  synchronized,
  synchronizedPlaying,
  videoRef,
  onVideoMount,
  missing,
  jobLabel,
  emptyMessage,
}: PaneProps) {
  const path = column?.path ?? null;
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [text, setText] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const src = path ? `/api/img/${encodeFilePathForUrl(path)}` : '';
  const displayMetadata = showMetadata && !playing && !synchronizedPlaying;
  const bindVideo = useCallback(
    (video: HTMLVideoElement | null) => {
      videoRef.current = video;
      onVideoMount();
    },
    [videoRef, onVideoMount],
  );

  useEffect(() => {
    const video = videoRef.current;
    const audio = audioRef.current;
    return () => {
      video?.pause();
      audio?.pause();
    };
  }, [path, videoRef]);

  useEffect(() => {
    if (!path || !isText(path)) return;
    const controller = new AbortController();
    fetch(src, { signal: controller.signal })
      .then(response => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.text();
      })
      .then(setText)
      .catch(cause => {
        if (cause?.name !== 'AbortError') setError(`Could not load text sample: ${cause.message ?? 'Unknown error'}`);
      });
    return () => controller.abort();
  }, [path, src]);

  const unavailableReason =
    emptyMessage ??
    (missing === 'deleted'
      ? 'This sample was deleted.'
      : missing === 'not-generated'
        ? 'This planned sample has not been generated.'
        : 'No media is available for this sample in the selected training row.');

  return (
    <section
      aria-label={`${side} comparison pane`}
      className="min-w-0 overflow-hidden rounded-md border border-gray-700"
    >
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-gray-700 bg-gray-900 px-3 py-2 text-sm">
        <span>{jobLabel ?? `${side}${column ? ` · Sample #${column.index + 1}` : ''}`}</span>
        {column && <span className="text-gray-300">{getComparisonWeightLabel(column, hasEma)}</span>}
      </div>
      <div className="relative flex min-h-64 items-center justify-center bg-black sm:min-h-80">
        {!path ? (
          <div className="p-6 text-center text-sm text-gray-300">
            <p className="font-semibold text-gray-100">Sample not available</p>
            <p className="mt-2">{unavailableReason}</p>
          </div>
        ) : isText(path) ? (
          <div
            className={`max-h-[55vh] w-full overflow-auto whitespace-pre-wrap break-words p-4 text-sm text-gray-100 ${displayMetadata ? 'pb-28' : ''}`}
          >
            {text ?? (error ? '' : 'Loading text sample…')}
          </div>
        ) : isAudio(path) ? (
          <div className="w-full p-4">
            <p className="mb-4 break-all text-sm text-gray-300">{path.split(/[\\/]/).pop()}</p>
            <audio
              ref={audioRef}
              src={src}
              controls
              preload="metadata"
              aria-label={`${side} audio sample`}
              className="w-full"
              onPlay={() => setPlaying(true)}
              onPause={() => setPlaying(false)}
              onEnded={() => setPlaying(false)}
              onError={() => setError('Could not load or decode this audio sample.')}
            />
          </div>
        ) : isVideo(path) ? (
          <video
            ref={bindVideo}
            src={src}
            muted
            playsInline
            preload="auto"
            controls={!synchronized}
            disablePictureInPicture={synchronized}
            aria-label={`${side} video sample`}
            className="max-h-[55vh] w-full object-contain"
            onPlay={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
            onEnded={() => setPlaying(false)}
            onError={() => setError('Could not load or decode this video sample.')}
          />
        ) : (
          <img
            src={src}
            alt={`${side}, sample #${(column?.index ?? 0) + 1}: ${column?.prompt ?? 'Prompt unavailable'}`}
            className="max-h-[55vh] w-full object-contain"
            onError={() => setError('Could not load this image sample.')}
          />
        )}
        {path && column && displayMetadata && !error && <SampleMetadataOverlay metadata={column.metadata} />}
        {error && (
          <p role="alert" className="absolute inset-x-0 top-0 bg-gray-950 p-4 text-sm text-red-300">
            {error}
          </p>
        )}
      </div>
      <p className="max-h-28 overflow-auto whitespace-pre-wrap break-words border-t border-gray-700 px-3 py-2 text-sm text-gray-300">
        <span className="text-gray-400">Prompt: </span>
        {column?.prompt ?? 'Unavailable'}
      </p>
    </section>
  );
}

function comparisonPaneLabel(side: string, name: string, step: number | null, sampleNumber: number | null): string {
  const parts = [side, name, step === null ? 'No step' : `Step ${step.toLocaleString()}`];
  if (sampleNumber !== null) parts.push(`Sample #${sampleNumber}`);
  return parts.join(' · ');
}

export default function SampleComparisonViewer({
  open,
  onClose,
  jobId,
  jobName,
  jobConfig,
  rows,
  sampleConfig,
  showMetadata,
  hasEma,
}: Props) {
  const id = useId();
  const [selection, setSelection] = useState<{ rowKey: string | null; columns: [number, number] }>({
    rowKey: null,
    columns: [-1, -1],
  });
  const [rightJobId, setRightJobId] = useState(jobId);
  const [trainJobs, setTrainJobs] = useState<ListedJob[]>([]);
  const [jobsStatus, setJobsStatus] = useState<'idle' | 'loading' | 'error' | 'success'>('idle');
  const [jobsError, setJobsError] = useState<string | null>(null);
  const [otherLoad, setOtherLoad] = useState<OtherLoad | null>(null);
  const [syncSteps, setSyncSteps] = useState(true);
  const [matchPrompt, setMatchPrompt] = useState(true);
  const [crossSelection, setCrossSelection] = useState<CrossJobSelection | null>(null);
  const [playback, setPlayback] = useState<ComparisonPlaybackState>(EMPTY_COMPARISON_PLAYBACK);
  const leftVideo = useRef<HTMLVideoElement | null>(null);
  const rightVideo = useRef<HTMLVideoElement | null>(null);
  const playbackRef = useRef<ComparisonPlayback | null>(null);
  const initializedJob = useRef<string | null>(null);
  const seenRightJob = useRef(rightJobId);
  const preferredLeftRow = useRef<string | null>(null);
  const activeRightJob = useRef(rightJobId);
  activeRightJob.current = rightJobId;
  const [videoMountRevision, setVideoMountRevision] = useState(0);
  const onVideoMount = useCallback(() => setVideoMountRevision(revision => revision + 1), []);
  const crossJob = rightJobId !== jobId;
  const currentPace = useMemo(() => getTrainingPace(jobConfig), [jobConfig]);
  const otherJob = trainJobs.find(job => job.id === rightJobId);
  const otherParsed = useMemo(() => readComparisonJob(otherJob?.job_config), [otherJob?.job_config]);
  const otherReady = otherLoad?.jobId === rightJobId && otherLoad.status === 'success' && otherLoad.evidence !== null;
  const otherRows = useMemo(() => {
    if (!otherReady || !otherLoad?.evidence) return [];
    const items = getSampleItems(otherParsed.sampleConfig);
    return buildSampleMatrix(otherLoad.evidence.samples ?? [], otherParsed.sampleConfig, Math.max(items.length, 1), {
      plannedSamples: otherLoad.evidence.plannedSamples ?? [],
      deletedSamples: otherLoad.evidence.deletedSamples ?? [],
    });
  }, [otherReady, otherLoad, otherParsed.sampleConfig]);
  const rowLabels = useMemo(() => getComparisonRowLabels(rows), [rows]);
  const otherRowLabels = useMemo(() => getComparisonRowLabels(otherRows), [otherRows]);
  const sameRow = rows.find(candidate => candidate.key === selection.rowKey) ?? rows[rows.length - 1];
  const sameColumns = useMemo(
    () => (sameRow ? getComparisonColumns(sameRow, sampleConfig) : []),
    [sameRow, sampleConfig],
  );
  const sameSelected = sameRow?.key === selection.rowKey ? selection.columns : getDefaultComparisonColumns(sameColumns);
  const leftCrossRow = rows.find(row => row.key === crossSelection?.left.rowKey) ?? null;
  const rightCrossRow = otherRows.find(row => row.key === crossSelection?.right.rowKey) ?? null;
  const leftCrossColumns = useMemo(
    () => getComparisonColumns(leftCrossRow ?? { key: 'missing', paths: [] }, sampleConfig),
    [leftCrossRow, sampleConfig],
  );
  const rightCrossColumns = useMemo(
    () => getComparisonColumns(rightCrossRow ?? { key: 'missing', paths: [] }, otherParsed.sampleConfig),
    [rightCrossRow, otherParsed.sampleConfig],
  );
  const left = crossJob ? leftCrossColumns[crossSelection?.left.column ?? -1] : sameColumns[sameSelected[0]];
  const right = crossJob ? rightCrossColumns[crossSelection?.right.column ?? -1] : sameColumns[sameSelected[1]];
  const leftPath = left?.path ?? null;
  const rightPath = right?.path ?? null;
  const synchronized = Boolean(leftPath && rightPath && isVideo(leftPath) && isVideo(rightPath));
  const leftStep = leftCrossRow ? getComparisonRowStep(leftCrossRow) : null;
  const rightStep = rightCrossRow ? getComparisonRowStep(rightCrossRow) : null;
  const otherName = otherJob?.name ?? 'The other job';
  const otherFailed = crossJob && otherLoad?.jobId === rightJobId && otherLoad.status === 'error';
  const stepNotice =
    !crossJob || !otherReady
      ? null
      : otherRows.length === 0
        ? `${otherName} has no samples.`
        : getCrossJobStepNotice(leftStep, rightStep, jobName, otherName);
  const paceNotice = crossJob && otherReady ? getTrainingPaceNotice(currentPace, otherParsed.pace) : null;
  const baseNotice =
    !crossJob || (otherReady && leftCrossRow && rightCrossRow) ? getComparisonNotice(left, right) : null;
  const notice = [baseNotice, stepNotice, paceNotice].filter(Boolean).join(' ');
  const offer = crossJob && otherReady && crossSelection ? nearestStepOffer(crossSelection, rows, otherRows) : null;
  const crossPending = crossJob && !otherFailed && (!otherReady || crossSelection === null);
  const rightEmptyMessage = !crossJob
    ? undefined
    : otherFailed
      ? 'Could not load samples for this job.'
      : crossPending
        ? 'Loading samples…'
        : rightCrossRow
          ? undefined
          : otherRows.length === 0
            ? `${otherName} has no samples.`
            : leftStep !== null
              ? `No sample at step ${leftStep.toLocaleString()}.`
              : 'No step selected.';
  const leftEmptyMessage =
    !crossJob || leftCrossRow
      ? undefined
      : crossPending
        ? 'Loading samples…'
        : rightStep !== null
          ? `No sample at step ${rightStep.toLocaleString()}.`
          : 'No step selected.';

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setJobsStatus('loading');
    apiClient
      .get('/api/jobs', { params: { job_type: 'train' } })
      .then(response => {
        if (cancelled) return;
        setTrainJobs(response.data.jobs ?? []);
        setJobsError(null);
        setJobsStatus('success');
      })
      .catch(() => {
        if (cancelled) return;
        setJobsError('Could not load other jobs.');
        setJobsStatus('error');
      });
    return () => {
      cancelled = true;
    };
  }, [open]);

  useEffect(() => {
    setRightJobId(jobId);
  }, [jobId]);

  useEffect(() => {
    if (seenRightJob.current === rightJobId) return;
    seenRightJob.current = rightJobId;
    setSyncSteps(true);
    setMatchPrompt(true);
  }, [rightJobId]);

  useEffect(() => {
    const key = crossJob ? crossSelection?.left.rowKey : selection.rowKey;
    if (key) preferredLeftRow.current = key;
  }, [crossJob, crossSelection?.left.rowKey, selection.rowKey]);

  usePollLoop(
    () => {
      if (!open || rightJobId === jobId) return;
      const requestedId = rightJobId;
      return apiClient
        .get(`/api/jobs/${encodeURIComponent(requestedId)}/samples`)
        .then(response => {
          if (activeRightJob.current !== requestedId) return;
          setOtherLoad({ jobId: requestedId, status: 'success', evidence: response.data });
        })
        .catch(() => {
          if (activeRightJob.current !== requestedId) return;
          setOtherLoad({ jobId: requestedId, status: 'error', evidence: null });
        });
    },
    open && crossJob ? 5000 : null,
    [open, crossJob, rightJobId, jobId],
  );

  useEffect(() => {
    if (!open || !rows.length || rows.some(candidate => candidate.key === selection.rowKey)) return;
    const initialRow =
      [...rows].reverse().find(candidate => {
        const options = getComparisonColumns(candidate, sampleConfig);
        return (
          options.some(option => option.path && option.isRaw === true) &&
          options.some(option => option.path && option.isRaw === false)
        );
      }) ?? rows[rows.length - 1];
    setSelection({
      rowKey: initialRow.key,
      columns: getDefaultComparisonColumns(getComparisonColumns(initialRow, sampleConfig)),
    });
  }, [open, rows, sampleConfig, selection.rowKey]);

  useEffect(() => {
    const jobsSettled = Boolean(otherJob) || jobsStatus === 'success' || jobsStatus === 'error';
    if (!open || !crossJob || !otherReady || !jobsSettled) return;
    const token = `${rightJobId}:${otherJob?.job_config ?? ''}`;
    if (initializedJob.current === token) return;
    initializedJob.current = token;
    setCrossSelection(
      createCrossJobSelection(rows, sampleConfig, otherRows, otherParsed.sampleConfig, preferredLeftRow.current),
    );
  }, [
    open,
    crossJob,
    otherReady,
    jobsStatus,
    otherJob,
    rightJobId,
    rows,
    sampleConfig,
    otherRows,
    otherParsed.sampleConfig,
  ]);

  useEffect(() => {
    if (!crossJob || !otherReady || initializedJob.current !== `${rightJobId}:${otherJob?.job_config ?? ''}`) return;
    setCrossSelection(current => (current ? reconcileCrossJobSelection(current, rows, otherRows, syncSteps) : current));
  }, [crossJob, otherReady, rightJobId, otherJob?.job_config, rows, otherRows, syncSteps]);

  useEffect(() => {
    setPlayback(EMPTY_COMPARISON_PLAYBACK);
    if (!open || !synchronized || !leftVideo.current || !rightVideo.current) return;
    const controller = createComparisonPlayback([leftVideo.current, rightVideo.current], setPlayback);
    playbackRef.current = controller;
    return () => {
      controller.dispose();
      if (playbackRef.current === controller) playbackRef.current = null;
    };
  }, [open, synchronized, leftPath, rightPath, videoMountRevision]);

  function close() {
    playbackRef.current?.pause();
    leftVideo.current?.pause();
    rightVideo.current?.pause();
    onClose();
  }

  function changeRow(key: string) {
    const nextRow = rows.find(candidate => candidate.key === key);
    if (!nextRow) return;
    playbackRef.current?.pause();
    setSelection({ rowKey: key, columns: getDefaultComparisonColumns(getComparisonColumns(nextRow, sampleConfig)) });
  }

  function changeColumn(side: 0 | 1, index: number) {
    playbackRef.current?.pause();
    if (!crossJob) {
      if (!sameRow) return;
      const next: [number, number] = [...sameSelected];
      next[side] = index;
      setSelection({ rowKey: sameRow.key, columns: next });
      return;
    }
    setCrossSelection(current =>
      current
        ? selectCrossJobColumn(
            current,
            side === 0 ? 'left' : 'right',
            index,
            leftCrossColumns,
            rightCrossColumns,
            matchPrompt,
          )
        : current,
    );
  }

  function changeCrossRow(side: 'left' | 'right', key: string) {
    if (!key) return;
    playbackRef.current?.pause();
    setCrossSelection(current =>
      current ? selectCrossJobRow(current, side, key, rows, otherRows, syncSteps) : current,
    );
  }

  function changeRightJob(id: string) {
    playbackRef.current?.pause();
    setRightJobId(id);
  }

  function enableSync(on: boolean) {
    setSyncSteps(on);
    if (!on) return;
    playbackRef.current?.pause();
    setCrossSelection(current => {
      if (!current) return current;
      if (current.left.rowKey) return selectCrossJobRow(current, 'left', current.left.rowKey, rows, otherRows, true);
      if (current.right.rowKey) return selectCrossJobRow(current, 'right', current.right.rowKey, rows, otherRows, true);
      return current;
    });
  }

  function enableMatchPrompt(on: boolean) {
    setMatchPrompt(on);
    if (!on) return;
    playbackRef.current?.pause();
    setCrossSelection(current =>
      current
        ? selectCrossJobColumn(current, 'left', current.left.column, leftCrossColumns, rightCrossColumns, true)
        : current,
    );
  }

  function useNearest() {
    if (!crossSelection) return;
    const next = applyNearestCrossJobStep(crossSelection, rows, otherRows);
    if (!next) return;
    playbackRef.current?.pause();
    setSyncSteps(false);
    setCrossSelection(next);
  }

  const columnValues: [number, number] = crossJob
    ? [crossSelection?.left.column ?? -1, crossSelection?.right.column ?? -1]
    : sameSelected;
  const columnSets = crossJob ? [leftCrossColumns, rightCrossColumns] : [sameColumns, sameColumns];
  const paneHasEma = crossJob ? [hasEma, otherParsed.hasEma] : [hasEma, hasEma];
  const nearestName = offer?.side === 'left' ? jobName : otherName;

  return (
    <Dialog open={open} onClose={close} className="relative z-30">
      <DialogBackdrop className="fixed inset-0 bg-gray-900/75" />
      <div className="fixed inset-0 overflow-y-auto p-2 sm:p-4">
        <div className="flex min-h-full items-center justify-center">
          <DialogPanel className="w-full max-w-7xl rounded-lg border border-gray-700 bg-gray-800 text-gray-100">
            <div className="flex items-start justify-between gap-4 border-b border-gray-700 p-4">
              <div>
                <DialogTitle className="text-lg font-semibold">Compare samples</DialogTitle>
                <DialogDescription className="mt-1 text-sm text-gray-400">
                  {crossJob
                    ? 'Compare this job with another run at a training step. A step that was not sampled stays empty.'
                    : 'Choose two samples from this training run, or pick another job on the right.'}
                </DialogDescription>
              </div>
              <button type="button" onClick={close} className={BUTTON_CLASS} data-autofocus>
                Close
              </button>
            </div>
            {open && (
              <div className="space-y-4 p-4">
                <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                  <div className="min-w-0">
                    <span className="mb-2 block text-sm font-medium">Left job</span>
                    <p className={FIELD_CLASS}>{jobName}</p>
                  </div>
                  <div className="min-w-0">
                    <label htmlFor={`${id}-job`} className="mb-2 block text-sm font-medium">
                      Right job
                    </label>
                    <select
                      id={`${id}-job`}
                      value={rightJobId}
                      onChange={event => changeRightJob(event.target.value)}
                      className={FIELD_CLASS}
                    >
                      <option value={jobId}>{jobName} (this job)</option>
                      {rightJobId !== jobId && !trainJobs.some(job => job.id === rightJobId) && (
                        <option value={rightJobId}>{otherName}</option>
                      )}
                      {trainJobs
                        .filter(job => job.id !== jobId)
                        .map(job => (
                          <option key={job.id} value={job.id}>
                            {job.name}
                            {job.status === 'running' ? ' · running' : ''}
                          </option>
                        ))}
                    </select>
                    {jobsError && (
                      <p role="alert" className="mt-2 text-sm text-red-300">
                        {jobsError}
                      </p>
                    )}
                  </div>
                </div>
                {crossJob && (
                  <div className="flex flex-wrap gap-x-6 gap-y-2">
                    <label
                      className="flex min-h-11 items-center gap-2 text-sm"
                      title="Keep both panes on the same training step. A missing step stays empty."
                    >
                      <input
                        type="checkbox"
                        checked={syncSteps}
                        onChange={event => enableSync(event.target.checked)}
                        className="h-4 w-4 accent-blue-500"
                      />
                      Sync steps
                    </label>
                    <label
                      className="flex min-h-11 items-center gap-2 text-sm"
                      title="Selecting a sample also selects the same prompt on the other job."
                    >
                      <input
                        type="checkbox"
                        checked={matchPrompt}
                        onChange={event => enableMatchPrompt(event.target.checked)}
                        className="h-4 w-4 accent-blue-500"
                      />
                      Match prompt
                    </label>
                  </div>
                )}
                {!crossJob && (
                  <div>
                    <label htmlFor={`${id}-row`} className="mb-2 block text-sm font-medium">
                      Training row
                    </label>
                    <select
                      id={`${id}-row`}
                      value={sameRow?.key ?? ''}
                      onChange={event => changeRow(event.target.value)}
                      disabled={!rows.length}
                      className={FIELD_CLASS}
                    >
                      {!rows.length && <option value="">No training rows available</option>}
                      {rows.map((candidate, index) => (
                        <option key={candidate.key} value={candidate.key}>
                          {rowLabels[index]}
                        </option>
                      ))}
                    </select>
                  </div>
                )}
                {crossJob && (
                  <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                    {(['Left', 'Right'] as const).map(side => {
                      const paneRows = side === 'Left' ? rows : otherRows;
                      const labels = side === 'Left' ? rowLabels : otherRowLabels;
                      const rowKey =
                        side === 'Left'
                          ? (crossSelection?.left.rowKey ?? null)
                          : (crossSelection?.right.rowKey ?? null);
                      const requestedStep = side === 'Left' ? rightStep : leftStep;
                      return (
                        <div key={side} className="min-w-0">
                          <label htmlFor={`${id}-${side}-step`} className="mb-2 block text-sm font-medium">
                            {side} training step
                          </label>
                          <select
                            id={`${id}-${side}-step`}
                            value={rowKey ?? ''}
                            onChange={event => changeCrossRow(side === 'Left' ? 'left' : 'right', event.target.value)}
                            disabled={crossSelection === null || (side === 'Right' && !otherReady)}
                            className={FIELD_CLASS}
                          >
                            {rowKey === null && (
                              <option value="">
                                {syncSteps && requestedStep !== null
                                  ? `Step ${requestedStep.toLocaleString()} · Not sampled`
                                  : 'No step selected'}
                              </option>
                            )}
                            {paneRows.map((candidate, index) => (
                              <option key={candidate.key} value={candidate.key}>
                                {labels[index]}
                              </option>
                            ))}
                          </select>
                        </div>
                      );
                    })}
                  </div>
                )}
                <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                  {(['Left', 'Right'] as const).map((side, paneIndex) => (
                    <div key={side} className="min-w-0">
                      <label htmlFor={`${id}-${side}`} className="mb-2 block text-sm font-medium">
                        {side} sample · sample # / weights / prompt
                      </label>
                      <select
                        id={`${id}-${side}`}
                        value={columnValues[paneIndex]}
                        onChange={event => changeColumn(paneIndex as 0 | 1, Number(event.target.value))}
                        disabled={
                          !columnSets[paneIndex].length ||
                          (crossJob && (crossSelection === null || (paneIndex === 1 && !otherReady)))
                        }
                        className={FIELD_CLASS}
                      >
                        <option value={-1}>Not available</option>
                        {columnSets[paneIndex].map(column => (
                          <option key={column.index} value={column.index}>
                            {`Sample #${column.index + 1} · ${getComparisonWeightLabel(column, paneHasEma[paneIndex])} · ${column.prompt ?? 'Prompt unavailable'}${column.path ? '' : ' · Not available'}`}
                          </option>
                        ))}
                      </select>
                    </div>
                  ))}
                </div>
                {notice && (
                  <p role="status" className="text-sm text-amber-200">
                    {notice}
                  </p>
                )}
                {offer && (
                  <button type="button" className={BUTTON_CLASS} onClick={useNearest}>
                    {`Show nearest step on ${nearestName} (${offer.nearestStep.toLocaleString()}, selected step is ${offer.selectedStep.toLocaleString()})`}
                  </button>
                )}
                <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                  <ComparisonPane
                    key={`left:${leftPath ?? `missing-${columnValues[0]}`}`}
                    side="Left"
                    column={left}
                    hasEma={paneHasEma[0]}
                    showMetadata={showMetadata}
                    synchronized={synchronized}
                    synchronizedPlaying={playback.playing}
                    videoRef={leftVideo}
                    onVideoMount={onVideoMount}
                    missing={crossJob ? leftCrossRow?.missing[columnValues[0]] : sameRow?.missing[sameSelected[0]]}
                    jobLabel={
                      crossJob
                        ? comparisonPaneLabel('Left', jobName, leftStep, left ? left.index + 1 : null)
                        : undefined
                    }
                    emptyMessage={leftEmptyMessage}
                  />
                  <ComparisonPane
                    key={`right:${rightJobId}:${rightPath ?? `missing-${columnValues[1]}`}`}
                    side="Right"
                    column={crossJob && !otherReady ? undefined : right}
                    hasEma={paneHasEma[1]}
                    showMetadata={showMetadata}
                    synchronized={synchronized}
                    synchronizedPlaying={playback.playing}
                    videoRef={rightVideo}
                    onVideoMount={onVideoMount}
                    missing={crossJob ? rightCrossRow?.missing[columnValues[1]] : sameRow?.missing[sameSelected[1]]}
                    jobLabel={
                      crossJob
                        ? comparisonPaneLabel(
                            'Right',
                            otherName,
                            rightStep,
                            otherReady && right ? right.index + 1 : null,
                          )
                        : undefined
                    }
                    emptyMessage={rightEmptyMessage}
                  />
                </div>
                {synchronized && (
                  <div className="space-y-3 border-t border-gray-700 pt-4">
                    <div className="flex flex-wrap items-center gap-3">
                      <button
                        type="button"
                        className={BUTTON_CLASS}
                        disabled={!playback.duration}
                        onClick={() =>
                          playback.playing || playback.pending
                            ? playbackRef.current?.pause()
                            : playbackRef.current?.play()
                        }
                      >
                        {playback.playing || playback.pending ? 'Pause both' : 'Play both'}
                      </button>
                      <button
                        type="button"
                        className={BUTTON_CLASS}
                        disabled={!playback.duration}
                        onClick={() => playbackRef.current?.restart()}
                      >
                        Restart both
                      </button>
                      <span role="status" className="text-sm text-gray-300">
                        {playback.pending
                          ? 'Waiting for both videos…'
                          : `${formatTime(playback.currentTime)} / ${formatTime(playback.duration)}`}
                      </span>
                    </div>
                    <label htmlFor={`${id}-seek`} className="block text-sm text-gray-300">
                      Seek both videos
                    </label>
                    <input
                      id={`${id}-seek`}
                      type="range"
                      min={0}
                      max={playback.duration || 1}
                      step={0.01}
                      value={playback.currentTime}
                      disabled={!playback.duration}
                      onChange={event => playbackRef.current?.seek(Number(event.target.value))}
                      aria-valuetext={`${formatTime(playback.currentTime)} of ${formatTime(playback.duration)}`}
                      className="min-h-11 w-full accent-blue-500 disabled:opacity-50"
                    />
                    <p className="text-xs text-gray-400">
                      Videos are muted. Shared playback stops at the shorter video’s duration.
                    </p>
                    {playback.error && (
                      <p role="alert" className="text-sm text-red-300">
                        {playback.error}
                      </p>
                    )}
                  </div>
                )}
              </div>
            )}
          </DialogPanel>
        </div>
      </div>
    </Dialog>
  );
}
