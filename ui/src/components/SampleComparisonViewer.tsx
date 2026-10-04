'use client';

import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react';
import type { RefObject } from 'react';
import { Dialog, DialogBackdrop, DialogDescription, DialogPanel, DialogTitle } from '@headlessui/react';
import type { SampleConfig } from '@/types';
import type { SampleRow } from '@/utils/sampleImages';
import { encodeFilePathForUrl, isAudio, isText, isVideo } from '@/utils/basic';
import SampleMetadataOverlay from './SampleMetadataOverlay';
import {
  getComparisonColumns,
  getComparisonNotice,
  getComparisonRowLabels,
  getComparisonWeightLabel,
  getDefaultComparisonColumns,
} from './sampleComparison';
import type { ComparisonColumn } from './sampleComparison';
import { createComparisonPlayback, EMPTY_COMPARISON_PLAYBACK } from './sampleComparisonPlayback';
import type { ComparisonPlayback, ComparisonPlaybackState } from './sampleComparisonPlayback';

interface Props {
  open: boolean;
  onClose: () => void;
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
    missing === 'deleted'
      ? 'This sample was deleted.'
      : missing === 'not-generated'
        ? 'This planned sample has not been generated.'
        : 'No media is available for this sample in the selected training row.';

  return (
    <section
      aria-label={`${side} comparison pane`}
      className="min-w-0 overflow-hidden rounded-md border border-gray-700"
    >
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-gray-700 bg-gray-900 px-3 py-2 text-sm">
        <span>
          {side}
          {column ? ` · Sample #${column.index + 1}` : ''}
        </span>
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

export default function SampleComparisonViewer({ open, onClose, rows, sampleConfig, showMetadata, hasEma }: Props) {
  const id = useId();
  const [selection, setSelection] = useState<{ rowKey: string | null; columns: [number, number] }>({
    rowKey: null,
    columns: [-1, -1],
  });
  const [playback, setPlayback] = useState<ComparisonPlaybackState>(EMPTY_COMPARISON_PLAYBACK);
  const leftVideo = useRef<HTMLVideoElement | null>(null);
  const rightVideo = useRef<HTMLVideoElement | null>(null);
  const playbackRef = useRef<ComparisonPlayback | null>(null);
  const [videoMountRevision, setVideoMountRevision] = useState(0);
  const onVideoMount = useCallback(() => setVideoMountRevision(revision => revision + 1), []);
  const rowLabels = useMemo(() => getComparisonRowLabels(rows), [rows]);
  const row = rows.find(candidate => candidate.key === selection.rowKey) ?? rows[rows.length - 1];
  const columns = useMemo(() => (row ? getComparisonColumns(row, sampleConfig) : []), [row, sampleConfig]);
  const selectedColumns = row?.key === selection.rowKey ? selection.columns : getDefaultComparisonColumns(columns);
  const left = columns[selectedColumns[0]];
  const right = columns[selectedColumns[1]];
  const leftPath = left?.path ?? null;
  const rightPath = right?.path ?? null;
  const synchronized = Boolean(leftPath && rightPath && isVideo(leftPath) && isVideo(rightPath));
  const notice = getComparisonNotice(left, right);

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
    if (!row) return;
    playbackRef.current?.pause();
    const next: [number, number] = [...selectedColumns];
    next[side] = index;
    setSelection({ rowKey: row.key, columns: next });
  }

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
                  Choose two samples from the same training run.
                </DialogDescription>
              </div>
              <button type="button" onClick={close} className={BUTTON_CLASS} data-autofocus>
                Close
              </button>
            </div>
            {open && (
              <div className="space-y-4 p-4">
                <div>
                  <label htmlFor={`${id}-row`} className="mb-2 block text-sm font-medium">
                    Training row
                  </label>
                  <select
                    id={`${id}-row`}
                    value={row?.key ?? ''}
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
                <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                  {(['Left', 'Right'] as const).map((side, paneIndex) => (
                    <div key={side} className="min-w-0">
                      <label htmlFor={`${id}-${side}`} className="mb-2 block text-sm font-medium">
                        {side} sample · sample # / weights / prompt
                      </label>
                      <select
                        id={`${id}-${side}`}
                        value={selectedColumns[paneIndex]}
                        onChange={event => changeColumn(paneIndex as 0 | 1, Number(event.target.value))}
                        disabled={!columns.length}
                        className={FIELD_CLASS}
                      >
                        <option value={-1}>Not available</option>
                        {columns.map(column => (
                          <option key={column.index} value={column.index}>
                            {`Sample #${column.index + 1} · ${getComparisonWeightLabel(column, hasEma)} · ${column.prompt ?? 'Prompt unavailable'}${column.path ? '' : ' · Not available'}`}
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
                <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                  <ComparisonPane
                    key={`left:${leftPath ?? `missing-${selectedColumns[0]}`}`}
                    side="Left"
                    column={left}
                    hasEma={hasEma}
                    showMetadata={showMetadata}
                    synchronized={synchronized}
                    synchronizedPlaying={playback.playing}
                    videoRef={leftVideo}
                    onVideoMount={onVideoMount}
                    missing={row?.missing[selectedColumns[0]]}
                  />
                  <ComparisonPane
                    key={`right:${rightPath ?? `missing-${selectedColumns[1]}`}`}
                    side="Right"
                    column={right}
                    hasEma={hasEma}
                    showMetadata={showMetadata}
                    synchronized={synchronized}
                    synchronizedPlaying={playback.playing}
                    videoRef={rightVideo}
                    onVideoMount={onVideoMount}
                    missing={row?.missing[selectedColumns[1]]}
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
