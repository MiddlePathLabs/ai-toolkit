// State + actions for the identity-eval feature on the Samples screen.
// Loads existing results once on mount, polls /status while a worker runs,
// and exposes the reference/selection/run/cancel/clear actions the panel
// and toolbar need. Sample overlays read `index` regardless of panel state.
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { Job } from '@prisma/client';
import { apiClient } from '@/utils/api';
import {
  buildIdentityIndex,
  rankDisagreements,
  type IdentityRefs,
  type IdentityResults,
  type IdentitySelection,
  type IdentitySource,
} from '@/utils/identityEval';
import { sanitizeCategories, type CheckpointReport } from '@/utils/identityCheckpoint';

export interface IdentityProgress {
  phase: string;
  done: number;
  total: number;
  running?: boolean;
  error?: string;
}

interface StatusResponse {
  running: boolean;
  state: { pid: number; started_at?: string; sources?: string[]; device_mode?: string; sample_count?: number } | null;
  progress: IdentityProgress | null;
  results: IdentityResults | null;
  models: Record<string, unknown> | null;
  selection: IdentitySelection | null;
  refs?: IdentityRefs | null;
}

export interface IdentityEvalApi {
  refs: IdentityRefs | null;
  selection: IdentitySelection | null;
  results: IdentityResults | null;
  index: Map<string, IdentityResults['samples'][number]>;
  disagreements: Map<string, boolean>;
  progress: IdentityProgress | null;
  running: boolean;
  loading: boolean;
  error: string | null;
  source: IdentitySource | 'both';
  setSource: (source: IdentitySource | 'both') => void;
  scoreSource: IdentitySource | 'combined';
  loadRefs: (refresh?: boolean) => Promise<void>;
  saveSelection: (update: { trainingRefs?: IdentitySelection['training_refs']; externalRefs?: IdentitySelection['external_refs'] }) => Promise<void>;
  addExternalFiles: (files: File[]) => Promise<void>;
  removeExternal: (names: string[]) => Promise<void>;
  startRun: (options: { samples: string[]; deviceMode: 'auto' | 'gpu' | 'cpu' }) => Promise<void>;
  cancel: () => Promise<void>;
  clear: (includeCache: boolean) => Promise<void>;
  report: CheckpointReport | null;
  reportLoading: boolean;
  loadReport: () => Promise<void>;
  saveCategories: (categories: Record<string, string>) => Promise<boolean>;
}

export default function useIdentityEval(job: Job): IdentityEvalApi {
  const [refs, setRefs] = useState<IdentityRefs | null>(null);
  const [selection, setSelection] = useState<IdentitySelection | null>(null);
  const [results, setResults] = useState<IdentityResults | null>(null);
  const [progress, setProgress] = useState<IdentityProgress | null>(null);
  const [running, setRunning] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [source, setSource] = useState<IdentitySource | 'both'>('training');
  const [report, setReport] = useState<CheckpointReport | null>(null);
  const [reportLoading, setReportLoading] = useState(false);
  const jobId = job.id;
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const wasRunning = useRef(false);

  const applyStatus = useCallback((data: StatusResponse) => {
    setRunning(data.running);
    setProgress(data.progress ?? null);
    if (data.results) setResults(data.results);
    if (data.selection) setSelection(data.selection);
    if (data.refs) setRefs(data.refs as IdentityRefs);
  }, []);

  const fetchStatus = useCallback(async () => {
    try {
      const res = await apiClient.get<StatusResponse>(`/api/identity-eval/status`, { params: { jobId } });
      applyStatus(res.data);
    } catch (err) {
      console.error('identity-eval status failed:', err);
    }
  }, [applyStatus, jobId]);

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  // poll while a worker is live (started here or elsewhere)
  useEffect(() => {
    if (!running) {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
      return;
    }
    pollRef.current = setInterval(fetchStatus, 2000);
    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [running, fetchStatus]);

  const loadRefs = useCallback(
    async (refresh = false) => {
      setLoading(true);
      setError(null);
      try {
        const res = await apiClient.get<{ refs: IdentityRefs; selection: IdentitySelection; error?: string }>(`/api/identity-eval/refs`, {
          params: { jobId, refresh: refresh ? 1 : undefined },
        });
        if (res.data.refs) setRefs(res.data.refs);
        if (res.data.selection) setSelection(res.data.selection);
      } catch (err) {
        console.error('identity-eval refs failed:', err);
        setError('Failed to load references');
      } finally {
        setLoading(false);
      }
    },
    [jobId],
  );

  const saveSelection = useCallback(
    async (update: { trainingRefs?: IdentitySelection['training_refs']; externalRefs?: IdentitySelection['external_refs'] }) => {
      setSelection(prev => {
        if (!prev) return prev;
        const merged = {
          version: 1,
          training_refs: { ...prev.training_refs, ...update.trainingRefs },
          external_refs: { ...prev.external_refs, ...update.externalRefs },
        } as IdentitySelection;
        return merged;
      });
      try {
        await apiClient.post(`/api/identity-eval/refs`, { jobId, trainingRefs: update.trainingRefs, externalRefs: update.externalRefs });
      } catch (err) {
        console.error('identity-eval selection save failed:', err);
        setError('Failed to save selection');
        fetchStatus();
      }
    },
    [jobId, fetchStatus],
  );

  const addExternalFiles = useCallback(
    async (files: File[]) => {
      if (files.length === 0) return;
      setLoading(true);
      setError(null);
      try {
        const form = new FormData();
        form.append('jobId', jobId);
        for (const file of files) form.append('files', file);
        const res = await apiClient.post<{ refs: IdentityRefs; selection: IdentitySelection; error?: string }>(`/api/identity-eval/refs`, form);
        if (res.data.refs) setRefs(res.data.refs);
        if (res.data.selection) setSelection(res.data.selection);
      } catch (err) {
        console.error('identity-eval external add failed:', err);
        setError('Failed to add external references');
      } finally {
        setLoading(false);
      }
    },
    [jobId],
  );

  const removeExternal = useCallback(
    async (names: string[]) => {
      setLoading(true);
      try {
        const res = await apiClient.post<{ refs: IdentityRefs; selection: IdentitySelection }>(`/api/identity-eval/refs`, {
          jobId,
          removeExternal: names,
        });
        if (res.data.refs) setRefs(res.data.refs);
        if (res.data.selection) setSelection(res.data.selection);
      } catch (err) {
        console.error('identity-eval external remove failed:', err);
        setError('Failed to remove external references');
      } finally {
        setLoading(false);
      }
    },
    [jobId],
  );

  const startRun = useCallback(
    async ({ samples, deviceMode }: { samples: string[]; deviceMode: 'auto' | 'gpu' | 'cpu' }) => {
      setError(null);
      const sources = source === 'both' ? ['training', 'external'] : [source];
      try {
        const res = await apiClient.post<{ started?: boolean; alreadyRunning?: boolean; error?: string; state?: unknown }>(
          `/api/identity-eval/run`,
          { jobId, sources, deviceMode, samples },
        );
        if (res.data.error) {
          setError(res.data.error);
          return;
        }
        setRunning(true);
        fetchStatus();
      } catch (err) {
        console.error('identity-eval run failed:', err);
        setError('Failed to start evaluation');
      }
    },
    [jobId, source, fetchStatus],
  );

  const cancel = useCallback(async () => {
    try {
      await apiClient.post(`/api/identity-eval/cancel`, { jobId });
      setRunning(false);
      fetchStatus();
    } catch (err) {
      console.error('identity-eval cancel failed:', err);
      setError('Failed to cancel');
    }
  }, [jobId, fetchStatus]);

  const clear = useCallback(
    async (includeCache: boolean) => {
      try {
        await apiClient.post(`/api/identity-eval/clear`, { jobId, includeCache });
        setResults(null);
        setProgress(null);
        setReport(null);
      } catch (err) {
        console.error('identity-eval clear failed:', err);
        setError('Failed to clear results');
      }
    },
    [jobId],
  );

  // Checkpoint comparison report (Phase 2b): served from cache by the route,
  // recomputed by the compare worker when results or category tags changed.
  const loadReport = useCallback(async () => {
    setReportLoading(true);
    try {
      const res = await apiClient.get<{ report: CheckpointReport | null; error?: string }>(`/api/identity-eval/compare`, {
        params: { jobId },
      });
      if (res.data.error) setError(res.data.error);
      setReport(res.data.report ?? null);
    } catch (err) {
      console.error('identity-eval compare failed:', err);
      setError('Failed to load checkpoint comparison');
    } finally {
      setReportLoading(false);
    }
  }, [jobId]);

  // Returns true when the tags were persisted and the report recomputed;
  // the caller keeps its draft on false so edits survive a failed save.
  const saveCategories = useCallback(
    async (categories: Record<string, string>): Promise<boolean> => {
      setReportLoading(true);
      let ok = false;
      try {
        const res = await apiClient.post<{ report: CheckpointReport | null; categories: Record<string, string>; error?: string }>(
          `/api/identity-eval/compare`,
          { jobId, categories: sanitizeCategories(categories) },
        );
        if (res.data.error) setError(res.data.error);
        if (res.data.report) setReport(res.data.report);
        ok = !res.data.error;
      } catch (err) {
        console.error('identity-eval category save failed:', err);
        setError('Failed to save prompt categories');
      } finally {
        setReportLoading(false);
      }
      return ok;
    },
    [jobId],
  );

  // refresh the report when a run finishes (results.json just changed)
  useEffect(() => {
    if (wasRunning.current && !running) {
      loadReport();
    }
    wasRunning.current = running;
  }, [running, loadReport]);

  useEffect(() => {
    loadReport();
  }, [loadReport]);

  const index = useMemo(() => buildIdentityIndex(results), [results]);
  const disagreements = useMemo(
    () => rankDisagreements(results?.samples ?? [], source === 'both' ? 'combined' : source),
    [results, source],
  );
  const scoreSource: IdentitySource | 'combined' = source === 'both' ? 'combined' : source;

  return {
    refs,
    selection,
    results,
    index,
    disagreements,
    progress,
    running,
    loading,
    error,
    source,
    setSource,
    scoreSource,
    loadRefs,
    saveSelection,
    addExternalFiles,
    removeExternal,
    startRun,
    cancel,
    clear,
    report,
    reportLoading,
    loadReport,
    saveCategories,
  };
}
