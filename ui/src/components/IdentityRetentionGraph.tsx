'use client';

// Identity retention graph (Phase 2c, fork-only): median similarity per
// training step, drawn with uPlot like JobLossGraph but always as two fully
// separated series -- ArcFace on its own left scale, SFace on its own right
// scale -- because the two models' cosine scales are incomparable and must
// never share an axis. The shaded band is the 10th-90th percentile of the
// sample distribution at that step; steps with fewer than 5 evaluated samples
// draw as hollow points (their median is noise-level). The dashed horizontal
// guide per scale is the reference self-similarity (LOO) band. Clicking near
// a point filters the Samples grid to that step. No composite score exists
// here: Phase 2a calibration has not shipped, so no combined series is drawn.
import { useEffect, useMemo, useRef, useState } from 'react';
import uPlot from 'uplot';
import 'uplot/dist/uPlot.min.css';
import {
  buildRetentionSeries,
  fmtScore,
  RETENTION_SPARSE_N,
  type CheckpointReport,
  type RetentionModelSeries,
  type RetentionWeightsKind,
  type ScoreModel,
  type ScoreSource,
} from '@/utils/identityCheckpoint';

// Palette slots 0 and 1 from JobLossGraph (blue / green): adjacent slots are
// maximally separated for colorblind viewers on the dark chart surface.
const MODEL_COLORS: Record<ScoreModel, { line: string; label: string }> = {
  arcface: { line: 'rgba(57,135,229,1)', label: 'ArcFace' },
  sface: { line: 'rgba(0,131,0,1)', label: 'SFace' },
};
// Chart surface is always dark (like JobLossGraph), independent of app theme.
const SURFACE = '#030712';
const AXIS_TEXT = 'rgba(255,255,255,0.55)';
const CHART_HEIGHT = 210;
// Max distance (CSS px) from the pointer to a step for a click to count.
const CLICK_SNAP_PX = 16;

function withAlpha(rgba: string, alpha: number): string {
  const m = rgba.match(/rgba?\((\d+),(\d+),(\d+)/);
  if (!m) return rgba;
  return `rgba(${m[1]},${m[2]},${m[3]},${alpha})`;
}

export default function IdentityRetentionGraph({
  report,
  source,
  weightsKind,
  category,
  onFilterStep,
}: {
  report: CheckpointReport;
  source: ScoreSource;
  weightsKind: RetentionWeightsKind;
  category: string | null;
  onFilterStep: (step: number) => void;
}) {
  const models = useMemo(
    () => buildRetentionSeries(report, source, weightsKind, category),
    [report, source, weightsKind, category],
  );
  const withData = models.filter(s => s.xs.length > 0);
  const hasData = withData.length > 0;

  // Series layout per model: p10, median, p90, filled points, hollow points,
  // LOO guide. Band edges (p10/p90) carry transparent strokes -- they exist
  // to bound the shaded band, not to be read as lines.
  const built = useMemo(() => {
    const data: (number | null)[][] = [[]]; // x filled below
    const seriesConfigs: uPlot.Series[] = [{}];
    const scales: uPlot.Scales = { x: { time: false } };
    const axes: uPlot.Axis[] = [
      {
        stroke: AXIS_TEXT,
        grid: { stroke: 'rgba(255,255,255,0.06)' },
        ticks: { stroke: 'rgba(255,255,255,0.15)' },
        values: (_u, ticks) => ticks.map(t => Math.round(t).toLocaleString()),
      },
    ];
    const bands: uPlot.Band[] = [];

    const xs = withData[0]?.xs ?? [];
    data[0] = xs;

    withData.forEach((perModel: RetentionModelSeries, mi) => {
      const color = MODEL_COLORS[perModel.model].line;
      const scaleKey = `ret:${perModel.model}`;
      const add = (values: (number | null)[], config: uPlot.Series): number => {
        data.push(values);
        seriesConfigs.push({ scale: scaleKey, ...config });
        return data.length - 1;
      };

      const p10Idx = add(perModel.p10, {
        label: '',
        stroke: 'rgba(0,0,0,0)',
        width: 1,
        spanGaps: true,
        points: { show: false },
      });
      add(perModel.median, {
        label: MODEL_COLORS[perModel.model].label,
        stroke: color,
        width: 2,
        spanGaps: true,
        points: { show: false },
      });
      const p90Idx = add(perModel.p90, {
        label: '',
        stroke: 'rgba(0,0,0,0)',
        width: 1,
        spanGaps: true,
        points: { show: false },
      });
      // scatter-only overlay: paths suppressed so no line joins these points
      add(
        perModel.median.map((v, i) => (!perModel.sparse[i] ? v : null)),
        {
          label: '',
          stroke: 'rgba(0,0,0,0)',
          paths: () => null,
          points: { show: true, size: 6, width: 1.5, fill: color, stroke: SURFACE },
        },
      );
      add(
        perModel.median.map((v, i) => (perModel.sparse[i] ? v : null)),
        {
          label: '',
          stroke: 'rgba(0,0,0,0)',
          paths: () => null,
          points: { show: true, size: 9, width: 2, fill: 'rgba(0,0,0,0)', stroke: color },
        },
      );
      add(
        perModel.loo == null ? xs.map(() => null) : xs.map(() => perModel.loo),
        {
          label: '',
          stroke: withAlpha(color, 0.55),
          width: 1.5,
          dash: [5, 4],
          spanGaps: true,
          points: { show: false },
        },
      );

      bands.push({ series: [p90Idx, p10Idx], fill: withAlpha(color, 0.14), dir: -1 });

      scales[scaleKey] = {};
      axes.push({
        scale: scaleKey,
        side: mi === 0 ? 3 : 1,
        stroke: color,
        label: MODEL_COLORS[perModel.model].label,
        labelSize: 13,
        size: 36,
        grid: { show: mi === 0, stroke: 'rgba(255,255,255,0.06)' },
        ticks: { stroke: 'rgba(255,255,255,0.15)' },
        values: (_u, ticks) => ticks.map(t => t.toFixed(2)),
      });
    });

    return { data: data as uPlot.AlignedData, seriesConfigs, scales, axes, bands, xs };
  }, [withData]);

  const hostRef = useRef<HTMLDivElement>(null);
  const mountRef = useRef<HTMLDivElement>(null);
  const uplotRef = useRef<uPlot | null>(null);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  // Structural recreate only when the series shape changes (models present /
  // filter selection); data-only changes go through setData.
  const structuralKey = `${withData.map(s => s.model).join('|')}|${source}|${weightsKind}|${category}|${hasData}`;

  useEffect(() => {
    if (uplotRef.current) {
      uplotRef.current.destroy();
      uplotRef.current = null;
    }
    if (!mountRef.current || !hostRef.current || !hasData) return;
    const width = hostRef.current.getBoundingClientRect().width || 380;
    const opts: uPlot.Options = {
      width,
      height: CHART_HEIGHT,
      padding: [8, 8, 0, 2],
      series: built.seriesConfigs,
      scales: built.scales,
      axes: built.axes,
      bands: built.bands,
      legend: { show: false },
      cursor: {
        drag: { x: false, y: false, setScale: false },
        points: { size: 6, width: 1 },
      },
      hooks: {
        setCursor: [
          u => {
            setHoverIdx(u.cursor.idx ?? null);
          },
        ],
      },
    };
    uplotRef.current = new uPlot(opts, built.data, mountRef.current);
    return () => {
      uplotRef.current?.destroy();
      uplotRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [structuralKey]);

  // Push new data without recreating the chart.
  useEffect(() => {
    uplotRef.current?.setData(built.data);
  }, [built]);

  // Keep the chart width in sync with the panel.
  useEffect(() => {
    const el = hostRef.current;
    if (!el || !hasData) return;
    const ro = new ResizeObserver(() => {
      const rect = el.getBoundingClientRect();
      if (rect.width > 0) uplotRef.current?.setSize({ width: rect.width, height: CHART_HEIGHT });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [hasData]);

  // Click near a step -> filter the Samples grid. All model series share xs
  // (built from the same filtered step list), so the first one anchors lookup.
  const onFilterStepRef = useRef(onFilterStep);
  useEffect(() => {
    onFilterStepRef.current = onFilterStep;
  }, [onFilterStep]);
  useEffect(() => {
    const u = uplotRef.current;
    if (!u || !hasData) return;
    const xs = built.xs;
    const handler = (e: MouseEvent) => {
      const rect = u.over.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const bbox = u.bbox;
      if (px < bbox.left || px > bbox.left + bbox.width) return;
      const xval = u.posToVal(px, 'x');
      let best = -1;
      let bestDist = Infinity;
      xs.forEach((x, i) => {
        const d = Math.abs(x - xval);
        if (d < bestDist) {
          bestDist = d;
          best = i;
        }
      });
      if (best < 0) return;
      const pointPx = u.valToPos(xs[best], 'x');
      if (Math.abs(pointPx - px) > CLICK_SNAP_PX) return;
      onFilterStepRef.current(xs[best]);
    };
    u.over.addEventListener('click', handler);
    return () => u.over.removeEventListener('click', handler);
  }, [built, hasData, structuralKey]);

  const readout = useMemo(() => {
    if (hoverIdx == null) return null;
    const bits = withData
      .map(s => {
        const median = s.median[hoverIdx];
        if (median == null) return null;
        const band =
          s.p10[hoverIdx] != null && s.p90[hoverIdx] != null
            ? `band ${fmtScore(s.p10[hoverIdx])}–${fmtScore(s.p90[hoverIdx])}`
            : null;
        const loo = s.loo != null ? `LOO ${fmtScore(s.loo)}` : null;
        return [
          `${MODEL_COLORS[s.model].label} ${fmtScore(median)}`,
          `n ${s.n[hoverIdx]}`,
          band,
          loo,
        ]
          .filter(Boolean)
          .join(' · ');
      })
      .filter(Boolean);
    if (bits.length === 0) return null;
    return `step ${(built.xs[hoverIdx] ?? 0).toLocaleString()} — ${bits.join('  |  ')}`;
  }, [hoverIdx, withData, built.xs]);

  return (
    <div className="space-y-1">
      <div ref={hostRef} className="relative rounded border border-gray-300 bg-gray-950 dark:border-gray-600" style={{ minHeight: CHART_HEIGHT }}>
        {!hasData ? (
          <div className="absolute inset-0 flex items-center justify-center px-4 text-center text-[10px] text-gray-400">
            no scored samples for this selection
          </div>
        ) : (
          <div ref={mountRef} />
        )}
      </div>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[10px] text-gray-500 dark:text-gray-400">
        {withData.map(s => (
          <span key={s.model} className="inline-flex items-center gap-1">
            <span className="inline-block h-2 w-2 rounded-full" style={{ background: MODEL_COLORS[s.model].line }} />
            {MODEL_COLORS[s.model].label}
            {s.loo != null ? (
              <span title="reference self-similarity (LOO) — how similar the reference photos are to each other">
                LOO {fmtScore(s.loo)}
              </span>
            ) : null}
          </span>
        ))}
        <span title={`steps with fewer than ${RETENTION_SPARSE_N} evaluated samples — the median is noise-level there`}>
          ○ &lt; {RETENTION_SPARSE_N} samples
        </span>
        <span title="10th–90th percentile of sample scores at that step">▮ p10–p90 band</span>
        <span>click a point to filter the grid</span>
      </div>
      {readout && <div className="tabular-nums text-[10px] text-gray-600 dark:text-gray-300">{readout}</div>}
    </div>
  );
}
