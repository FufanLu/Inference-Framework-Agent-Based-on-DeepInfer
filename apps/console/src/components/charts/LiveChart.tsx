import { scaleLinear, scaleTime } from "d3-scale";
import { line } from "d3-shape";
import { useId, useMemo } from "react";
import type { Bucket } from "@/lib/stats";
import { useElementSize } from "@/lib/useElementSize";
import { fmtClock } from "@/lib/format";
import { useT } from "@/i18n";

export interface ChartSeries {
  label: string;
  data: Bucket[];
  /** A second, lighter curve (e.g. p99 next to p50). Dashes stay reserved for stale state. */
  secondary?: boolean;
}

interface LiveChartProps {
  series: ChartSeries[];
  bucketMs: number;
  /** Tick label for the value axis. */
  format: (value: number) => string;
  /** Axis title, e.g. "ms" or "req/s". */
  unit: string;
  height?: number;
  /** Lowest upper bound of the value axis, so tiny noise is not magnified. */
  minMax?: number;
  /** Fixed upper bound, e.g. 1 for ratios. */
  fixedMax?: number;
  /** The window the headline figure covers, marked as a band on the time axis. */
  windowMs?: number;
  windowLabel?: string;
  ariaLabel: string;
}

const MARGIN = { top: 8, right: 10, bottom: 22, left: 44 };

/**
 * A characteristic curve, datasheet style: framed plot, dotted major grid,
 * clock-time axis, gaps where a bucket has no value, a hatched and labelled
 * region where the console was not observing, and a band on the time axis for
 * the window the figure above the curve covers.
 */
export function LiveChart({ series, bucketMs, format, unit, height = 112, minMax = 0, fixedMax, windowMs, windowLabel, ariaLabel }: LiveChartProps) {
  const t = useT();
  const [ref, { width }] = useElementSize<HTMLDivElement>();
  const patternId = useId().replace(/:/g, "");
  const first = useMemo(() => series[0]?.data ?? [], [series]);

  const geometry = useMemo(() => {
    if (width < 40 || first.length === 0) return null;
    const innerW = width - MARGIN.left - MARGIN.right;
    const innerH = height - MARGIN.top - MARGIN.bottom;
    const start = (first[0]?.t ?? 0) - bucketMs;
    const end = first[first.length - 1]?.t ?? 0;
    const x = scaleTime().domain([start, end]).range([0, innerW]);
    const values = series.flatMap((s) => s.data.flatMap((b) => (b.value === null ? [] : [b.value])));
    const max = fixedMax ?? Math.max(minMax, ...values, 0);
    const y = scaleLinear()
      .domain([0, max > 0 ? max : 1])
      .range([innerH, 0])
      .nice(3);
    const yTicks = y.ticks(3);
    const xTicks = x.ticks(Math.max(2, Math.min(6, Math.floor(innerW / 90))));
    const path = line<Bucket>()
      .defined((b) => b.value !== null)
      .x((b) => x(b.t - bucketMs / 2))
      .y((b) => y(b.value ?? 0));
    const uncovered = first.findIndex((b) => b.covered);
    const hatchEnd = uncovered === -1 ? end : (first[uncovered]?.t ?? end) - bucketMs;
    const windowStart = windowMs ? Math.max(start, end - windowMs) : null;
    return { innerW, innerH, x, y, yTicks, xTicks, path, hatchEnd, start, end, windowStart };
  }, [width, height, first, series, bucketMs, minMax, fixedMax, windowMs]);

  const hasValue = series.some((s) => s.data.some((b) => b.value !== null));
  const observed = first.some((b) => b.covered);

  return (
    <div ref={ref} className="relative w-full" style={{ height }}>
      {geometry ? (
        <svg width={width} height={height} role="img" aria-label={ariaLabel} className="block overflow-visible">
          <defs>
            <pattern id={patternId} width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <line x1="0" y1="0" x2="0" y2="6" stroke="var(--rule)" strokeWidth="1.5" />
            </pattern>
          </defs>
          <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
            {geometry.windowStart !== null ? (
              <rect
                x={geometry.x(geometry.windowStart)}
                y={0}
                width={Math.max(0, geometry.x(geometry.end) - geometry.x(geometry.windowStart))}
                height={geometry.innerH}
                fill="var(--ink)"
                opacity="0.045"
              />
            ) : null}
            {geometry.hatchEnd > geometry.start ? (
              <g>
                <rect x={0} y={0} width={Math.max(0, geometry.x(geometry.hatchEnd))} height={geometry.innerH} fill={`url(#${patternId})`}>
                  <title>{t.chart.unobserved}</title>
                </rect>
                {geometry.x(geometry.hatchEnd) > 72 ? (
                  <text
                    x={geometry.x(geometry.hatchEnd) / 2}
                    y={geometry.innerH / 2}
                    dy="0.32em"
                    textAnchor="middle"
                    className="fill-ink-3 text-[10.5px]"
                    style={{ paintOrder: "stroke", stroke: "var(--paper)", strokeWidth: 4 }}
                  >
                    {t.chart.unobservedShort}
                  </text>
                ) : null}
              </g>
            ) : null}
            {geometry.yTicks.map((tick) => (
              <g key={`y${tick}`} transform={`translate(0,${geometry.y(tick)})`}>
                <line x2={geometry.innerW} stroke="var(--rule)" strokeWidth="0.75" strokeDasharray={tick === 0 ? undefined : "1 3"} />
                <text x={-6} dy="0.32em" textAnchor="end" className="fill-ink-3 text-[10.5px] tabular-nums">
                  {format(tick)}
                </text>
              </g>
            ))}
            {geometry.xTicks.map((tick) => (
              <g key={`x${tick.getTime()}`} transform={`translate(${geometry.x(tick)},0)`}>
                <line y2={geometry.innerH} stroke="var(--rule)" strokeWidth="0.75" strokeDasharray="1 3" />
                <text y={geometry.innerH + 14} textAnchor="middle" className="fill-ink-3 text-[10.5px] tabular-nums">
                  {fmtClock(tick.getTime()).slice(0, bucketMs >= 60_000 ? 5 : 8)}
                </text>
              </g>
            ))}
            <rect width={geometry.innerW} height={geometry.innerH} fill="none" stroke="var(--rule-2)" strokeWidth="0.75" />
            {geometry.windowStart !== null ? (
              <g>
                <rect
                  x={geometry.x(geometry.windowStart)}
                  y={geometry.innerH - 1}
                  width={Math.max(2, geometry.x(geometry.end) - geometry.x(geometry.windowStart))}
                  height={3}
                  fill="var(--ink)"
                />
                {windowLabel ? (
                  <text
                    x={geometry.x(geometry.windowStart) - 4}
                    y={geometry.innerH - 4}
                    textAnchor="end"
                    className="fill-ink-2 text-[10px]"
                    style={{ fontStretch: "80%", paintOrder: "stroke", stroke: "var(--paper)", strokeWidth: 3 }}
                  >
                    {t.chart.window(windowLabel)}
                  </text>
                ) : null}
              </g>
            ) : null}
            <text x={-MARGIN.left + 2} y={-1} className="fill-ink-3 text-[10px]" style={{ fontStretch: "80%" }}>
              {unit}
            </text>
            {series.map((s) => {
              const d = geometry.path(s.data);
              const last = [...s.data].reverse().find((b) => b.value !== null);
              const stroke = s.secondary ? "var(--ink-3)" : "var(--ink)";
              return (
                <g key={s.label}>
                  {d ? <path d={d} fill="none" stroke={stroke} strokeWidth={s.secondary ? 1.25 : 1.6} strokeLinejoin="round" /> : null}
                  {last && last.value !== null ? (
                    <rect
                      x={geometry.x(last.t - bucketMs / 2) - 2.5}
                      y={geometry.y(last.value) - 2.5}
                      width="5"
                      height="5"
                      fill={s.secondary ? "var(--sheet)" : "var(--ink)"}
                      stroke={stroke}
                      strokeWidth="1"
                    />
                  ) : null}
                </g>
              );
            })}
          </g>
        </svg>
      ) : null}
      {geometry && !hasValue ? (
        <p className="pointer-events-none absolute inset-0 flex items-center justify-center pl-11 text-2xs text-ink-3">
          {observed ? t.chart.noSamples : t.chart.notObserved}
        </p>
      ) : null}
    </div>
  );
}
