import { scaleBand, scaleLinear } from "d3-scale";
import { useId, useMemo } from "react";
import { useT } from "@/i18n";
import { fmtInt } from "@/lib/format";
import { useElementSize } from "@/lib/useElementSize";
import type { DayTotals } from "./usage";

const MARGIN = { top: 20, right: 8, bottom: 22, left: 52 };

function compact(value: number): string {
  if (value >= 1e9) return `${(value / 1e9).toFixed(1)}G`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(value >= 1e4 ? 0 : 1)}k`;
  return String(Math.round(value));
}

/**
 * Daily tokens stacked as uncached prompt, cached prompt and completion: the
 * three add up to billable tokens (cached prompt tokens are part of prompt).
 */
export function UsageChart({ data, height = 220 }: { data: DayTotals[]; height?: number }) {
  const t = useT();
  const k = t.tenants.usage;
  const [ref, { width }] = useElementSize<HTMLDivElement>();
  const hatch = useId().replace(/:/g, "");

  const geometry = useMemo(() => {
    if (width < 80 || data.length === 0) return null;
    const innerW = width - MARGIN.left - MARGIN.right;
    const innerH = height - MARGIN.top - MARGIN.bottom;
    const x = scaleBand<string>()
      .domain(data.map((d) => d.day))
      .range([0, innerW])
      .paddingInner(0.25)
      .paddingOuter(0.1);
    const max = Math.max(1, ...data.map((d) => d.uncached + d.cached + d.completion));
    const y = scaleLinear().domain([0, max]).range([innerH, 0]).nice(4);
    const every = Math.max(1, Math.ceil(data.length / Math.max(2, Math.floor(innerW / 64))));
    return { innerW, innerH, x, y, ticks: y.ticks(4), every };
  }, [width, height, data]);

  return (
    <div ref={ref} className="w-full" style={{ height }}>
      {geometry ? (
        <svg width={width} height={height} role="img" aria-label={k.aria} className="block">
          <defs>
            <pattern id={hatch} width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <rect width="4" height="4" fill="var(--sheet)" />
              <line x1="0" y1="0" x2="0" y2="4" stroke="var(--ink-2)" strokeWidth="1.5" />
            </pattern>
          </defs>
          <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
            {geometry.ticks.map((tick) => (
              <g key={tick} transform={`translate(0,${geometry.y(tick)})`}>
                <line x2={geometry.innerW} stroke="var(--rule)" strokeWidth="0.75" strokeDasharray={tick === 0 ? undefined : "1 3"} />
                <text x={-6} dy="0.32em" textAnchor="end" className="fill-ink-3 text-[10.5px] tabular-nums">
                  {compact(tick)}
                </text>
              </g>
            ))}
            <text x={-MARGIN.left + 2} y={-12} className="fill-ink-3 text-[10px]" style={{ fontStretch: "80%" }}>
              {k.unit}
            </text>
            {data.map((d, index) => {
              const x0 = geometry.x(d.day) ?? 0;
              const w = geometry.x.bandwidth();
              const segments = [
                { key: "uncached", value: d.uncached, fill: "var(--ink-3)" },
                { key: "cached", value: d.cached, fill: `url(#${hatch})` },
                { key: "completion", value: d.completion, fill: "var(--ink)" },
              ];
              let base = 0;
              return (
                <g key={d.day}>
                  <title>
                    {`${d.day} · ${k.requests(fmtInt(d.requests))}\n${k.uncached}: ${fmtInt(d.uncached)}\n${k.cached}: ${fmtInt(d.cached)}\n${k.completion}: ${fmtInt(d.completion)}`}
                  </title>
                  {segments.map((segment) => {
                    const yTop = geometry.y(base + segment.value);
                    const yBottom = geometry.y(base);
                    base += segment.value;
                    return segment.value > 0 ? (
                      <rect key={segment.key} x={x0} y={yTop} width={w} height={Math.max(0.5, yBottom - yTop)} fill={segment.fill} stroke={segment.key === "cached" ? "var(--ink-2)" : undefined} strokeWidth={segment.key === "cached" ? 0.75 : undefined} />
                    ) : null;
                  })}
                  {(data.length - 1 - index) % geometry.every === 0 ? (
                    <text x={x0 + w / 2} y={geometry.innerH + 14} textAnchor="middle" className="fill-ink-3 text-[10.5px] tabular-nums">
                      {d.day.slice(5)}
                    </text>
                  ) : null}
                </g>
              );
            })}
            <line x1={0} x2={geometry.innerW} y1={geometry.innerH} y2={geometry.innerH} stroke="var(--ink)" strokeWidth="1" />
          </g>
        </svg>
      ) : null}
    </div>
  );
}

export function UsageLegend() {
  const t = useT();
  const k = t.tenants.usage;
  const hatch = useId().replace(/:/g, "");
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-2xs text-ink-2">
      <span className="inline-flex items-center gap-1.5">
        <span className="inline-block h-2.5 w-4 bg-ink-3" /> {k.uncached}
      </span>
      <span className="inline-flex items-center gap-1.5">
        <svg width="16" height="10" aria-hidden>
          <defs>
            <pattern id={hatch} width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <rect width="4" height="4" fill="var(--sheet)" />
              <line x1="0" y1="0" x2="0" y2="4" stroke="var(--ink-2)" strokeWidth="1.5" />
            </pattern>
          </defs>
          <rect width="16" height="10" fill={`url(#${hatch})`} stroke="var(--ink-2)" strokeWidth="0.75" />
        </svg>
        {k.cached}
      </span>
      <span className="inline-flex items-center gap-1.5">
        <span className="inline-block h-2.5 w-4 bg-ink" /> {k.completion}
      </span>
    </div>
  );
}
