import { scaleLinear } from "d3-scale";
import type { AgentStep } from "./runAgent";
import { useT, type Messages } from "@/i18n";
import { fmtMs } from "@/lib/format";
import { useElementSize } from "@/lib/useElementSize";

const LABEL_W = 132;
const VALUE_W = 72;
const ROW_H = 26;

function stepLabel(step: AgentStep, t: Messages) {
  return step.kind === "model" ? t.agent.modelLane(step.turn) : step.name || "tool";
}

/** A timing diagram of the agent run: one lane per step on a shared time axis. */
export function Timeline({ steps, now }: { steps: AgentStep[]; now: number }) {
  const t = useT();
  const [ref, { width }] = useElementSize<HTMLDivElement>();
  const end = Math.max(1, ...steps.map((step) => step.end ?? now));
  const plotW = Math.max(40, width - LABEL_W - VALUE_W);
  const x = scaleLinear().domain([0, end]).range([0, plotW]).nice(4);
  const ticks = x.ticks(Math.max(2, Math.min(6, Math.floor(plotW / 80))));
  const height = steps.length * ROW_H + 24;

  return (
    <div ref={ref} className="w-full">
      {width > 0 ? (
        <svg width={width} height={height} role="img" aria-label={t.agent.timelineAria} className="block">
          <g transform={`translate(${LABEL_W},0)`}>
            {ticks.map((tick) => (
              <g key={tick} transform={`translate(${x(tick)},0)`}>
                <line y2={steps.length * ROW_H} stroke="var(--rule)" strokeDasharray="1 3" />
                <text y={steps.length * ROW_H + 14} textAnchor="middle" className="fill-ink-3 text-[10.5px] tabular-nums">
                  {fmtMs(tick)}
                </text>
              </g>
            ))}
            <line x1={0} x2={plotW} y1={steps.length * ROW_H} y2={steps.length * ROW_H} stroke="var(--ink)" strokeWidth="1" />
          </g>
          {steps.map((step, index) => {
            const y = index * ROW_H;
            const stop = step.end ?? now;
            const x0 = LABEL_W + x(step.start);
            const w = Math.max(2, x(stop) - x(step.start));
            const first = step.kind === "model" && step.firstToken !== null ? Math.max(0, x(step.firstToken) - x(step.start)) : null;
            const failed = step.kind === "model" ? step.error !== null : step.error !== null;
            return (
              <g key={index}>
                <text x={0} y={y + ROW_H / 2} dy="0.32em" className={step.kind === "model" ? "fill-ink text-[12px]" : "fill-ink-2 text-[12px]"}>
                  {stepLabel(step, t)}
                </text>
                {step.kind === "model" ? (
                  <>
                    <rect x={x0} y={y + 7} width={w} height={ROW_H - 14} fill="var(--wash)" stroke="var(--ink)" strokeWidth="1" opacity={step.end === null ? 0.7 : 1} />
                    {first !== null ? <rect x={x0} y={y + 7} width={Math.min(first, w)} height={ROW_H - 14} fill="var(--ink)" /> : null}
                  </>
                ) : (
                  <rect x={x0} y={y + 9} width={w} height={ROW_H - 18} fill="var(--ink-3)" />
                )}
                {failed ? <line x1={x0} x2={x0 + w} y1={y + ROW_H / 2} y2={y + ROW_H / 2} stroke="var(--bad)" strokeWidth="2" /> : null}
                <text x={width - 2} y={y + ROW_H / 2} dy="0.32em" textAnchor="end" className="fill-ink-2 text-[11px] tabular-nums">
                  {step.end === null ? "…" : fmtMs(stop - step.start)}
                </text>
              </g>
            );
          })}
        </svg>
      ) : null}
    </div>
  );
}
