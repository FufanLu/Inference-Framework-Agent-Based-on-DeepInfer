import { scaleLinear } from "d3-scale";
import type { TimelineStage } from "@/api/types";
import { useT } from "@/i18n";
import { cn } from "@/lib/cn";
import { fmtMs, fmtSpan } from "@/lib/format";

type Kind = "gateway" | "worker" | "decode";

function kindOf(name: string): Kind {
  if (name === "decode") return "decode";
  if (name.startsWith("worker.")) return "worker";
  return "gateway";
}

const BAR: Record<Kind, string> = {
  gateway: "bg-ink-3/55",
  worker: "bg-ink",
  decode: "border border-ink bg-wash",
};

/**
 * The request's stages as a Gantt chart on one millisecond axis from arrival.
 * Gateway stages are microseconds wide and show as ticks; worker prefill and
 * decode take the room.
 */
export function Waterfall({ stages, e2e }: { stages: TimelineStage[]; e2e: number }) {
  const t = useT();
  const k = t.requests.drawer;
  const end = Math.max(e2e, ...stages.map((stage) => stage.start_ms + stage.duration_ms), 0.001);
  const x = scaleLinear().domain([0, end]).nice(5);
  const domainEnd = x.domain()[1] ?? end;
  const ticks = x.ticks(5);
  const pct = (ms: number) => `${(ms / domainEnd) * 100}%`;

  return (
    <div className="flex flex-col gap-2">
      <div role="table" aria-label={k.waterfall} className="grid grid-cols-[7.5rem_minmax(0,1fr)_4.5rem] items-center gap-x-3 gap-y-[3px]">
        {stages.map((stage, index) => {
          const kind = kindOf(stage.name);
          return (
            <div key={`${stage.name}-${index}`} role="row" className="contents">
              <span role="cell" className={cn("truncate font-mono text-xs", kind === "gateway" ? "text-ink-2" : "text-ink")}>
                {stage.name}
              </span>
              <div role="cell" className="relative h-4">
                {ticks.map((tick) => (
                  <span key={tick} aria-hidden className="absolute inset-y-0 w-px bg-rule" style={{ left: pct(tick) }} />
                ))}
                <span
                  className={cn("absolute top-[3px] bottom-[3px] min-w-[2px]", BAR[kind])}
                  style={{ left: pct(stage.start_ms), width: pct(stage.duration_ms) }}
                  title={`${stage.name}: +${fmtSpan(stage.start_ms)} · ${fmtSpan(stage.duration_ms)}`}
                />
              </div>
              <span role="cell" className="text-right font-mono text-xs tabular-nums text-ink-2">
                {fmtSpan(stage.duration_ms)}
              </span>
            </div>
          );
        })}
        <span aria-hidden />
        <div aria-hidden className="relative h-4 border-t border-ink">
          {ticks.map((tick) => (
            <span key={tick} className="absolute top-0.5 -translate-x-1/2 text-[10.5px] whitespace-nowrap text-ink-3 tabular-nums" style={{ left: pct(tick) }}>
              {tick === 0 ? "0" : fmtMs(tick)}
            </span>
          ))}
        </div>
        <span aria-hidden />
      </div>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 pl-[8.25rem] text-2xs text-ink-3">
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2 w-4 bg-ink-3/55" /> {k.gatewayStage}
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2 w-4 bg-ink" /> {k.workerStage}
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2 w-4 border border-ink bg-wash" /> decode
        </span>
      </div>
      <p className="text-2xs text-ink-3">{k.waterfallNote}</p>
    </div>
  );
}
