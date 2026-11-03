import { cn } from "@/lib/cn";

/**
 * State as line form, so it survives without colour: live is a solid rule,
 * stale is dashed, idle dotted, down is struck through.
 */
export type LineState = "live" | "stale" | "idle" | "down";
export type Tone = "ok" | "warn" | "bad" | "ink" | "muted" | "route";

const toneClass: Record<Tone, string> = {
  ok: "text-ok",
  warn: "text-warn",
  bad: "text-bad",
  ink: "text-ink",
  muted: "text-ink-3",
  route: "text-route",
};

export function StatusMark({
  state,
  tone,
  label,
  className,
  width = 18,
}: {
  state: LineState;
  tone: Tone;
  label?: string;
  className?: string;
  width?: number;
}) {
  return (
    <span className={cn("inline-flex items-center gap-1.5", toneClass[tone], className)}>
      <svg width={width} height="10" viewBox={`0 0 ${width} 10`} aria-hidden className="shrink-0">
        <line
          x1="1"
          y1="5"
          x2={width - 1}
          y2="5"
          stroke="currentColor"
          strokeWidth="2"
          strokeDasharray={state === "stale" ? "3 2.5" : state === "idle" ? "1 2.5" : undefined}
          strokeLinecap={state === "idle" ? "round" : "butt"}
        />
        {state === "down" ? (
          <line x1={width / 2 - 3.5} y1="1" x2={width / 2 + 3.5} y2="9" stroke="currentColor" strokeWidth="1.75" />
        ) : null}
      </svg>
      {label ? <span className="text-xs leading-none">{label}</span> : null}
    </span>
  );
}
