import { useState } from "react";
import type { SseFrame } from "@/api/client";
import { EmptyState } from "@/components/States";
import { useT } from "@/i18n";
import { cn } from "@/lib/cn";
import { fmtInt } from "@/lib/format";

/** Every SSE frame as it arrived: arrival time, event name, data. */
export function FrameLog({ frames, truncated }: { frames: SseFrame[]; truncated: number }) {
  const t = useT();
  const k = t.playground.frames;
  const [open, setOpen] = useState<number | null>(null);
  if (frames.length === 0) {
    return (
      <EmptyState title={k.emptyTitle}>
        <p>{k.emptyBody}</p>
      </EmptyState>
    );
  }
  return (
    <div className="flex flex-col gap-2">
      <p className="text-xs text-ink-3">
        {k.count(fmtInt(frames.length + truncated))}
        {truncated > 0 ? k.kept(fmtInt(frames.length)) : ""}. {k.expand}
      </p>
      <div className="max-h-[70vh] overflow-auto border-y-[1.5px] border-ink">
        <table className="table-3l border-0 font-mono">
          <thead className="sticky top-0 bg-paper">
            <tr>
              <th className="num w-12">#</th>
              <th className="num w-20">
                t <span className="unit">(ms)</span>
              </th>
              <th className="w-48">event</th>
              <th>data</th>
            </tr>
          </thead>
          <tbody>
            {frames.map((frame, index) => {
              const n = index + truncated;
              const expanded = open === n;
              return (
                <tr key={n} className="row-action align-top" onClick={() => setOpen(expanded ? null : n)} aria-expanded={expanded}>
                  <td className="num font-sans text-2xs text-ink-3">{n + 1}</td>
                  <td className="num font-sans text-2xs text-ink-2">{frame.t.toFixed(1)}</td>
                  <td className="truncate text-2xs text-ink-2">{frame.event ?? "—"}</td>
                  <td className={cn("text-2xs", expanded ? "break-all whitespace-pre-wrap" : "max-w-0 truncate")}>{frame.data}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
