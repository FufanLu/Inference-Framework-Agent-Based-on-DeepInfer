import type { WorkerView } from "@/api/types";
import type { LineState, Tone } from "@/components/ui/StatusMark";
import type { Messages } from "@/i18n";

/** Load reports arrive every 250 ms; older than this counts as stale. */
export const STALE_LOAD_MS = 2_000;

export function workerLineState(worker: WorkerView, t: Messages): { state: LineState; tone: Tone; label: string } {
  if (!worker.healthy) return { state: "down", tone: "bad", label: t.worker.unhealthy };
  if (worker.load_age_ms === null) return { state: "idle", tone: "muted", label: t.worker.awaiting };
  if (worker.load_age_ms > STALE_LOAD_MS) {
    return { state: "stale", tone: "warn", label: t.worker.stale((worker.load_age_ms / 1000).toFixed(0)) };
  }
  return { state: "live", tone: "ok", label: t.worker.online };
}
