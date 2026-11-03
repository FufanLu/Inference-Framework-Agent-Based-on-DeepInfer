import { useT } from "@/i18n";
import { fmtMs } from "@/lib/format";

/**
 * One request on a time axis shared by every row of the view: the dark span is
 * time to first token, the light span the rest of the stream.
 */
export function TimingBar({ ttft, e2e, max }: { ttft: number | null; e2e: number; max: number }) {
  const t = useT();
  const scale = max > 0 ? 100 / max : 0;
  const total = Math.max(0.6, e2e * scale);
  const first = ttft === null ? 0 : Math.min(total, ttft * scale);
  return (
    <div
      className="relative h-2 w-full"
      title={`TTFT ${fmtMs(ttft)} · E2E ${fmtMs(e2e)}`}
      role="img"
      aria-label={t.requests.timingBar(fmtMs(ttft), fmtMs(e2e))}
    >
      <div className="absolute inset-y-0 left-0 border border-ink-3/60 bg-wash" style={{ width: `${total}%` }} />
      {first > 0 ? <div className="absolute inset-y-0 left-0 bg-ink" style={{ width: `${first}%` }} /> : null}
    </div>
  );
}
