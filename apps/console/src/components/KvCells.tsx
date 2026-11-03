import { useT } from "@/i18n";
import { cn } from "@/lib/cn";

/** The router penalises workers above this KV usage (`RouterConfig::max_kv_usage`). */
export const KV_PENALTY_AT = 0.95;

function clamp(value: number) {
  return Math.min(1, Math.max(0, value));
}

/**
 * KV usage drawn as discrete block cells rather than a smooth fill: the cache
 * is managed in blocks, and so is this meter. Dark cells are blocks pinned by
 * running sequences (the worker's load report); light cells, when `cached` is
 * given, are prefix blocks the router's index holds for the worker.
 */
export function KvCells({
  usage,
  cached = null,
  cells = 24,
  className,
  size = "md",
}: {
  usage: number | null;
  cached?: number | null;
  cells?: number;
  className?: string;
  size?: "sm" | "md";
}) {
  const t = useT();
  const ratio = usage === null ? 0 : clamp(usage);
  const filled = usage === null ? 0 : Math.round(ratio * cells);
  const cachedCells = cached === null ? 0 : Math.max(filled, Math.round(clamp(cached) * cells));
  const hot = ratio > KV_PENALTY_AT;
  const label =
    usage === null
      ? t.kv.unknown
      : `${t.kv.usage(`${(ratio * 100).toFixed(1)}%`)}${cached === null ? "" : ` · ${t.kv.cached(`${(clamp(cached) * 100).toFixed(1)}%`)}`}`;
  return (
    <div
      role="meter"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={usage === null ? undefined : Math.round(ratio * 100)}
      title={label}
      className={cn("flex gap-[2px]", className)}
    >
      {Array.from({ length: cells }, (_, index) => (
        <span
          key={index}
          className={cn(
            "block flex-1",
            size === "sm" ? "h-2 min-w-[3px]" : "h-2.5 min-w-[4px]",
            index < filled
              ? hot
                ? "bg-bad"
                : "bg-ink"
              : index < cachedCells
                ? "border border-ink-3 bg-ink-3/60"
                : "border border-rule-2 bg-transparent",
            usage === null && "border-dotted",
          )}
        />
      ))}
    </div>
  );
}
