import { getLocale } from "@/i18n";

const intFormat = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });

export function fmtInt(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return intFormat.format(value);
}

/** Up to `digits` decimals, dropping trailing zeros. */
export function fmtNum(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: digits }).format(value);
}

/** Milliseconds, with precision that follows magnitude. */
export function fmtMs(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) return "—";
  if (ms < 1) return `${ms.toFixed(2)} ms`;
  if (ms < 10) return `${ms.toFixed(1)} ms`;
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 10_000) return `${(ms / 1000).toFixed(2)} s`;
  return `${(ms / 1000).toFixed(1)} s`;
}

/** The number part of fmtMs and its unit, for tables that set units apart. */
export function splitMs(ms: number | null | undefined): { value: string; unit: string } {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) return { value: "—", unit: "" };
  if (ms < 10) return { value: ms.toFixed(ms < 1 ? 2 : 1), unit: "ms" };
  if (ms < 1000) return { value: String(Math.round(ms)), unit: "ms" };
  return { value: (ms / 1000).toFixed(ms < 10_000 ? 2 : 1), unit: "s" };
}

export function fmtPct(ratio: number | null | undefined, digits = 1): string {
  if (ratio === null || ratio === undefined || !Number.isFinite(ratio)) return "—";
  return `${(ratio * 100).toFixed(digits)}%`;
}

export function fmtTokens(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  if (value >= 10_000) return `${(value / 1000).toFixed(1)}k`;
  return intFormat.format(value);
}

function pad(value: number, width = 2) {
  return String(value).padStart(width, "0");
}

/** Wall-clock time, 24 h. */
export function fmtClock(epochMs: number, withMs = false): string {
  const date = new Date(epochMs);
  const base = `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
  return withMs ? `${base}.${pad(date.getMilliseconds(), 3)}` : base;
}

export function fmtDateTime(epochMs: number): string {
  const date = new Date(epochMs);
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${fmtClock(epochMs, true)}`;
}

export function fmtUptime(seconds: number): string {
  const d = Math.floor(seconds / 86_400);
  const h = Math.floor((seconds % 86_400) / 3_600);
  const m = Math.floor((seconds % 3_600) / 60);
  const s = Math.floor(seconds % 60);
  if (getLocale() === "zh") {
    if (d > 0) return `${d} 天 ${h} 小时`;
    if (h > 0) return `${h} 小时 ${m} 分`;
    if (m > 0) return `${m} 分 ${s} 秒`;
    return `${s} 秒`;
  }
  if (d > 0) return `${d} d ${h} h`;
  if (h > 0) return `${h} h ${m} min`;
  if (m > 0) return `${m} min ${s} s`;
  return `${s} s`;
}

/** Microseconds for sub-millisecond spans, otherwise fmtMs. */
export function fmtSpan(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) return "—";
  if (ms < 1) return `${Math.max(1, Math.round(ms * 1000))} µs`;
  return fmtMs(ms);
}

/** A UTC timestamp (RFC 3339) as local date and time, without seconds. */
export function fmtStamp(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

/**
 * A short, distinguishing form of a request id. UUIDv7 ids start with a
 * millisecond timestamp, so their head repeats across requests; the random
 * tail does not.
 */
export function shortId(id: string, length = 8): string {
  const bare = id.replace(/^(chatcmpl-|resp_|msg_|replay-)/, "").replace(/-/g, "");
  return bare.slice(-length);
}

export function fmtBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
}
