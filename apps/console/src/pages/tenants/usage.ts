import type { DailyUsage } from "@/api/types";

export interface DayTotals {
  day: string;
  requests: number;
  uncached: number;
  cached: number;
  completion: number;
}

/** UTC days from `days - 1` days ago through today, as YYYY-MM-DD. */
export function dayRange(days: number, now = Date.now()): string[] {
  const out: string[] = [];
  const today = new Date(now);
  const base = Date.UTC(today.getUTCFullYear(), today.getUTCMonth(), today.getUTCDate());
  for (let i = days - 1; i >= 0; i -= 1) out.push(new Date(base - i * 86_400_000).toISOString().slice(0, 10));
  return out;
}

/** Sum ledger rows per day; days without rows had no usage. */
export function totalsByDay(rows: DailyUsage[], days: string[]): DayTotals[] {
  const byDay = new Map<string, DayTotals>(days.map((day) => [day, { day, requests: 0, uncached: 0, cached: 0, completion: 0 }]));
  for (const row of rows) {
    const entry = byDay.get(row.day);
    if (!entry) continue;
    entry.requests += row.requests;
    entry.cached += row.cached_tokens;
    entry.uncached += Math.max(0, row.prompt_tokens - row.cached_tokens);
    entry.completion += row.completion_tokens;
  }
  return days.map((day) => byDay.get(day) as DayTotals);
}
