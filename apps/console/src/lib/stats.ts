import type { AuditEntry, Outcome, ProtocolName, RequestDoneEvent } from "@/api/types";

/**
 * One completed request, merged from the live `request_done` event and the
 * audit log entry with the same request id. The live event arrives first; the
 * audit entry adds TPOT and speculative-decoding counts.
 */
export interface Sample {
  id: string;
  /** When the request finished (epoch ms). */
  doneAt: number;
  outcome: Outcome;
  protocol: ProtocolName | null;
  worker: string | null;
  prompt: number;
  cached: number;
  completion: number;
  ttft: number | null;
  e2e: number;
  tpot: number | null;
  draft: number | null;
  accepted: number | null;
}

export function sampleFromDone(event: RequestDoneEvent): Sample {
  return {
    id: event.request_id,
    doneAt: event.at_ms,
    outcome: event.outcome,
    protocol: event.protocol ?? null,
    worker: event.worker_id,
    prompt: event.prompt_tokens,
    cached: event.cached_tokens,
    completion: event.completion_tokens,
    ttft: event.ttft_ms,
    e2e: event.e2e_ms,
    tpot: event.tpot_ms ?? null,
    draft: event.spec_draft_tokens ?? null,
    accepted: event.spec_accepted_tokens ?? null,
  };
}

export function sampleFromAudit(entry: AuditEntry): Sample {
  return {
    id: entry.request_id,
    doneAt: entry.received_at_ms + entry.e2e_ms,
    outcome: entry.outcome,
    protocol: entry.protocol,
    worker: entry.worker_id,
    prompt: entry.prompt_tokens,
    cached: entry.cached_tokens,
    completion: entry.completion_tokens,
    ttft: entry.ttft_ms,
    e2e: entry.e2e_ms,
    tpot: entry.tpot_ms,
    draft: entry.spec_draft_tokens,
    accepted: entry.spec_accepted_tokens,
  };
}

/** Merge samples by id; audit fields fill what the live event lacks. */
export function mergeSamples(live: Sample[], audit: Sample[]): Sample[] {
  const byId = new Map<string, Sample>();
  for (const sample of audit) byId.set(sample.id, sample);
  for (const sample of live) {
    const known = byId.get(sample.id);
    byId.set(
      sample.id,
      known
        ? {
            ...known,
            doneAt: sample.doneAt,
            tpot: sample.tpot ?? known.tpot,
            draft: sample.draft ?? known.draft,
            accepted: sample.accepted ?? known.accepted,
            protocol: sample.protocol ?? known.protocol,
          }
        : sample,
    );
  }
  return [...byId.values()].sort((a, b) => a.doneAt - b.doneAt);
}

/** Quantile of an ascending array (linear interpolation). */
export function quantile(sorted: number[], q: number): number | null {
  if (sorted.length === 0) return null;
  if (sorted.length === 1) return sorted[0] ?? null;
  const position = (sorted.length - 1) * q;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const a = sorted[lower] ?? 0;
  const b = sorted[upper] ?? a;
  return a + (b - a) * (position - lower);
}

function ascending(values: number[]) {
  return values.slice().sort((a, b) => a - b);
}

export interface WindowStats {
  /** Length of time actually observed inside the window, ms. */
  spanMs: number;
  count: number;
  ok: number;
  failed: number;
  rps: number | null;
  ttftP50: number | null;
  ttftP99: number | null;
  ttftCount: number;
  tpotP50: number | null;
  tpotCount: number;
  hitRate: number | null;
  promptTokens: number;
  cachedTokens: number;
  outTokensPerS: number | null;
  completionTokens: number;
  acceptRate: number | null;
  draftTokens: number;
  acceptedTokens: number;
}

/**
 * Statistics over requests finished in (now - windowMs, now], counting only
 * time the console could observe (from `coverageStart`).
 */
export function windowStats(samples: Sample[], now: number, windowMs: number, coverageStart: number | null): WindowStats {
  const start = Math.max(now - windowMs, coverageStart ?? now);
  const spanMs = Math.max(0, now - start);
  const inWindow = samples.filter((sample) => sample.doneAt > start && sample.doneAt <= now);
  const served = inWindow.filter((sample) => sample.outcome !== "rejected");
  const ttfts = ascending(inWindow.flatMap((s) => (s.outcome === "ok" && s.ttft !== null ? [s.ttft] : [])));
  const tpots = ascending(inWindow.flatMap((s) => (s.outcome === "ok" && s.tpot !== null ? [s.tpot] : [])));
  const promptTokens = served.reduce((sum, s) => sum + s.prompt, 0);
  const cachedTokens = served.reduce((sum, s) => sum + s.cached, 0);
  const completionTokens = inWindow.reduce((sum, s) => sum + s.completion, 0);
  const withSpec = inWindow.filter((s) => s.draft !== null && s.accepted !== null);
  const draftTokens = withSpec.reduce((sum, s) => sum + (s.draft ?? 0), 0);
  const acceptedTokens = withSpec.reduce((sum, s) => sum + (s.accepted ?? 0), 0);
  const seconds = spanMs / 1000;
  const enough = seconds >= 1;
  return {
    spanMs,
    count: inWindow.length,
    ok: inWindow.filter((s) => s.outcome === "ok").length,
    failed: inWindow.filter((s) => s.outcome === "error" || s.outcome === "rejected").length,
    rps: enough ? inWindow.length / seconds : null,
    ttftP50: quantile(ttfts, 0.5),
    ttftP99: quantile(ttfts, 0.99),
    ttftCount: ttfts.length,
    tpotP50: quantile(tpots, 0.5),
    tpotCount: tpots.length,
    hitRate: promptTokens > 0 ? cachedTokens / promptTokens : null,
    promptTokens,
    cachedTokens,
    outTokensPerS: enough ? completionTokens / seconds : null,
    completionTokens,
    acceptRate: draftTokens > 0 ? acceptedTokens / draftTokens : null,
    draftTokens,
    acceptedTokens,
  };
}

export interface Bucket {
  /** Bucket end (epoch ms). */
  t: number;
  /** null: no defined value (no requests, or not observed). */
  value: number | null;
  /** The console was observing during this whole bucket. */
  covered: boolean;
}

export type BucketMetric = (samples: Sample[], seconds: number) => number | null;

/** Fixed-width time buckets ending at `now`, for live charts. */
export function bucketSeries(
  samples: Sample[],
  now: number,
  spanMs: number,
  bucketMs: number,
  coverageStart: number | null,
  metric: BucketMetric,
): Bucket[] {
  const count = Math.round(spanMs / bucketMs);
  const end = Math.floor(now / bucketMs) * bucketMs;
  const first = end - count * bucketMs;
  const groups: Sample[][] = Array.from({ length: count }, () => []);
  for (const sample of samples) {
    if (sample.doneAt <= first || sample.doneAt > end) continue;
    const index = Math.min(count - 1, Math.floor((sample.doneAt - first - 1) / bucketMs));
    groups[index]?.push(sample);
  }
  return groups.map((group, index) => {
    const bucketStart = first + index * bucketMs;
    const bucketEnd = bucketStart + bucketMs;
    const covered = coverageStart !== null && coverageStart <= bucketStart;
    return { t: bucketEnd, value: covered ? metric(group, bucketMs / 1000) : null, covered };
  });
}

export const metrics = {
  rps: ((group, seconds) => group.length / seconds) as BucketMetric,
  ttftP50: ((group) =>
    quantile(ascending(group.flatMap((s) => (s.outcome === "ok" && s.ttft !== null ? [s.ttft] : []))), 0.5)) as BucketMetric,
  ttftP99: ((group) =>
    quantile(ascending(group.flatMap((s) => (s.outcome === "ok" && s.ttft !== null ? [s.ttft] : []))), 0.99)) as BucketMetric,
  tpotP50: ((group) =>
    quantile(ascending(group.flatMap((s) => (s.outcome === "ok" && s.tpot !== null ? [s.tpot] : []))), 0.5)) as BucketMetric,
  hitRate: ((group) => {
    const served = group.filter((s) => s.outcome !== "rejected");
    const prompt = served.reduce((sum, s) => sum + s.prompt, 0);
    return prompt > 0 ? served.reduce((sum, s) => sum + s.cached, 0) / prompt : null;
  }) as BucketMetric,
  outTps: ((group, seconds) => group.reduce((sum, s) => sum + s.completion, 0) / seconds) as BucketMetric,
  acceptRate: ((group) => {
    const withSpec = group.filter((s) => s.draft !== null && s.accepted !== null);
    const draft = withSpec.reduce((sum, s) => sum + (s.draft ?? 0), 0);
    return draft > 0 ? withSpec.reduce((sum, s) => sum + (s.accepted ?? 0), 0) / draft : null;
  }) as BucketMetric,
};
