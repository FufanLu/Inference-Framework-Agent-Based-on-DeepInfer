import { useMemo } from "react";
import { useLiveData, useLiveState } from "@/api/live";
import { useOverview, useRequests } from "@/api/queries";
import { mergeSamples, sampleFromAudit, sampleFromDone, type Sample } from "@/lib/stats";

/** Audit entries fetched to seed and enrich the live window. */
export const AUDIT_LIMIT = 1000;

export interface Telemetry {
  samples: Sample[];
  /** Since when the console has a complete record of finished requests. */
  coverageStart: number | null;
  auditError: unknown;
  auditLoading: boolean;
  liveOpen: boolean;
}

/**
 * Finished requests from two sources: `request_done` live events (immediate)
 * and the audit log (complete records with TPOT and DSpark counts, and the
 * history from before this page opened).
 */
export function useTelemetry(): Telemetry {
  const audit = useRequests(AUDIT_LIMIT, 4_000);
  const live = useLiveData();
  const liveState = useLiveState();
  const overview = useOverview();

  const samples = useMemo(() => {
    const fromAudit = (audit.data?.requests ?? []).map(sampleFromAudit);
    const fromLive = live.done.map(sampleFromDone);
    return mergeSamples(fromLive, fromAudit);
  }, [audit.data, live.done]);

  const coverageStart = useMemo(() => {
    const starts: number[] = [];
    const entries = audit.data?.requests;
    if (entries && audit.dataUpdatedAt > 0) {
      if (entries.length < AUDIT_LIMIT) {
        // The log holds every request since the gateway started.
        if (overview.data) starts.push(audit.dataUpdatedAt - overview.data.uptime_s * 1000);
        else if (entries.length > 0) starts.push(Math.min(...entries.map((e) => e.received_at_ms)));
      } else {
        starts.push(Math.min(...entries.map((e) => e.received_at_ms + e.e2e_ms)));
      }
    }
    if (liveState.firstOpenedAt !== null) starts.push(liveState.firstOpenedAt);
    return starts.length > 0 ? Math.min(...starts) : null;
  }, [audit.data, audit.dataUpdatedAt, overview.data, liveState.firstOpenedAt]);

  return {
    samples,
    coverageStart,
    auditError: audit.error,
    auditLoading: audit.isPending,
    liveOpen: liveState.status === "open",
  };
}
