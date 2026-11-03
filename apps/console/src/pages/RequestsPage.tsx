import { useMemo, useState } from "react";
import { useSearchParams } from "react-router";
import { useRequests } from "@/api/queries";
import { OUTCOMES, PROTOCOLS, type AuditEntry, type Outcome, type ProtocolName } from "@/api/types";
import { PROTOCOL_LABEL } from "@/api/protocols";
import { PageHeader } from "@/components/PageHeader";
import { Command, EmptyState, ErrorState, Skeleton } from "@/components/States";
import { OutcomeTag } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Input, Select } from "@/components/ui/Field";
import { Segmented } from "@/components/ui/Segmented";
import { Switch } from "@/components/ui/Switch";
import { useT } from "@/i18n";
import { cn } from "@/lib/cn";
import { fmtClock, fmtInt, fmtMs, fmtNum, shortId } from "@/lib/format";
import { RequestDrawer } from "./requests/RequestDrawer";
import { TimingBar } from "./requests/TimingBar";

const LIMITS = [
  { value: "200", label: "200" },
  { value: "500", label: "500" },
  { value: "1000", label: "1000" },
] as const;

const PAGE = 100;
const PROTOCOL_SHORT: Record<ProtocolName, string> = { chat_completions: "chat", responses: "responses", messages: "messages" };

function workerLabel(entry: AuditEntry) {
  if (entry.prefill_worker_id && entry.worker_id) return `${entry.prefill_worker_id} → ${entry.worker_id}`;
  return entry.worker_id ?? "—";
}

export function RequestsPage() {
  const t = useT();
  const k = t.requests;
  const [params, setParams] = useSearchParams();
  const selected = params.get("id");
  const [limit, setLimit] = useState(500);
  const [live, setLive] = useState(true);
  const [outcome, setOutcome] = useState<Outcome | "all">("all");
  const [worker, setWorker] = useState("all");
  const [protocol, setProtocol] = useState<ProtocolName | "all">("all");
  const [query, setQuery] = useState("");
  const [shown, setShown] = useState(PAGE);
  const requests = useRequests(limit, live ? 2_000 : false);
  // The log is written when a request finishes; the table is ordered by arrival.
  const entries = useMemo(() => [...(requests.data?.requests ?? [])].sort((a, b) => b.received_at_ms - a.received_at_ms), [requests.data]);

  const workers = useMemo(() => {
    const ids = new Set<string>();
    for (const entry of entries) {
      if (entry.worker_id) ids.add(entry.worker_id);
      if (entry.prefill_worker_id) ids.add(entry.prefill_worker_id);
    }
    return [...ids].sort();
  }, [entries]);

  const counts = useMemo(() => {
    const result: Record<string, number> = { all: entries.length };
    for (const entry of entries) result[entry.outcome] = (result[entry.outcome] ?? 0) + 1;
    return result;
  }, [entries]);

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return entries.filter(
      (entry) =>
        (outcome === "all" || entry.outcome === outcome) &&
        (worker === "all" || entry.worker_id === worker || entry.prefill_worker_id === worker) &&
        (protocol === "all" || entry.protocol === protocol) &&
        (!needle ||
          entry.request_id.toLowerCase().includes(needle) ||
          entry.response_id.toLowerCase().includes(needle) ||
          entry.tenant.toLowerCase().includes(needle)),
    );
  }, [entries, outcome, worker, protocol, query]);

  const visible = filtered.slice(0, shown);
  const maxE2e = Math.max(1, ...visible.map((entry) => entry.e2e_ms));

  const open = (id: string | null) => {
    const next = new URLSearchParams(params);
    if (id) next.set("id", id);
    else next.delete("id");
    setParams(next, { replace: true });
  };

  return (
    <div className="flex flex-col gap-5">
      <PageHeader
        title={k.title}
        description={k.description}
        actions={
          <>
            <div className="flex items-center gap-2">
              <span className="text-xs text-ink-3">{k.fetch}</span>
              <Segmented label={k.fetchLabel} size="sm" value={String(limit)} options={LIMITS} onChange={(value) => setLimit(Number(value))} />
            </div>
            <Switch label={k.autoRefresh} checked={live} onChange={setLive} stateText={live ? k.every2s : k.paused} />
          </>
        }
      />

      <div className="flex flex-wrap items-center gap-x-5 gap-y-3">
        <Segmented<Outcome | "all">
          label={k.outcomeFilter}
          size="sm"
          value={outcome}
          onChange={(value) => {
            setOutcome(value);
            setShown(PAGE);
          }}
          options={[
            { value: "all", label: k.allCount(fmtInt(counts.all ?? 0)) },
            ...OUTCOMES.map((o) => ({ value: o, label: `${o} ${fmtInt(counts[o] ?? 0)}` })),
          ]}
        />
        <label className="flex items-center gap-2 text-xs text-ink-3">
          {k.worker}
          <Select value={worker} onChange={(event) => setWorker(event.target.value)} className="h-7 w-40 font-mono text-xs">
            <option value="all">{t.common.all}</option>
            {workers.map((id) => (
              <option key={id} value={id}>
                {id}
              </option>
            ))}
          </Select>
        </label>
        <label className="flex items-center gap-2 text-xs text-ink-3">
          {k.protocol}
          <Select
            value={protocol}
            onChange={(event) => setProtocol(event.target.value as ProtocolName | "all")}
            className="h-7 w-44 text-xs"
          >
            <option value="all">{t.common.all}</option>
            {PROTOCOLS.map((p) => (
              <option key={p} value={p}>
                {PROTOCOL_LABEL[p]}
              </option>
            ))}
          </Select>
        </label>
        <Input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder={k.search}
          className="h-7 w-72 text-xs"
          aria-label={k.search}
        />
      </div>

      {requests.error ? (
        <ErrorState error={requests.error} what={k.readError} />
      ) : requests.isPending ? (
        <Skeleton className="h-96 w-full" />
      ) : entries.length === 0 ? (
        <div className="border-y-[1.5px] border-ink">
          <EmptyState title={k.emptyTitle} className="px-1">
            <p>{k.emptyBody}</p>
            <Command>{"cargo run -p lev-bench -- --requests 100 --concurrency 8 --policies cache_aware"}</Command>
          </EmptyState>
        </div>
      ) : filtered.length === 0 ? (
        <div className="border-y-[1.5px] border-ink">
          <EmptyState
            title={k.noneTitle}
            className="px-1"
            action={
              <Button
                size="sm"
                onClick={() => {
                  setOutcome("all");
                  setWorker("all");
                  setProtocol("all");
                  setQuery("");
                }}
              >
                {k.clear}
              </Button>
            }
          >
            <p>{k.noneBody(fmtInt(entries.length))}</p>
          </EmptyState>
        </div>
      ) : (
        <>
          <div className="overflow-x-auto">
            <table className="table-3l min-w-[1040px]">
              <thead>
                <tr>
                  <th>{k.received}</th>
                  <th>{k.request}</th>
                  <th>{k.protocol}</th>
                  <th>{k.tenant}</th>
                  <th>{k.worker}</th>
                  <th>{k.policy}</th>
                  <th className="num">{k.tokens}</th>
                  <th className="num">
                    TTFT <span className="unit">(ms)</span>
                  </th>
                  <th className="num">
                    E2E <span className="unit">(ms)</span>
                  </th>
                  <th className="w-[13%]">
                    {k.timeline} <span className="unit">{k.sharedAxis(fmtMs(maxE2e))}</span>
                  </th>
                  <th>{k.outcome}</th>
                </tr>
              </thead>
              <tbody>
                {visible.map((entry) => (
                  <tr
                    key={entry.request_id}
                    className="row-action"
                    aria-selected={entry.request_id === selected}
                    onClick={() => open(entry.request_id)}
                    tabIndex={0}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        open(entry.request_id);
                      }
                    }}
                  >
                    <td className="text-xs text-ink-2 tabular-nums">{fmtClock(entry.received_at_ms, true)}</td>
                    <td className="font-mono text-xs">{shortId(entry.request_id)}</td>
                    <td className="text-xs">{PROTOCOL_SHORT[entry.protocol]}</td>
                    <td className="max-w-[7rem] truncate text-xs">{entry.tenant}</td>
                    <td className="max-w-[12rem] truncate font-mono text-xs">{workerLabel(entry)}</td>
                    <td className="font-mono text-xs text-ink-2">{entry.policy ?? "—"}</td>
                    <td className="num text-xs">
                      {fmtInt(entry.prompt_tokens)}
                      <span className="text-ink-3">
                        {" "}
                        / {fmtInt(entry.cached_tokens)} / {fmtInt(entry.completion_tokens)}
                      </span>
                    </td>
                    <td className="num text-xs">{entry.ttft_ms === null ? "—" : fmtNum(entry.ttft_ms, entry.ttft_ms < 10 ? 1 : 0)}</td>
                    <td className="num text-xs">{fmtNum(entry.e2e_ms, entry.e2e_ms < 10 ? 1 : 0)}</td>
                    <td>
                      <TimingBar ttft={entry.ttft_ms} e2e={entry.e2e_ms} max={maxE2e} />
                    </td>
                    <td>
                      <OutcomeTag outcome={entry.outcome} />
                      {entry.attempts > 1 ? <span className="ml-1.5 text-2xs text-warn">{k.retries(entry.attempts - 1)}</span> : null}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className={cn("flex items-center justify-between gap-3 text-xs text-ink-3")}>
            <span>{k.footer(fmtInt(visible.length), fmtInt(filtered.length), fmtInt(entries.length))}</span>
            {visible.length < filtered.length ? (
              <Button size="sm" onClick={() => setShown((n) => n + PAGE)}>
                {k.more(Math.min(PAGE, filtered.length - visible.length))}
              </Button>
            ) : null}
          </div>
        </>
      )}

      <RequestDrawer id={selected} onClose={() => open(null)} />
    </div>
  );
}
