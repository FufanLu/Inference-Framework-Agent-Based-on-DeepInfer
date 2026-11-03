import { useMemo, useState, type ReactNode } from "react";
import { Link, useNavigate } from "react-router";
import { ArrowRight } from "lucide-react";
import { useLiveData } from "@/api/live";
import { useOverview, useRouterSnapshot, useStats } from "@/api/queries";
import type { StatsResult } from "@/api/types";
import { LiveChart, type ChartSeries } from "@/components/charts/LiveChart";
import { KvCells } from "@/components/KvCells";
import { PageHeader, Section } from "@/components/PageHeader";
import { Command, EmptyState, ErrorState, Skeleton } from "@/components/States";
import { OutcomeTag, RoleTag } from "@/components/ui/Badge";
import { Segmented } from "@/components/ui/Segmented";
import { StatusMark } from "@/components/ui/StatusMark";
import { useT, type Messages } from "@/i18n";
import { workerLineState } from "@/lib/workers";
import { cn } from "@/lib/cn";
import { fmtClock, fmtInt, fmtNum, fmtPct, fmtUptime, shortId, splitMs } from "@/lib/format";
import { bucketSeries, metrics, type Sample } from "@/lib/stats";
import { useNow } from "@/lib/useNow";
import { useTelemetry } from "@/state/telemetry";

const WINDOWS = [
  { value: "30", label: "30 s" },
  { value: "60", label: "60 s" },
  { value: "300", label: "5 min" },
] as const;

const CHART_SPAN_MS = 5 * 60_000;
const BUCKET_MS = 10_000;
export const BENCH_COMMAND = "cargo run -p lev-bench -- --requests 100 --concurrency 8 --policies cache_aware";

function windowLabel(seconds: number) {
  return seconds >= 120 ? `${seconds / 60} min` : `${seconds} s`;
}

interface CharacteristicProps {
  name: string;
  symbol: string;
  value: ReactNode;
  conditions: ReactNode;
  chart: ReactNode;
}

/** One measured characteristic: name, symbol, value under stated conditions, and its curve. */
function Characteristic({ name, symbol, value, conditions, chart }: CharacteristicProps) {
  return (
    <div className="flex min-w-0 flex-col gap-2 bg-paper px-4 pt-3 pb-3">
      <div className="flex items-baseline justify-between gap-3">
        <h3 className="cond text-md leading-6 font-semibold text-ink">{name}</h3>
        <span className="cond text-xs text-ink-3">{symbol}</span>
      </div>
      <div className="flex min-h-[2.5rem] flex-wrap items-baseline gap-x-4 tabular-nums">{value}</div>
      <p className="min-h-[1.125rem] text-xs text-ink-3 tabular-nums">{conditions}</p>
      <div className="mt-1">{chart}</div>
    </div>
  );
}

function Reading({ value, unit, qualifier }: { value: string; unit?: string; qualifier?: string }) {
  return (
    <span className="inline-flex items-baseline gap-1">
      {qualifier ? <span className="cond mr-0.5 text-xs font-semibold text-ink-2">{qualifier}</span> : null}
      <span className={cn("text-2xl font-semibold tracking-[-0.02em]", value === "—" ? "text-ink-3" : "text-ink")}>{value}</span>
      {unit ? <span className="text-sm text-ink-2">{unit}</span> : null}
    </span>
  );
}

function MsReading({ ms, qualifier }: { ms: number | null; qualifier?: string }) {
  const { value, unit } = splitMs(ms);
  return <Reading value={value} unit={unit || undefined} qualifier={qualifier} />;
}

function Characteristics({ stats, windowS, uptime, series, t }: {
  stats: StatsResult;
  windowS: number;
  uptime: number | null;
  series: Record<string, ChartSeries["data"]>;
  t: Messages;
}) {
  const k = t.overview;
  const ok = stats.outcomes.ok ?? 0;
  const failed = (stats.outcomes.error ?? 0) + (stats.outcomes.rejected ?? 0);
  const short = uptime !== null && uptime < windowS ? ` · ${k.upFor(String(uptime))}` : "";
  const span = `${k.last(windowLabel(windowS))}${short}`;
  const completion = Math.round(stats.output_tokens_per_s * stats.window_s);
  const chart = (label: string, s: ChartSeries[], unit: string, format: (v: number) => string, extra?: { fixedMax?: number; minMax?: number }) => (
    <LiveChart
      series={s}
      bucketMs={BUCKET_MS}
      unit={unit}
      format={format}
      windowMs={windowS * 1000}
      windowLabel={windowLabel(windowS)}
      ariaLabel={t.chart.aria(label)}
      {...extra}
    />
  );
  const data = (key: string) => series[key] ?? [];
  return (
    <>
      <Characteristic
        name={k.kpi.rps}
        symbol="requests/s"
        value={<Reading value={fmtNum(stats.requests_per_s, 2)} unit="req/s" />}
        conditions={
          <>
            {span} · {k.requests(fmtInt(stats.n))}
            {failed > 0 ? <span className="text-bad"> · {k.failed(fmtInt(failed))}</span> : null} · {k.source}
          </>
        }
        chart={chart(k.kpi.rps, [{ label: "req/s", data: data("rps") }], "req/s", (v) => fmtNum(v, 1))}
      />
      <Characteristic
        name={k.kpi.ttft}
        symbol="TTFT"
        value={
          <>
            <MsReading ms={stats.ttft_ms.p50} qualifier="p50" />
            <MsReading ms={stats.ttft_ms.p99} qualifier="p99" />
          </>
        }
        conditions={ok > 0 ? `${span} · ${k.n(fmtInt(ok))} · ${k.ttftNote}` : `${span} · ${k.noOk}`}
        chart={chart(
          k.kpi.ttft,
          [
            { label: "p50", data: data("ttftP50") },
            { label: "p99", data: data("ttftP99"), secondary: true },
          ],
          "ms",
          (v) => fmtNum(v, 0),
          { minMax: 10 },
        )}
      />
      <Characteristic
        name={k.kpi.tpot}
        symbol="TPOT"
        value={
          <>
            <MsReading ms={stats.tpot_ms.p50} qualifier="p50" />
            <MsReading ms={stats.tpot_ms.p90} qualifier="p90" />
          </>
        }
        conditions={stats.tpot_ms.p50 !== null ? `${span} · ${k.n(fmtInt(ok))} · ${k.tpotNote}` : `${span} · ${k.tpotNone}`}
        chart={chart(k.kpi.tpot, [{ label: "p50", data: data("tpot") }], "ms", (v) => fmtNum(v, 1), { minMax: 1 })}
      />
      <Characteristic
        name={k.kpi.hit}
        symbol="cached / prompt"
        value={<Reading value={stats.cache_hit_rate === null ? "—" : (stats.cache_hit_rate * 100).toFixed(1)} unit="%" />}
        conditions={
          stats.prompt_tokens > 0 ? `${span} · ${k.hit(fmtInt(stats.cached_tokens), fmtInt(stats.prompt_tokens))}` : `${span} · ${k.hitNone}`
        }
        chart={chart(k.kpi.hit, [{ label: "hit", data: data("hit") }], "%", (v) => `${Math.round(v * 100)}`, { fixedMax: 1 })}
      />
      <Characteristic
        name={k.kpi.out}
        symbol="output tokens/s"
        value={<Reading value={fmtNum(stats.output_tokens_per_s, 0)} unit="tok/s" />}
        conditions={`${span} · ${k.out(fmtInt(completion))}`}
        chart={chart(k.kpi.out, [{ label: "tok/s", data: data("out") }], "tok/s", (v) => fmtNum(v, 0))}
      />
      <Characteristic
        name={k.kpi.accept}
        symbol="accepted / draft"
        value={<Reading value={stats.spec_accept_rate === null ? "—" : (stats.spec_accept_rate * 100).toFixed(1)} unit="%" />}
        conditions={stats.spec_accept_rate !== null ? `${span} · ${k.accept} · ${k.source}` : `${span} · ${k.acceptNone}`}
        chart={chart(k.kpi.accept, [{ label: "accept", data: data("accept") }], "%", (v) => `${Math.round(v * 100)}`, { fixedMax: 1 })}
      />
    </>
  );
}

export function OverviewPage() {
  const t = useT();
  const [windowS, setWindowS] = useState(60);
  const now = useNow(1000);
  const telemetry = useTelemetry();
  const stats = useStats(windowS, 2_000);
  const overview = useOverview();
  const { samples, coverageStart } = telemetry;

  // Curves advance every 2 s, not on every clock tick.
  const chartNow = Math.floor(now / 2000) * 2000;
  const series = useMemo(() => {
    const make = (metric: (typeof metrics)[keyof typeof metrics]) =>
      bucketSeries(samples, chartNow, CHART_SPAN_MS, BUCKET_MS, coverageStart, metric);
    return {
      rps: make(metrics.rps),
      ttftP50: make(metrics.ttftP50),
      ttftP99: make(metrics.ttftP99),
      tpot: make(metrics.tpotP50),
      hit: make(metrics.hitRate),
      out: make(metrics.outTps),
      accept: make(metrics.acceptRate),
    };
  }, [samples, chartNow, coverageStart]);

  const uptime = overview.data?.uptime_s ?? null;

  return (
    <div className="flex flex-col gap-7">
      <PageHeader
        title={t.overview.title}
        description={t.overview.description}
        actions={
          <div className="flex items-center gap-2.5">
            <span className="text-xs text-ink-3">{t.overview.window}</span>
            <Segmented label={t.overview.window} size="sm" value={String(windowS)} options={WINDOWS} onChange={(value) => setWindowS(Number(value))} />
          </div>
        }
      />

      <IdentityLine />

      {stats.error && !stats.data ? (
        <ErrorState error={stats.error} what="/admin/stats" />
      ) : (
        <section aria-label={t.overview.title} className="flex flex-col gap-2">
          <div className="grid grid-cols-1 gap-px border-y-[1.5px] border-ink bg-rule md:grid-cols-2 2xl:grid-cols-3">
            {!stats.data ? (
              Array.from({ length: 6 }, (_, i) => (
                <div key={i} className="flex flex-col gap-3 bg-paper px-4 py-4">
                  <Skeleton className="h-4 w-28" />
                  <Skeleton className="h-8 w-36" />
                  <Skeleton className="h-24 w-full" />
                </div>
              ))
            ) : (
              <Characteristics stats={stats.data} windowS={windowS} uptime={uptime} series={series} t={t} />
            )}
          </div>
          <p className="text-2xs text-ink-3">{t.overview.chartsNote}</p>
        </section>
      )}

      {stats.data && stats.data.n === 0 && samples.length === 0 ? (
        <EmptyState title={t.overview.noTrafficTitle} compact className="-mt-3">
          <p>{t.overview.noTrafficBody}</p>
          <Command>{BENCH_COMMAND}</Command>
        </EmptyState>
      ) : null}

      <div className="grid grid-cols-1 gap-8 xl:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
        <RecentRequests samples={samples} />
        <ClusterSummary />
      </div>
    </div>
  );
}

function IdentityLine() {
  const t = useT();
  const overview = useOverview();
  if (overview.error) return <ErrorState error={overview.error} what="/admin/overview" compact />;
  if (!overview.data) return <Skeleton className="h-5 w-[min(640px,100%)]" />;
  const o = overview.data;
  const k = t.overview.identity;
  const items: [string, ReactNode][] = [
    [k.gateway, <span className="font-mono">v{o.version}</span>],
    [k.instance, <span className="font-mono">{o.fingerprint}</span>],
    [k.uptime, fmtUptime(o.uptime_s)],
    [k.model, o.models.join(", ")],
    [k.policy, <span className="font-mono">{o.policy}</span>],
    [k.index, <span className="font-mono">{o.index_mode}</span>],
    [k.pd, o.disaggregation ? t.common.on : t.common.off],
    [k.prefixIndex, t.common.blocks(fmtInt(o.indexed_blocks))],
  ];
  return (
    <dl className="-mt-2 flex flex-wrap gap-x-6 gap-y-1.5 text-sm">
      {items.map(([label, value]) => (
        <div key={label} className="flex items-baseline gap-2">
          <dt className="text-xs text-ink-3">{label}</dt>
          <dd className="text-ink">{value}</dd>
        </div>
      ))}
    </dl>
  );
}

function RecentRequests({ samples }: { samples: Sample[] }) {
  const t = useT();
  const k = t.overview.recent;
  const live = useLiveData();
  const navigate = useNavigate();
  const rows = useMemo(() => samples.slice(-14).reverse(), [samples]);
  const liveIds = useMemo(() => new Set(live.done.slice(-30).map((d) => d.request_id)), [live.done]);

  return (
    <Section
      title={k.title}
      caption={k.caption}
      aside={
        <Link to="/requests" className="inline-flex items-center gap-1 text-xs text-ink-2 hover:text-ink">
          {k.all} <ArrowRight className="size-3" />
        </Link>
      }
    >
      {rows.length === 0 ? (
        <div className="border-y-[1.5px] border-ink">
          <EmptyState compact title={k.emptyTitle} className="px-1">
            <p>{k.emptyBody}</p>
          </EmptyState>
        </div>
      ) : (
        <table className="table-3l">
          <thead>
            <tr>
              <th>{k.finished}</th>
              <th>{k.request}</th>
              <th>{k.worker}</th>
              <th className="num">
                {k.promptCached} <span className="unit">(tok)</span>
              </th>
              <th className="num">
                {k.output} <span className="unit">(tok)</span>
              </th>
              <th className="num">
                TTFT <span className="unit">(ms)</span>
              </th>
              <th className="num">
                E2E <span className="unit">(ms)</span>
              </th>
              <th>{k.outcome}</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr
                key={row.id}
                className={cn("row-action", liveIds.has(row.id) && "anim-row-in")}
                onClick={() => navigate(`/requests?id=${encodeURIComponent(row.id)}`)}
              >
                <td className="text-xs text-ink-2 tabular-nums">{fmtClock(row.doneAt, true)}</td>
                <td className="font-mono text-xs">{shortId(row.id)}</td>
                <td className="font-mono text-xs text-ink-2">{row.worker ?? "—"}</td>
                <td className="num">
                  {fmtInt(row.prompt)}
                  <span className="text-ink-3"> / {fmtInt(row.cached)}</span>
                </td>
                <td className="num">{fmtInt(row.completion)}</td>
                <td className="num">{row.ttft === null ? "—" : fmtNum(row.ttft, row.ttft < 10 ? 1 : 0)}</td>
                <td className="num">{fmtNum(row.e2e, row.e2e < 10 ? 1 : 0)}</td>
                <td>
                  <OutcomeTag outcome={row.outcome} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Section>
  );
}

function ClusterSummary() {
  const t = useT();
  const k = t.overview.cluster;
  // The same live source and cadence as the topology page.
  const router = useRouterSnapshot(1_000);
  const snapshot = router.data;
  return (
    <Section
      title={k.title}
      caption={snapshot ? k.caption(snapshot.policy, snapshot.index_mode) : undefined}
      aside={
        <Link to="/cluster" className="inline-flex items-center gap-1 text-xs text-ink-2 hover:text-ink">
          {k.link} <ArrowRight className="size-3" />
        </Link>
      }
    >
      {router.error ? (
        <ErrorState error={router.error} what="/admin/router" compact />
      ) : !snapshot ? (
        <Skeleton className="h-40 w-full" />
      ) : snapshot.workers.length === 0 ? (
        <div className="border-y-[1.5px] border-ink">
          <EmptyState compact title={k.emptyTitle} className="px-1">
            <p>
              {k.emptyBefore} <code className="font-mono text-ink">--worker http://127.0.0.1:50051</code>
              {k.emptyAfter}
            </p>
          </EmptyState>
        </div>
      ) : (
        <>
          <table className="table-3l">
            <thead>
              <tr>
                <th>{k.worker}</th>
                <th>{k.role}</th>
                <th>{k.state}</th>
                <th className="w-[30%]">{k.kv}</th>
                <th className="num">{k.seqs}</th>
                <th className="num">{k.inflight}</th>
              </tr>
            </thead>
            <tbody>
              {snapshot.workers.map((worker) => (
                <tr key={worker.id}>
                  <td className="font-mono text-xs whitespace-nowrap">{worker.id}</td>
                  <td>
                    <RoleTag role={worker.role} />
                  </td>
                  <td>
                    <StatusMark {...workerLineState(worker, t)} />
                  </td>
                  <td>
                    <div className="flex items-center gap-2">
                      <KvCells
                        usage={worker.load_age_ms === null ? null : worker.load.kv_usage}
                        cached={worker.total_blocks > 0 && worker.index_source !== "off" ? worker.gpu_blocks / worker.total_blocks : null}
                        cells={10}
                        size="sm"
                        className="flex-1"
                      />
                      <span className="w-10 shrink-0 text-right text-xs tabular-nums text-ink-2">
                        {worker.load_age_ms === null ? "—" : fmtPct(worker.load.kv_usage, worker.load.kv_usage < 0.1 ? 1 : 0)}
                      </span>
                    </div>
                  </td>
                  <td className="num whitespace-nowrap">
                    {worker.load_age_ms === null ? (
                      "—"
                    ) : (
                      <>
                        {worker.load.running}
                        <span className="text-ink-3"> + {worker.load.waiting}</span>
                      </>
                    )}
                  </td>
                  <td className="num">{fmtInt(worker.inflight)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-2 text-xs text-ink-3">{k.footnote}</p>
        </>
      )}
    </Section>
  );
}
