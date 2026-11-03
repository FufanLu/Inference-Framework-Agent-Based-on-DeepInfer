import { Pause, Play } from "lucide-react";
import type { ReactNode } from "react";
import { Link } from "react-router";
import type { CandidateScore, RouteEvent, StageEvent } from "@/api/types";
import { PROTOCOL_LABEL } from "@/api/protocols";
import { Command, EmptyState } from "@/components/States";
import { SectionHeading } from "@/components/PageHeader";
import { Button } from "@/components/ui/Button";
import { useT, type Messages } from "@/i18n";
import { cn } from "@/lib/cn";
import { fmtClock, fmtInt, fmtMs, fmtPct, shortId } from "@/lib/format";
import { planSummary } from "@/lib/router";

interface DecisionPanelProps {
  routes: RouteEvent[];
  stages: StageEvent[];
  /** The decision to show; the newest one when null. */
  pinned: RouteEvent | null;
  onPin: (route: RouteEvent | null) => void;
}

/** Why the router chose this worker: the candidates of the latest decision. */
export function DecisionPanel({ routes, stages, pinned, onPin }: DecisionPanelProps) {
  const t = useT();
  const k = t.cluster.decision;
  const latest = routes[routes.length - 1] ?? null;
  const shown = pinned ?? latest;
  const log = routes.slice(-12).reverse();

  return (
    <aside aria-label={k.title} className="flex min-w-0 flex-col gap-6">
      <section>
        <div className="mb-2 flex items-center justify-between gap-3">
          <div className="flex items-baseline gap-2">
            <SectionHeading>{k.title}</SectionHeading>
            <span className="text-xs text-ink-3">{pinned ? k.paused : k.latest}</span>
          </div>
          {latest ? (
            <Button
              size="sm"
              variant="ghost"
              icon={pinned ? <Play className="size-3.5" /> : <Pause className="size-3.5" />}
              onClick={() => onPin(pinned ? null : latest)}
            >
              {pinned ? k.resume : k.pause}
            </Button>
          ) : null}
        </div>
        {shown ? <Decision route={shown} stages={stages} t={t} /> : <Waiting t={t} />}
      </section>

      {log.length > 0 ? (
        <section>
          <div className="mb-2 flex items-baseline justify-between gap-2">
            <SectionHeading>{k.logTitle}</SectionHeading>
            <span className="text-xs text-ink-3">{k.logCaption(log.length)}</span>
          </div>
          <table className="table-3l table-fixed">
            <thead>
              <tr>
                <th className="w-[7.5rem]">{k.time}</th>
                <th>{k.target}</th>
                <th className="num w-[7.5rem]">{k.hit}</th>
              </tr>
            </thead>
            <tbody>
              {log.map((route) => {
                const chosen = route.plan.kind === "unified" ? route.plan.worker : route.plan.prefill;
                return (
                  <tr
                    key={`${route.request_id}-${route.attempt}`}
                    className="row-action"
                    aria-selected={shown === route}
                    onClick={() => onPin(route)}
                  >
                    <td className="text-xs text-ink-2 tabular-nums" title={route.request_id}>
                      {fmtClock(route.at_ms, true)}
                    </td>
                    <td className="truncate font-mono text-xs" title={route.request_id}>
                      {planSummary(route.plan)}
                      {route.attempt > 1 ? <span className="ml-1 text-warn">#{route.attempt}</span> : null}
                    </td>
                    <td className="num text-xs">
                      {fmtInt(chosen.cached_tokens)}
                      <span className="text-ink-3"> / {fmtInt(route.prompt_tokens)}</span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </section>
      ) : null}
    </aside>
  );
}

function Waiting({ t }: { t: Messages }) {
  const k = t.cluster.decision;
  return (
    <div className="border-y-[1.5px] border-ink">
      <EmptyState compact title={k.waitingTitle} className="px-1">
        <p>{k.waitingBody}</p>
        <p>
          {k.waitingBefore}{" "}
          <Link to="/playground" className="text-ink underline">
            Playground
          </Link>
          {k.waitingAfter}
        </p>
        <Command>{"cargo run -p lev-bench -- --requests 200 --concurrency 8"}</Command>
      </EmptyState>
    </div>
  );
}

function Decision({ route, stages, t }: { route: RouteEvent; stages: StageEvent[]; t: Messages }) {
  const k = t.cluster.decision;
  const candidates = [...route.candidates].sort((a, b) => a.est_ttft_us - b.est_ttft_us);
  const max = Math.max(...candidates.map((c) => c.est_ttft_us), 1);
  const best = candidates[0];
  const chosen = candidates.find((c) => c.chosen);
  const chosenIsBest = !chosen || !best || chosen.est_ttft_us <= best.est_ttft_us;
  const prefillDone = stages.find((s) => s.request_id === route.request_id && s.stage === "prefill_done");
  const decodeReady = stages.find((s) => s.request_id === route.request_id && s.stage === "decode_ready");
  const facts: [string, ReactNode][] = [
    [k.facts.request, <span className="font-mono">{shortId(route.request_id, 13)}</span>],
    [k.facts.protocol, PROTOCOL_LABEL[route.protocol] ?? route.protocol],
    [k.facts.prompt, t.common.tokens(fmtInt(route.prompt_tokens))],
    [k.facts.policy, <span className="font-mono">{route.policy}</span>],
    [
      k.facts.plan,
      <span className="font-mono">
        {route.plan.kind} · {planSummary(route.plan)}
      </span>,
    ],
    [k.facts.time, `${fmtClock(route.at_ms, true)}${route.attempt > 1 ? k.attempt(route.attempt) : ""}${route.affinity ? k.affinity : ""}`],
  ];
  if (route.plan.kind === "disaggregated") {
    facts.push([
      k.facts.stages,
      prefillDone ? (
        <span className="inline-flex flex-wrap gap-x-1 text-xs">
          <span className="whitespace-nowrap">{k.handoff(fmtInt(prefillDone.transferred_tokens), fmtMs(prefillDone.duration_ms).replace(" ", "\u00a0"))}</span>
          {decodeReady ? <span className="whitespace-nowrap">· {k.decodeReady(fmtMs(decodeReady.duration_ms).replace(" ", "\u00a0"))}</span> : null}
        </span>
      ) : (
        <span className="text-ink-3">{k.stagesPending}</span>
      ),
    ]);
  }
  return (
    <div className="flex flex-col gap-3">
      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
        {facts.map(([label, value]) => (
          <div key={label} className="contents">
            <dt className="text-xs leading-5 text-ink-3">{label}</dt>
            <dd className={cn("min-w-0 text-ink", label === k.facts.stages ? "break-words" : "truncate")}>{value}</dd>
          </div>
        ))}
      </dl>

      <table className="table-3l">
        <caption className="sr-only">{k.scores}</caption>
        <thead>
          <tr>
            <th>{k.candidate}</th>
            <th className="num">
              {k.cached} <span className="unit">(tok)</span>
            </th>
            <th className="num">{k.seqs}</th>
            <th className="num">{k.kv}</th>
            <th className="w-[38%]">
              {k.ttft} <span className="unit">{k.sharedAxis}</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {candidates.map((candidate) => (
            <CandidateRow key={candidate.worker_id} candidate={candidate} max={max} prompt={route.prompt_tokens} t={t} />
          ))}
        </tbody>
        <tfoot aria-hidden>
          <tr>
            <td colSpan={4} />
            <td className="pt-0.5 pb-1.5">
              <div className="flex items-center gap-2 text-2xs text-ink-3 tabular-nums">
                <div className="flex flex-1 justify-between gap-2 border-t border-rule-2 pt-0.5 whitespace-nowrap">
                  <span>0</span>
                  <span>{fmtMs(max / 1000)}</span>
                </div>
                <span className="w-14" />
              </div>
            </td>
          </tr>
        </tfoot>
      </table>

      <p className="text-xs leading-relaxed text-ink-2">
        <span className="font-mono">{route.policy}</span>: {t.policyHelp[route.policy]}.
        {!chosenIsBest ? <span className="text-ink"> {k.notBest}</span> : null}
      </p>
      <p className="border-t border-rule pt-2 text-2xs leading-relaxed text-ink-3">
        {k.formula}
        {route.plan.kind === "disaggregated" ? k.scopePd : k.scopeUnified}
      </p>
    </div>
  );
}

function CandidateRow({ candidate, max, prompt, t }: { candidate: CandidateScore; max: number; prompt: number; t: Messages }) {
  const width = `${Math.max(1.5, (candidate.est_ttft_us / max) * 100)}%`;
  const hit = prompt > 0 ? candidate.cached_tokens / prompt : 0;
  return (
    <tr aria-selected={candidate.chosen} className={cn(candidate.chosen && "[&>td]:bg-route-wash")}>
      <td className="font-mono text-xs whitespace-nowrap">
        <span className="inline-flex items-center gap-1.5">
          <span aria-hidden className={cn("inline-block size-1.5", candidate.chosen ? "bg-route" : "bg-transparent")} />
          <span className={cn(candidate.chosen && "font-semibold text-route")}>{candidate.worker_id}</span>
        </span>
        {candidate.chosen ? <span className="sr-only"> ({t.cluster.decision.chosen})</span> : null}
      </td>
      <td className="num text-xs" title={t.cluster.decision.hitTitle(fmtPct(hit))}>
        {fmtInt(candidate.cached_tokens)}
      </td>
      <td className="num text-xs">{candidate.sequences}</td>
      <td className="num text-xs">{fmtPct(candidate.kv_usage, candidate.kv_usage < 0.1 ? 1 : 0)}</td>
      <td>
        <div className="flex items-center gap-2">
          <div className="relative h-2.5 flex-1">
            <div className={cn("absolute inset-y-0 left-0", candidate.chosen ? "bg-route" : "bg-ink-3/45")} style={{ width }} />
          </div>
          <span className={cn("w-14 text-right text-xs tabular-nums", candidate.chosen ? "text-route" : "text-ink-2")}>
            {fmtMs(candidate.est_ttft_us / 1000)}
          </span>
        </div>
      </td>
    </tr>
  );
}
