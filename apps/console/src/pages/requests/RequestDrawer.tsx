import { useMutation, useQuery } from "@tanstack/react-query";
import { RotateCcw } from "lucide-react";
import { useState, type ReactNode } from "react";
import { api } from "@/api/client";
import { useRequestDetail } from "@/api/queries";
import type { AuditEntry, RenderToken, ReplayResult } from "@/api/types";
import { PROTOCOL_LABEL } from "@/api/protocols";
import { ErrorState, Skeleton } from "@/components/States";
import { OutcomeTag } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Drawer } from "@/components/ui/Drawer";
import { StatusMark } from "@/components/ui/StatusMark";
import { useT, type Messages } from "@/i18n";
import { cn } from "@/lib/cn";
import { fmtDateTime, fmtInt, fmtMs, fmtPct, shortId } from "@/lib/format";
import { planSummary } from "@/lib/router";
import { TokenChip } from "../playground/TokenMap";
import { Waterfall } from "./Waterfall";

export function RequestDrawer({ id, onClose }: { id: string | null; onClose: () => void }) {
  const t = useT();
  const k = t.requests.drawer;
  const detail = useRequestDetail(id);
  const entry = detail.data?.request;
  return (
    <Drawer
      open={id !== null}
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
      title={id ? k.title(shortId(id, 13)) : k.title("")}
      description={
        entry ? (
          <>
            <span className="font-mono">{entry.response_id}</span> · {PROTOCOL_LABEL[entry.protocol]} · {k.tenantOf(entry.tenant)} ·{" "}
            {fmtDateTime(entry.received_at_ms)}
          </>
        ) : undefined
      }
    >
      {detail.error ? (
        <ErrorState error={detail.error} what={k.readError} />
      ) : !detail.data || !entry ? (
        <div className="flex flex-col gap-4">
          <Skeleton className="h-24 w-full" />
          <Skeleton className="h-40 w-full" />
        </div>
      ) : (
        <div className="flex flex-col gap-7">
          <Timing entry={entry} t={t} />
          <section>
            <SectionTitle title={k.waterfall} meta={k.waterfallMeta} />
            {entry.timeline && entry.timeline.length > 0 ? (
              <Waterfall stages={entry.timeline} e2e={entry.e2e_ms} />
            ) : (
              <p className="text-sm text-ink-3">{k.noTimeline}</p>
            )}
          </section>
          <Facts entry={entry} t={t} />
          <Replay entry={entry} t={t} />
          <Output ids={detail.data.output_token_ids} text={detail.data.output_text} t={t} />
          <section>
            <SectionTitle title={k.plan} meta={k.planMeta} />
            <pre className="overflow-auto border border-rule bg-wash p-3 font-mono text-xs leading-relaxed">
              {entry.plan ? JSON.stringify(entry.plan, null, 2) : k.planNull}
            </pre>
          </section>
          <section className="flex flex-col gap-3">
            <SectionTitle title={k.tokenIds} meta={k.tokenIdsMeta} />
            <TokenIds label="prompt" ids={detail.data.prompt_token_ids} t={t} />
            <TokenIds label="output" ids={detail.data.output_token_ids} t={t} />
          </section>
        </div>
      )}
    </Drawer>
  );
}

function SectionTitle({ title, meta }: { title: string; meta?: ReactNode }) {
  return (
    <header className="mb-2 flex flex-wrap items-baseline justify-between gap-x-3 gap-y-0.5 border-b-[1.5px] border-ink pb-1">
      <h3 className="cond text-md leading-6 font-bold">{title}</h3>
      {meta ? <span className="text-xs text-ink-3">{meta}</span> : null}
    </header>
  );
}

/** Timing diagram: received → first token → done. */
function Timing({ entry, t }: { entry: AuditEntry; t: Messages }) {
  const k = t.requests.drawer;
  const e2e = Math.max(entry.e2e_ms, 0.001);
  const ttft = entry.ttft_ms;
  const pct = (ms: number) => `${Math.min(100, (ms / e2e) * 100)}%`;
  return (
    <section>
      <SectionTitle title={k.timing} meta={k.timingMeta} />
      <div className="relative mt-6 mb-9 h-6">
        <div className="absolute inset-x-0 top-1/2 h-px bg-ink-3" />
        {ttft !== null ? (
          <>
            <div className="absolute inset-y-1 left-0 bg-ink" style={{ width: pct(ttft) }} />
            <div className="absolute inset-y-1 border border-ink bg-wash" style={{ left: pct(ttft), right: 0 }} />
          </>
        ) : (
          <div className="absolute inset-y-1 right-0 left-0 border border-ink-3" />
        )}
        <Marker at="0%" label={k.received} value="0" align="start" />
        {ttft !== null ? <Marker at={pct(ttft)} label={k.firstToken} value={fmtMs(ttft)} /> : null}
        <Marker at="100%" label={k.done} value={fmtMs(entry.e2e_ms)} align="end" />
      </div>
      <dl className="grid grid-cols-2 gap-x-6 gap-y-1.5 text-sm sm:grid-cols-4">
        <Fact label="TTFT" value={fmtMs(ttft)} />
        <Fact label="TPOT" value={fmtMs(entry.tpot_ms)} />
        <Fact label="E2E" value={fmtMs(entry.e2e_ms)} />
        <Fact label={k.attempts} value={fmtInt(entry.attempts)} />
      </dl>
    </section>
  );
}

function Marker({ at, label, value, align }: { at: string; label: string; value: string; align?: "start" | "end" }) {
  return (
    <div className="absolute top-0 h-full" style={{ left: at }}>
      <div className="absolute -top-1 bottom-[-6px] w-px bg-ink" />
      <div
        className={cn(
          "absolute top-[calc(100%+8px)] text-2xs whitespace-nowrap",
          align === "end" ? "right-0 text-right" : align === "start" ? "left-0 text-left" : "-translate-x-1/2 text-center",
        )}
      >
        <div className="text-ink-3">{label}</div>
        <div className="font-mono text-ink tabular-nums">{value}</div>
      </div>
    </div>
  );
}

function Fact({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="min-w-0">
      <dt className="cond text-2xs font-semibold text-ink-3">{label}</dt>
      <dd className="truncate tabular-nums">{value}</dd>
    </div>
  );
}

function Facts({ entry, t }: { entry: AuditEntry; t: Messages }) {
  const k = t.requests.drawer;
  const accept = entry.spec_draft_tokens > 0 ? entry.spec_accepted_tokens / entry.spec_draft_tokens : null;
  return (
    <section>
      <SectionTitle title={k.routing} />
      <dl className="grid grid-cols-2 gap-x-6 gap-y-3 text-sm sm:grid-cols-4">
        <Fact label={k.outcome} value={<OutcomeTag outcome={entry.outcome} />} />
        <Fact label={k.finish} value={<span className="font-mono">{entry.finish_reason ?? "—"}</span>} />
        <Fact label={k.policy} value={<span className="font-mono">{entry.policy ?? "—"}</span>} />
        <Fact label={k.worker} value={<span className="font-mono">{planSummary(entry.plan)}</span>} />
        <Fact label={k.promptTokens} value={fmtInt(entry.prompt_tokens)} />
        <Fact
          label={k.cached}
          value={`${fmtInt(entry.cached_tokens)} (${fmtPct(entry.prompt_tokens > 0 ? entry.cached_tokens / entry.prompt_tokens : null)})`}
        />
        <Fact label={k.output} value={fmtInt(entry.completion_tokens)} />
        <Fact
          label={k.dspark}
          value={accept === null ? "—" : `${fmtInt(entry.spec_accepted_tokens)} / ${fmtInt(entry.spec_draft_tokens)} (${fmtPct(accept)})`}
        />
        <Fact label={k.seed} value={<span className="font-mono">{entry.seed}</span>} />
        <Fact label={k.model} value={entry.model} />
        <Fact label={k.protocol} value={PROTOCOL_LABEL[entry.protocol]} />
        <Fact label={k.tenant} value={entry.tenant} />
      </dl>
      {entry.error ? (
        <p role="alert" className="mt-3 border-t border-rule pt-2 text-sm text-bad">
          {k.error(entry.error)}
        </p>
      ) : null}
    </section>
  );
}

/** Output as text, or as token chips through /v1/leviathan/detokenize. */
function Output({ ids, text, t }: { ids: number[]; text: string | null; t: Messages }) {
  const k = t.requests.drawer;
  const [asTokens, setAsTokens] = useState(false);
  const tokens = useQuery({
    queryKey: ["detokenize", ids.length, ids.slice(0, 8).join(","), ids.slice(-8).join(",")],
    queryFn: ({ signal }) => api.detokenize(ids, signal),
    enabled: asTokens && ids.length > 0,
    staleTime: Infinity,
  });
  return (
    <section>
      <SectionTitle
        title={k.outputText}
        meta={
          <span className="inline-flex items-center gap-3">
            {k.outputMeta(fmtInt(ids.length))}
            {ids.length > 0 ? (
              <button type="button" className="text-ink-2 underline decoration-rule-2 hover:text-ink" onClick={() => setAsTokens((v) => !v)}>
                {asTokens ? k.hideTokens : k.showTokens}
              </button>
            ) : null}
          </span>
        }
      />
      {ids.length === 0 || (!asTokens && !text) ? (
        <p className="text-sm text-ink-3">{k.noOutput}</p>
      ) : !asTokens ? (
        <pre className="max-h-80 overflow-auto border border-rule bg-wash p-3 font-mono text-xs leading-relaxed whitespace-pre-wrap">{text}</pre>
      ) : tokens.error ? (
        <ErrorState error={tokens.error} what="/v1/leviathan/detokenize" compact />
      ) : !tokens.data ? (
        <p className="text-sm text-ink-3">{k.tokensLoading}</p>
      ) : (
        <div className="flex max-h-80 flex-wrap gap-[3px] overflow-auto border border-rule p-2">
          {tokens.data.tokens.map((token, index) => (
            <TokenChip key={index} token={token} showId={false} index={index} />
          ))}
        </div>
      )}
    </section>
  );
}

function TokenIds({ label, ids, t }: { label: string; ids: number[]; t: Messages }) {
  return (
    <details>
      <summary className="cursor-pointer text-sm text-ink-2">
        <span className="font-mono">{label}</span> · {t.requests.drawer.count(fmtInt(ids.length))}
      </summary>
      <p className="mt-2 max-h-48 overflow-auto border border-rule bg-wash p-2.5 font-mono text-2xs leading-relaxed break-words text-ink-2">
        {ids.length > 0 ? ids.join(" ") : t.common.empty}
      </p>
    </details>
  );
}

// ------------------------------------------------------------------ replay

function Replay({ entry, t }: { entry: AuditEntry; t: Messages }) {
  const k = t.requests.replay;
  const replay = useMutation({ mutationFn: () => api.replay(entry.request_id) });
  const result = replay.data;
  return (
    <section>
      <SectionTitle title={k.title} meta={k.meta} />
      <div className="flex flex-wrap items-center gap-3">
        <Button
          variant="primary"
          icon={<RotateCcw className="size-3.5" />}
          busy={replay.isPending}
          onClick={() => replay.mutate()}
          disabled={entry.outcome === "rejected"}
        >
          {result ? k.again : k.run}
        </Button>
        <span className="text-xs text-ink-3">{entry.outcome === "rejected" ? k.notReplayable : k.info(entry.seed, entry.worker_id ?? "—")}</span>
      </div>
      {replay.error ? <ErrorState error={replay.error} what={k.failed} compact /> : null}
      {result ? <ReplayVerdict result={result} t={t} /> : null}
    </section>
  );
}

function TokenRow({ label, tokens, from, to, mark }: { label: string; tokens: RenderToken[]; from: number; to: number; mark: number | null }) {
  return (
    <div className="grid grid-cols-[6rem_minmax(0,1fr)] items-start gap-3">
      <span className="pt-0.5 text-xs text-ink-2">{label}</span>
      <div className="flex flex-wrap gap-[3px]">
        {tokens.slice(from, to).map((token, offset) => (
          <TokenChip key={from + offset} token={token} showId index={from + offset} highlight={from + offset === mark} />
        ))}
        {to > tokens.length && mark !== null && mark >= tokens.length ? <span className="self-center font-mono text-2xs text-bad">EOS</span> : null}
      </div>
    </div>
  );
}

function commonPrefix(a: string, b: string): number {
  const n = Math.min(a.length, b.length);
  let i = 0;
  while (i < n && a[i] === b[i]) i += 1;
  return i;
}

function ReplayVerdict({ result, t }: { result: ReplayResult; t: Messages }) {
  const k = t.requests.replay;
  const original = result.original_tokens_view ?? null;
  const replayed = result.replay_tokens_view ?? null;
  const at = result.first_divergence;
  const from = at === null ? 0 : Math.max(0, at - 8);
  const to = at === null ? 0 : at + 12;
  const split = commonPrefix(result.original_text ?? "", result.replay_text ?? "");
  return (
    <div className="mt-4 flex flex-col gap-4">
      <div className={cn("flex flex-wrap items-center gap-3 border-y-[1.5px] py-2.5", result.identical ? "border-ok" : "border-bad")}>
        <StatusMark state={result.identical ? "live" : "down"} tone={result.identical ? "ok" : "bad"} width={24} />
        <p className="text-sm">
          {result.identical ? (
            <>
              <strong className="font-semibold text-ok">{k.identicalStrong}</strong>
              {k.identical(fmtInt(result.original_tokens))}
            </>
          ) : (
            <>
              <strong className="font-semibold text-bad">{k.divergedStrong(at ?? 0)}</strong>
              {k.diverged(fmtInt(result.original_tokens), fmtInt(result.replay_tokens))}
            </>
          )}
        </p>
      </div>
      <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-4">
        <Fact label={k.originalWorker} value={<span className="font-mono">{result.original_worker ?? "—"}</span>} />
        <Fact label={k.replayWorker} value={<span className="font-mono">{result.replay_worker}</span>} />
        <Fact label={k.seed} value={<span className="font-mono">{result.seed}</span>} />
        <Fact label={k.replayFinish} value={<span className="font-mono">{result.finish_reason?.replace(/^FINISH_REASON_/, "").toLowerCase() ?? "—"}</span>} />
      </dl>
      <p className="text-xs text-ink-3">
        {k.replayId} <span className="font-mono">{result.replay_id}</span>
        {result.original_worker && result.original_worker !== result.replay_worker ? k.otherWorker : ""}
      </p>

      {result.identical && original ? (
        <details>
          <summary className="cursor-pointer text-sm text-ink-2">{k.sameTokens(fmtInt(original.length))}</summary>
          <div className="mt-2 flex max-h-64 flex-wrap gap-[3px] overflow-auto border border-rule p-2">
            {original.map((token, index) => (
              <TokenChip key={index} token={token} showId={false} index={index} />
            ))}
          </div>
        </details>
      ) : null}

      {!result.identical && at !== null ? (
        original && replayed ? (
          <div className="flex flex-col gap-2">
            <p className="text-xs text-ink-2">{k.around(at)}</p>
            <TokenRow label={k.original} tokens={original} from={from} to={to} mark={at} />
            <TokenRow label={k.replayed} tokens={replayed} from={from} to={to} mark={at} />
          </div>
        ) : (
          <div className="grid gap-3 md:grid-cols-2">
            <DiffText label={k.original} text={result.original_text ?? ""} split={split} t={t} />
            <DiffText label={k.replayed} text={result.replay_text ?? ""} split={split} t={t} />
          </div>
        )
      ) : null}
    </div>
  );
}

function DiffText({ label, text, split, t }: { label: string; text: string; split: number; t: Messages }) {
  return (
    <div className="min-w-0">
      <p className="mb-1 text-xs text-ink-2">{label}</p>
      <pre className="max-h-64 overflow-auto border border-rule bg-wash p-2.5 font-mono text-xs leading-relaxed whitespace-pre-wrap">
        <span className="text-ink-3">{text.slice(0, split)}</span>
        <mark className="bg-bad-wash text-bad">{text.slice(split) || t.common.empty}</mark>
      </pre>
    </div>
  );
}
