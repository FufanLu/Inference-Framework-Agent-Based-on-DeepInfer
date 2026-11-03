import type { ReactNode } from "react";
import { Link } from "react-router";
import type { ParsedOutput } from "@/api/protocols";
import type { ProtocolName } from "@/api/types";
import { EmptyState } from "@/components/States";
import { StatusMark } from "@/components/ui/StatusMark";
import { useT, type Messages } from "@/i18n";
import { cn } from "@/lib/cn";
import { fmtInt, fmtMs, shortId } from "@/lib/format";
import { useNow } from "@/lib/useNow";

export interface Run {
  protocol: ProtocolName;
  body: Record<string, unknown>;
  startedAt: number;
  /** Wall clock when the response ended, for Retry-After countdowns. */
  endedAt: number | null;
  status: "streaming" | "done" | "error" | "aborted";
  output: ParsedOutput;
  httpStatus: number | null;
  requestId: string | null;
  headersMs: number | null;
  totalMs: number | null;
  /** Refused before streaming, or a transport failure. */
  error: string | null;
  errorCode: string | null;
  retryAfter: number | null;
}

function prettyJson(text: string): { text: string; valid: boolean } {
  try {
    return { text: JSON.stringify(JSON.parse(text), null, 2), valid: true };
  } catch {
    return { text, valid: false };
  }
}

function Block({ label, meta, children, tone }: { label: string; meta?: string; children: ReactNode; tone?: "muted" }) {
  return (
    <section className="border-t-[1.5px] border-ink pt-2">
      <header className="mb-1.5 flex items-baseline justify-between gap-3">
        <h3 className="cond text-xs font-semibold text-ink-2">{label}</h3>
        {meta ? <span className="font-mono text-2xs text-ink-3">{meta}</span> : null}
      </header>
      <div className={cn("text-sm leading-relaxed", tone === "muted" ? "text-ink-2" : "text-ink")}>{children}</div>
    </section>
  );
}

/**
 * Why the gateway refused a request, with the recovery: 401 (bad or missing
 * API key), 429 (rate limit with Retry-After, or exhausted quota), 503.
 */
export function Refusal({
  status,
  code,
  message,
  retryAfter,
  endedAt,
  t,
}: {
  status: number | null;
  code: string | null;
  message: string;
  retryAfter: number | null;
  endedAt: number | null;
  t: Messages;
}) {
  const k = t.playground.output;
  const now = useNow(1000);
  const quota = code === "insufficient_quota";
  const limited = status === 429;
  const left = retryAfter !== null && endedAt !== null ? Math.max(0, Math.ceil(retryAfter - (now - endedAt) / 1000)) : null;
  const title = limited ? (quota ? k.quotaTitle : k.rateTitle) : k.failedTitle;
  return (
    <div role="alert" className={cn("border-t-[1.5px] pt-2 text-sm", limited ? "border-warn" : "border-bad")}>
      <p className={cn("font-semibold", limited ? "text-warn" : "text-bad")}>
        {title}
        {status ? ` · HTTP ${status}` : ""}
        {code ? <code className="ml-2 font-mono text-xs font-normal">{code}</code> : null}
      </p>
      <p className="mt-1 text-ink-2">{message}</p>
      {limited ? <p className="mt-1 text-ink-2">{quota ? k.quotaBody : k.rateBody}</p> : null}
      {limited && retryAfter !== null ? (
        <p className="mt-1 text-sm font-semibold text-ink tabular-nums" aria-live="polite">
          {left !== null && left > 0 ? k.retryIn(left) : k.retryNow}
        </p>
      ) : null}
      {status === 401 ? <p className="mt-1 text-ink-2">{k.hint401}</p> : null}
      {status === 503 ? <p className="mt-1 text-ink-2">{k.hint503}</p> : null}
    </div>
  );
}

export function OutputView({ run, expectJson }: { run: Run | null; expectJson: boolean }) {
  const t = useT();
  const k = t.playground.output;
  if (!run) {
    return (
      <EmptyState title={k.emptyTitle}>
        <p>{k.emptyBody}</p>
      </EmptyState>
    );
  }
  const { output } = run;
  const streaming = run.status === "streaming";
  const json = expectJson && output.content ? prettyJson(output.content) : null;
  const stateMark =
    run.status === "streaming"
      ? ({ state: "live", tone: "route", label: k.receiving } as const)
      : run.status === "done" && !output.error
        ? ({ state: "live", tone: "ok", label: k.done } as const)
        : run.status === "aborted"
          ? ({ state: "idle", tone: "muted", label: k.stopped } as const)
          : ({ state: "down", tone: "bad", label: k.failed } as const);

  const facts: [string, ReactNode][] = [
    [k.state, <StatusMark {...stateMark} />],
    [k.http, run.httpStatus ?? "—"],
    [k.firstToken, fmtMs(output.firstTokenMs)],
    [k.total, streaming ? "…" : fmtMs(run.totalMs)],
    [k.frames, fmtInt(output.frames)],
    [k.finish, <span className="font-mono">{output.finishReason ?? "—"}</span>],
  ];

  return (
    <div className="flex flex-col gap-5">
      <dl className="grid grid-cols-3 gap-x-6 gap-y-2 text-sm sm:grid-cols-6">
        {facts.map(([label, value]) => (
          <div key={label} className="min-w-0">
            <dt className="cond text-2xs font-semibold text-ink-3">{label}</dt>
            <dd className="truncate tabular-nums">{value}</dd>
          </div>
        ))}
      </dl>
      <p className="-mt-2 text-xs text-ink-3">
        {k.clientTiming}
        {run.requestId ? (
          <>
            {" "}
            {k.requestId}{" "}
            <Link className="font-mono text-ink underline decoration-rule-2" to={`/requests?id=${encodeURIComponent(run.requestId)}`}>
              {shortId(run.requestId, 13)}
            </Link>
            {k.requestIdAfter}
          </>
        ) : null}
      </p>

      {run.error ? (
        <Refusal status={run.httpStatus} code={run.errorCode} message={run.error} retryAfter={run.retryAfter} endedAt={run.endedAt} t={t} />
      ) : null}

      {output.reasoning ? (
        <Block label={k.reasoning} meta={run.protocol === "messages" ? "thinking" : run.protocol === "responses" ? "reasoning_text" : "reasoning_content"} tone="muted">
          <p className={cn("whitespace-pre-wrap", streaming && !output.content && output.toolCalls.length === 0 && "stream-caret")}>{output.reasoning}</p>
        </Block>
      ) : null}

      {output.content || (streaming && !output.reasoning && output.toolCalls.length === 0) ? (
        <Block label={k.content} meta={json ? (json.valid ? k.validJson : k.invalidJson) : undefined}>
          {json && json.valid ? (
            <pre className="overflow-x-auto font-mono text-xs leading-relaxed">{json.text}</pre>
          ) : (
            <p className={cn("whitespace-pre-wrap", streaming && "stream-caret")}>{output.content}</p>
          )}
        </Block>
      ) : null}

      {output.toolCalls.length > 0 ? (
        <Block label={k.toolCalls} meta={String(output.toolCalls.length)}>
          <div className="flex flex-col gap-3">
            {output.toolCalls.map((call, index) => (
              <div key={`${call.id}-${index}`} className="border-l border-ink pl-3">
                <p className="font-mono text-sm font-semibold">
                  {call.name || "…"}
                  <span className="ml-2 text-2xs font-normal text-ink-3">{call.id}</span>
                </p>
                <pre className="mt-1 overflow-x-auto font-mono text-xs leading-relaxed text-ink-2">{prettyJson(call.arguments || "{}").text}</pre>
              </div>
            ))}
          </div>
        </Block>
      ) : null}

      {output.error ? (
        <Block label={k.streamError}>
          <p className="text-bad">{output.error}</p>
        </Block>
      ) : null}

      {output.usage ? (
        <Block label={k.usage} meta="usage">
          <dl className="grid grid-cols-2 gap-x-6 gap-y-1 tabular-nums sm:grid-cols-4">
            <div>
              <dt className="text-xs text-ink-3">{k.usagePrompt}</dt>
              <dd>{fmtInt(output.usage.prompt)}</dd>
            </div>
            <div>
              <dt className="text-xs text-ink-3">{k.usageCached}</dt>
              <dd>{fmtInt(output.usage.cached)}</dd>
            </div>
            <div>
              <dt className="text-xs text-ink-3">{k.usageOutput}</dt>
              <dd>{fmtInt(output.usage.completion)}</dd>
            </div>
            <div>
              <dt className="text-xs text-ink-3">{k.usageReasoning}</dt>
              <dd>{output.usage.reasoning === null ? "—" : fmtInt(output.usage.reasoning)}</dd>
            </div>
          </dl>
        </Block>
      ) : null}
    </div>
  );
}
