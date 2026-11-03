import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Link } from "react-router";
import { EFFORTS, type Effort } from "@/api/protocols";
import { useWorkers } from "@/api/queries";
import { PageHeader, SectionHeading } from "@/components/PageHeader";
import { EmptyState } from "@/components/States";
import { Button } from "@/components/ui/Button";
import { Field, Input, Textarea } from "@/components/ui/Field";
import { Segmented } from "@/components/ui/Segmented";
import { StatusMark } from "@/components/ui/StatusMark";
import { Switch } from "@/components/ui/Switch";
import { getT, useT, type Messages } from "@/i18n";
import { cn } from "@/lib/cn";
import { DEMO_TOOLS } from "@/lib/demoTools";
import { fmtInt, fmtMs, shortId } from "@/lib/format";
import { useSettings } from "@/state/settings";
import { runAgent, type AgentRun, type ModelStep, type ToolStep } from "./agent/runAgent";
import { Timeline } from "./agent/Timeline";
import { Refusal } from "./playground/OutputView";

function pretty(value: unknown): string {
  if (typeof value === "string") {
    try {
      return JSON.stringify(JSON.parse(value), null, 2);
    } catch {
      return value;
    }
  }
  return JSON.stringify(value, null, 2);
}

export function AgentStudioPage() {
  const t = useT();
  const k = t.agent;
  const [task, setTask] = useState(() => getT().agent.defaults.task);
  const [system, setSystem] = useState(() => getT().agent.defaults.system);
  const [enabled, setEnabled] = useState<string[]>(DEMO_TOOLS.map((tool) => tool.spec.name));
  const [thinking, setThinking] = useState(true);
  const [effort, setEffort] = useState<Effort>("high");
  const [maxTurns, setMaxTurns] = useState("6");
  const [run, setRun] = useState<AgentRun | null>(null);
  const [elapsedLive, setElapsedLive] = useState(0);
  const startedAt = useRef<number | null>(null);
  const controller = useRef<AbortController | null>(null);
  const workers = useWorkers(10_000);
  const { apiKey } = useSettings();

  const running = run?.status === "running";
  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(() => {
      if (startedAt.current !== null) setElapsedLive(performance.now() - startedAt.current);
    }, 100);
    return () => window.clearInterval(timer);
  }, [running]);
  useEffect(() => () => controller.current?.abort(), []);

  const turns = Number(maxTurns);
  const turnsValid = Number.isInteger(turns) && turns >= 1 && turns <= 20;
  const tools = DEMO_TOOLS.filter((tool) => enabled.includes(tool.spec.name));
  const backends = useMemo(() => new Set((workers.data?.workers ?? []).map((w) => w.backend)), [workers.data]);
  const simOnly = backends.size === 1 && backends.has("sim");
  const elapsed = run?.total ?? elapsedLive;

  async function start() {
    controller.current?.abort();
    const abort = new AbortController();
    controller.current = abort;
    startedAt.current = performance.now();
    setElapsedLive(0);
    await runAgent({ task, system, tools, thinking, effort, maxTurns: turns }, setRun, abort.signal);
  }

  return (
    <div className="flex flex-col gap-6">
      <PageHeader title={k.title} description={k.description} />
      <div className="grid items-start gap-x-8 gap-y-8 lg:grid-cols-[minmax(300px,360px)_minmax(0,1fr)]">
        <form
          className="flex flex-col gap-4 lg:sticky lg:top-5"
          onSubmit={(event) => {
            event.preventDefault();
            if (!running && task.trim() && turnsValid) void start();
          }}
        >
          <Field label={k.task} htmlFor="agent-task">
            <Textarea id="agent-task" rows={3} value={task} onChange={(event) => setTask(event.target.value)} />
          </Field>
          <Field label={k.system} htmlFor="agent-system">
            <Textarea id="agent-system" rows={2} value={system} onChange={(event) => setSystem(event.target.value)} />
          </Field>

          <fieldset className="flex flex-col gap-2 border-t border-rule pt-3">
            <legend className="cond mb-1 text-xs font-semibold text-ink-2">{k.tools}</legend>
            {DEMO_TOOLS.map((tool) => {
              const checked = enabled.includes(tool.spec.name);
              const [label, summary] = k.toolLabels[tool.spec.name] ?? [tool.spec.name, tool.spec.description];
              return (
                <label key={tool.spec.name} className="flex cursor-pointer items-start gap-2.5 text-sm">
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={(event) =>
                      setEnabled((list) =>
                        event.target.checked
                          ? DEMO_TOOLS.map((d) => d.spec.name).filter((name) => name === tool.spec.name || list.includes(name))
                          : list.filter((name) => name !== tool.spec.name),
                      )
                    }
                    className="mt-1 size-3.5 accent-[var(--ink)]"
                  />
                  <span className="min-w-0">
                    <span className="font-mono text-sm">{tool.spec.name}</span>
                    <span className="ml-2 text-ink-2">{label}</span>
                    <span className="block text-xs text-ink-3">{summary}</span>
                  </span>
                </label>
              );
            })}
            <p className="text-xs leading-relaxed text-ink-3">
              {k.toolsNote}
              {simOnly ? k.simNote : ""}
            </p>
          </fieldset>

          <div className="flex flex-wrap items-center justify-between gap-3 border-t border-rule pt-3">
            <Switch label={k.thinking} checked={thinking} onChange={setThinking} stateText={thinking ? t.common.on : t.common.off} />
            <Segmented<Effort>
              label={k.effort}
              size="sm"
              mono
              value={effort}
              disabled={!thinking}
              onChange={setEffort}
              options={EFFORTS.map((value) => ({ value, label: value }))}
            />
          </div>
          <Field label={k.maxTurns} htmlFor="agent-turns" error={turnsValid ? null : k.maxTurnsInvalid}>
            <Input
              id="agent-turns"
              inputMode="numeric"
              value={maxTurns}
              onChange={(event) => setMaxTurns(event.target.value)}
              className="w-24 font-mono"
              aria-invalid={!turnsValid}
            />
          </Field>
          <p className="text-xs text-ink-3">{apiKey ? k.keyUsed(apiKey.slice(0, 8)) : k.keyAnonymous}</p>

          <div className="flex items-center gap-2 border-t-[1.5px] border-ink pt-3">
            {running ? (
              <Button variant="danger" onClick={() => controller.current?.abort()}>
                {k.stop}
              </Button>
            ) : (
              <Button type="submit" variant="primary" disabled={!task.trim() || !turnsValid}>
                {k.run}
              </Button>
            )}
            {tools.length === 0 ? <span className="text-xs text-warn">{k.noTools}</span> : null}
          </div>
        </form>

        <div className="flex min-w-0 flex-col gap-6">
          {!run ? (
            <EmptyState title={k.emptyTitle}>
              <p>{k.emptyBody}</p>
            </EmptyState>
          ) : (
            <>
              <RunSummary run={run} elapsed={elapsed} t={t} />
              <section>
                <SectionHeading className="mb-2">{k.timing}</SectionHeading>
                <Timeline steps={run.steps} now={elapsed} />
                <p className="mt-1 text-2xs text-ink-3">{k.timingNote}</p>
              </section>
              <section className="flex flex-col gap-5">
                <SectionHeading>{k.steps}</SectionHeading>
                {run.steps.map((step, index) =>
                  step.kind === "model" ? (
                    <ModelCard key={index} step={step} final={run.status === "done" && index === run.steps.length - 1} t={t} />
                  ) : (
                    <ToolCard key={index} step={step} t={t} />
                  ),
                )}
              </section>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function RunSummary({ run, elapsed, t }: { run: AgentRun; elapsed: number; t: Messages }) {
  const k = t.agent;
  const models = run.steps.filter((s): s is ModelStep => s.kind === "model");
  const toolSteps = run.steps.filter((s): s is ToolStep => s.kind === "tool");
  const tokens = models.reduce((sum, s) => sum + (s.output.usage?.completion ?? 0), 0);
  const mark =
    run.status === "running"
      ? ({ state: "live", tone: "route", label: k.running } as const)
      : run.status === "done"
        ? ({ state: "live", tone: "ok", label: k.answered } as const)
        : run.status === "exhausted"
          ? ({ state: "stale", tone: "warn", label: k.exhausted } as const)
          : run.status === "aborted"
            ? ({ state: "idle", tone: "muted", label: k.stopped } as const)
            : ({ state: "down", tone: "bad", label: k.failed } as const);
  const facts: [string, ReactNode][] = [
    [k.state, <StatusMark {...mark} />],
    [k.total, fmtMs(elapsed)],
    [k.modelCalls, fmtInt(models.length)],
    [k.toolCalls, fmtInt(toolSteps.length)],
    [k.outputTokens, fmtInt(tokens)],
  ];
  return (
    <div className="flex flex-col gap-3">
      <dl className="grid grid-cols-3 gap-x-6 gap-y-2 border-y-[1.5px] border-ink py-2.5 text-sm sm:grid-cols-5">
        {facts.map(([label, value]) => (
          <div key={label}>
            <dt className="cond text-2xs font-semibold text-ink-3">{label}</dt>
            <dd className="tabular-nums">{value}</dd>
          </div>
        ))}
      </dl>
      {run.answer !== null ? (
        <div>
          <p className="cond mb-1 text-xs font-semibold text-ink-2">{k.finalAnswer}</p>
          <p className="text-md leading-relaxed whitespace-pre-wrap">{run.answer || t.common.empty}</p>
        </div>
      ) : null}
    </div>
  );
}

function StepHeader({ title, meta, children }: { title: string; meta: ReactNode; children?: ReactNode }) {
  return (
    <header className="mb-2 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1 border-b border-ink pb-1">
      <h3 className="cond text-md leading-6 font-bold">
        {title}
        {children}
      </h3>
      <span className="text-xs text-ink-3 tabular-nums">{meta}</span>
    </header>
  );
}

function ModelCard({ step, final, t }: { step: ModelStep; final: boolean; t: Messages }) {
  const k = t.agent;
  const duration = step.end === null ? null : step.end - step.start;
  const ttft = step.firstToken === null ? null : step.firstToken - step.start;
  return (
    <article>
      <StepHeader
        title={k.modelStep(step.turn)}
        meta={
          <>
            +{fmtMs(step.start)} · {k.firstToken} {fmtMs(ttft)} · {k.took} {duration === null ? "…" : fmtMs(duration)}
            {step.output.finishReason ? ` · ${step.output.finishReason}` : ""}
          </>
        }
      >
        {step.requestId ? (
          <Link
            to={`/requests?id=${encodeURIComponent(step.requestId)}`}
            className="ml-2 font-mono text-xs font-normal text-ink-2 underline decoration-rule-2 hover:text-ink"
          >
            {shortId(step.requestId)}
          </Link>
        ) : null}
      </StepHeader>
      <div className="flex flex-col gap-3 text-sm">
        {step.output.reasoning ? (
          <div>
            <p className="cond mb-0.5 text-2xs font-semibold text-ink-3">{k.reasoning}</p>
            <p
              className={cn(
                "leading-relaxed whitespace-pre-wrap text-ink-2",
                step.end === null && !step.output.content && step.output.toolCalls.length === 0 && "stream-caret",
              )}
            >
              {step.output.reasoning}
            </p>
          </div>
        ) : null}
        {step.output.toolCalls.map((call, index) => (
          <div key={`${call.id}-${index}`}>
            <p className="cond mb-0.5 text-2xs font-semibold text-ink-3">{k.toolCall}</p>
            <p className="font-mono text-sm font-semibold">
              {call.name}
              <span className="ml-2 text-2xs font-normal text-ink-3">{call.id}</span>
            </p>
            <pre className="mt-1 overflow-x-auto border border-rule bg-wash p-2 font-mono text-xs">{pretty(call.arguments || "{}")}</pre>
          </div>
        ))}
        {step.output.content ? (
          <div>
            <p className="cond mb-0.5 text-2xs font-semibold text-ink-3">{final ? k.answer : k.content}</p>
            <p className="leading-relaxed whitespace-pre-wrap">{step.output.content}</p>
          </div>
        ) : null}
        {step.error ? (
          <Refusal status={step.httpStatus} code={step.errorCode} message={step.error} retryAfter={step.retryAfter} endedAt={step.endedAt} t={t} />
        ) : null}
      </div>
    </article>
  );
}

function ToolCard({ step, t }: { step: ToolStep; t: Messages }) {
  const k = t.agent;
  const duration = step.end === null ? null : step.end - step.start;
  return (
    <article>
      <StepHeader title={k.toolStep(step.name)} meta={`+${fmtMs(step.start)} · ${k.toolMeta(duration === null ? "…" : fmtMs(duration))}`} />
      <div className="grid gap-3 md:grid-cols-2">
        <div className="min-w-0">
          <p className="cond mb-0.5 text-2xs font-semibold text-ink-3">{k.args}</p>
          <pre className="overflow-x-auto border border-rule bg-wash p-2 font-mono text-xs">{pretty(step.rawArguments || "{}")}</pre>
        </div>
        <div className="min-w-0">
          <p className={cn("cond mb-0.5 text-2xs font-semibold", step.error ? "text-bad" : "text-ink-3")}>{step.error ? k.resultError : k.result}</p>
          <pre className="overflow-x-auto border border-rule bg-wash p-2 font-mono text-xs">{pretty(step.result)}</pre>
        </div>
      </div>
    </article>
  );
}
