import { useEffect, useState, type ReactNode } from "react";
import { api } from "@/api/client";
import { buildBody, type Compose } from "@/api/protocols";
import type { ProtocolName, RenderResult } from "@/api/types";
import { EmptyState, ErrorState, Skeleton } from "@/components/States";
import { Button } from "@/components/ui/Button";
import { Switch } from "@/components/ui/Switch";
import { useT } from "@/i18n";
import { fmtInt, fmtSpan } from "@/lib/format";
import { TokenMap } from "./TokenMap";

interface InspectorProps {
  compose: Compose | null;
  protocol: ProtocolName;
}

/** POST /v1/leviathan/render: exactly what the model will see. */
export function Inspector({ compose, protocol }: InspectorProps) {
  const t = useT();
  const k = t.playground.inspector;
  const [result, setResult] = useState<RenderResult | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const [showId, setShowId] = useState(true);
  const [showPrompt, setShowPrompt] = useState(false);
  const key = compose ? JSON.stringify([protocol, buildBody(protocol, compose)]) : null;

  // Re-render the prompt whenever the request changes (debounced).
  useEffect(() => {
    if (!compose || key === null) return;
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      setLoading(true);
      api
        .render(protocol, buildBody(protocol, compose), controller.signal)
        .then((value) => {
          setResult(value);
          setError(null);
        })
        .catch((reason: unknown) => {
          if (!controller.signal.aborted) setError(reason);
        })
        .finally(() => {
          if (!controller.signal.aborted) setLoading(false);
        });
    }, 350);
    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
    // `key` captures compose + protocol.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  if (!compose) {
    return (
      <EmptyState title={k.incompleteTitle}>
        <p>{k.incompleteBody}</p>
      </EmptyState>
    );
  }
  if (error && !result) return <ErrorState error={error} what={k.renderFailed} />;
  if (!result) {
    return (
      <div className="flex flex-col gap-3">
        <Skeleton className="h-16 w-full" />
        <Skeleton className="h-48 w-full" />
      </div>
    );
  }

  const full = result.block_hashes.length;
  const tail = result.prompt_tokens - full * result.block_size;
  const specials = result.tokens.filter((token) => token.special).length;
  const facts: [string, ReactNode][] = [
    [k.protocol, <span className="font-mono">{result.protocol}</span>],
    [k.promptTokens, fmtInt(result.prompt_tokens)],
    [k.controlTokens, fmtInt(specials)],
    [k.blockSize, fmtInt(result.block_size)],
    [k.fullBlocksLabel, k.fullBlocks(fmtInt(full), fmtInt(tail))],
    [k.maxTokens, fmtInt(result.max_tokens)],
    [k.context, fmtInt(result.context_limit)],
    [k.seed, <span className="font-mono">{result.seed}</span>],
    [k.thinking, result.thinking_mode ? k.thinkingOn : k.thinkingOff],
    [k.constraint, <span className="font-mono">{result.constraint ? result.constraint.kind : t.common.none}</span>],
  ];

  return (
    <div className="flex flex-col gap-4">
      <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm md:grid-cols-4 2xl:grid-cols-5">
        {facts.map(([label, value]) => (
          <div key={label} className="min-w-0">
            <dt className="cond text-2xs font-semibold text-ink-3">{label}</dt>
            <dd className={label === k.seed ? "break-all tabular-nums" : "break-words tabular-nums"}>{value}</dd>
          </div>
        ))}
      </dl>
      <div className="-mt-1 flex flex-col gap-1 text-xs text-ink-3">
        <p>
          {compose.seed === null ? k.seedNote : ""}
          {k.blockNote}{" "}
          <span className="font-mono">
            {Object.entries(result.hints)
              .map(([name, value]) => `${name}=${value}`)
              .join(" · ")}
          </span>
        </p>
        {result.timeline && result.timeline.length > 0 ? (
          <p>
            {k.gatewayTiming}{" "}
            <span className="tabular-nums">{result.timeline.map((stage) => `${stage.name} ${fmtSpan(stage.duration_ms)}`).join(" · ")}</span>
          </p>
        ) : null}
      </div>

      <div className="flex flex-wrap items-center gap-4">
        <Switch label={k.showIds} checked={showId} onChange={setShowId} />
        <Button size="sm" variant="ghost" onClick={() => setShowPrompt((v) => !v)} aria-expanded={showPrompt}>
          {showPrompt ? k.hidePrompt : k.showPrompt}
        </Button>
        {loading ? <span className="text-xs text-ink-3">{k.rendering}</span> : null}
        {error ? <span className="text-xs text-bad">{k.stale}</span> : null}
        <span className="ml-auto flex items-center gap-2 text-2xs text-ink-3">
          <span className="inline-block h-3 w-5 rounded-xs bg-ink" /> {k.legend}
        </span>
      </div>

      {showPrompt ? (
        <pre className="max-h-72 overflow-auto border border-rule bg-wash p-3 font-mono text-xs leading-relaxed whitespace-pre-wrap">{result.prompt}</pre>
      ) : null}

      <TokenMap tokens={result.tokens} blockSize={result.block_size} hashes={result.block_hashes} showId={showId} />

      {result.constraint?.schema ? (
        <details className="text-xs">
          <summary className="cursor-pointer text-ink-2">{k.schema}</summary>
          <pre className="mt-2 max-h-60 overflow-auto border border-rule bg-wash p-3 font-mono">{result.constraint.schema}</pre>
        </details>
      ) : null}
    </div>
  );
}
