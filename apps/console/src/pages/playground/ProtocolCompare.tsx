import { useState } from "react";
import { api } from "@/api/client";
import { buildBody, PROTOCOL_LABEL, PROTOCOL_PATH, type Compose } from "@/api/protocols";
import { PROTOCOLS, type ProtocolName, type RenderResult } from "@/api/types";
import { EmptyState, ErrorState } from "@/components/States";
import { Button } from "@/components/ui/Button";
import { StatusMark } from "@/components/ui/StatusMark";
import { useT } from "@/i18n";
import { cn } from "@/lib/cn";
import { fmtInt } from "@/lib/format";
import { TokenChip } from "./TokenMap";

type Results = Record<ProtocolName, RenderResult>;

function firstMismatch(a: number[], b: number[]): number | null {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i += 1) if (a[i] !== b[i]) return i;
  return a.length === b.length ? null : n;
}

/**
 * Render one conversation through all three protocols and compare the token
 * IDs. Identical IDs mean identical block hashes, so the prefix cache is
 * shared across protocols (docs/illustrated/02).
 */
export function ProtocolCompare({ compose }: { compose: Compose | null }) {
  const t = useT();
  const k = t.playground.compare;
  const [results, setResults] = useState<Results | null>(null);
  const [bodies, setBodies] = useState<Record<ProtocolName, Record<string, unknown>> | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);

  async function run() {
    if (!compose) return;
    const base: Compose = { ...compose, schema: null, seed: null };
    const built = Object.fromEntries(PROTOCOLS.map((p) => [p, buildBody(p, base)])) as Record<ProtocolName, Record<string, unknown>>;
    setLoading(true);
    setError(null);
    try {
      const rendered = await Promise.all(PROTOCOLS.map((p) => api.render(p, built[p])));
      setResults(Object.fromEntries(PROTOCOLS.map((p, i) => [p, rendered[i]])) as Results);
      setBodies(built);
    } catch (reason) {
      setError(reason);
    } finally {
      setLoading(false);
    }
  }

  const reference = results ? results.chat_completions.tokens.map((t) => t.id) : [];
  const mismatches = results
    ? (Object.fromEntries(PROTOCOLS.map((p) => [p, firstMismatch(reference, results[p].tokens.map((t) => t.id))])) as Record<ProtocolName, number | null>)
    : null;
  const identical = mismatches ? PROTOCOLS.every((p) => mismatches[p] === null) : false;
  const hashesIdentical = results
    ? PROTOCOLS.every((p) => results[p].block_hashes.join() === results.chat_completions.block_hashes.join())
    : false;
  const divergence = mismatches ? Math.min(...PROTOCOLS.map((p) => mismatches[p] ?? Infinity)) : Infinity;

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-wrap items-center gap-3">
        <Button variant="primary" onClick={run} busy={loading} disabled={!compose}>
          {k.run}
        </Button>
        <p className="text-xs text-ink-3">
          {k.explain}
          {compose?.schema ? k.schemaExcluded : ""}
        </p>
      </div>

      {error ? <ErrorState error={error} what={k.failed} /> : null}

      {!results && !error ? (
        <EmptyState title={k.emptyTitle}>
          <p>{k.emptyBody}</p>
        </EmptyState>
      ) : null}

      {results && mismatches ? (
        <>
          <div className={cn("flex items-baseline gap-3 border-y-[1.5px] py-2.5", identical ? "border-ok" : "border-bad")}>
            <StatusMark state={identical ? "live" : "down"} tone={identical ? "ok" : "bad"} width={24} className="shrink-0 self-center" />
            <p className="min-w-0 flex-1 text-sm">
              {identical ? (
                <>
                  {k.identicalLead}
                  <strong className="font-semibold">{k.identicalStrong(fmtInt(reference.length))}</strong>
                  {hashesIdentical && results.chat_completions.block_hashes.length > 0
                    ? k.identicalHashes(fmtInt(results.chat_completions.block_hashes.length))
                    : ""}
                  {k.identicalTail}
                </>
              ) : (
                k.differs(divergence)
              )}
            </p>
          </div>

          <table className="table-3l">
            <thead>
              <tr>
                <th>{k.protocol}</th>
                <th>{k.endpoint}</th>
                <th className="num">{k.promptTokens}</th>
                <th className="num">{k.fullBlocks}</th>
                <th>{k.lastHash}</th>
                <th>{k.vsChat}</th>
              </tr>
            </thead>
            <tbody>
              {PROTOCOLS.map((p) => {
                const r = results[p];
                const m = mismatches[p];
                return (
                  <tr key={p}>
                    <td className="font-semibold whitespace-nowrap">{PROTOCOL_LABEL[p]}</td>
                    <td className="font-mono text-xs text-ink-2">{PROTOCOL_PATH[p]}</td>
                    <td className="num">{fmtInt(r.prompt_tokens)}</td>
                    <td className="num">{fmtInt(r.block_hashes.length)}</td>
                    <td className="font-mono text-xs" title={r.block_hashes.length === 0 ? k.notFull : undefined}>
                      {r.block_hashes[r.block_hashes.length - 1] ?? "—"}
                    </td>
                    <td>
                      {p === "chat_completions" ? (
                        <span className="text-xs text-ink-3">{k.baseline}</span>
                      ) : m === null ? (
                        <StatusMark state="live" tone="ok" label={k.same} />
                      ) : (
                        <StatusMark state="down" tone="bad" label={k.differsAt(m)} />
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          <section className="flex flex-col gap-2">
            <h3 className="cond text-md leading-6 font-bold">
              {identical ? k.sequence : k.near}
              <span className="ml-2 text-xs font-normal text-ink-3">{k.controlNote}</span>
            </h3>
            {identical ? (
              <div className="flex flex-wrap gap-[3px]">
                {results.chat_completions.tokens.map((token, index) => (
                  <TokenChip key={index} token={token} showId index={index} />
                ))}
              </div>
            ) : (
              <div className="flex flex-col gap-3">
                {PROTOCOLS.map((p) => {
                  const tokens = results[p].tokens;
                  const from = Math.max(0, divergence - 6);
                  return (
                    <div key={p} className="grid grid-cols-[10rem_minmax(0,1fr)] items-start gap-3">
                      <span className="text-xs text-ink-2">{PROTOCOL_LABEL[p]}</span>
                      <div className="flex flex-wrap gap-[3px]">
                        {tokens.slice(from, divergence + 10).map((token, offset) => (
                          <TokenChip key={offset} token={token} showId index={from + offset} highlight={from + offset === divergence} />
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </section>

          {bodies ? (
            <details>
              <summary className="cursor-pointer text-sm text-ink-2">{k.bodies}</summary>
              <div className="mt-3 grid gap-3 lg:grid-cols-3">
                {PROTOCOLS.map((p) => (
                  <div key={p} className="min-w-0">
                    <p className="mb-1 font-mono text-2xs text-ink-3">POST {PROTOCOL_PATH[p]}</p>
                    <pre className="max-h-80 overflow-auto border border-rule bg-wash p-2.5 font-mono text-[0.6875rem] leading-relaxed">
                      {JSON.stringify(bodies[p], null, 2)}
                    </pre>
                  </div>
                ))}
              </div>
            </details>
          ) : null}
        </>
      ) : null}
    </div>
  );
}
