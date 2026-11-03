import type { RenderToken } from "@/api/types";
import { useT } from "@/i18n";
import { cn } from "@/lib/cn";
import { visibleText } from "@/lib/tokens";

export function TokenChip({ token, showId, index, highlight }: { token: RenderToken; showId: boolean; index: number; highlight?: boolean }) {
  const t = useT();
  return (
    <span
      title={`${t.playground.inspector.chipTitle(index, token.id, token.special)}\n${JSON.stringify(token.text)}`}
      className={cn(
        "inline-flex min-w-[1.25rem] flex-col items-center rounded-xs border px-1 py-px font-mono leading-tight",
        token.special ? "border-ink bg-ink text-paper" : "border-rule-2 bg-sheet text-ink",
        highlight && "border-bad ring-1 ring-bad",
      )}
    >
      <span className="text-xs whitespace-pre">{visibleText(token.text)}</span>
      {showId ? <span className={cn("text-[0.625rem] tabular-nums", token.special ? "text-paper/75" : "text-ink-3")}>{token.id}</span> : null}
    </span>
  );
}

/**
 * The prompt as a memory map: one row per KV block of `blockSize` tokens,
 * addressed by its chained block hash. The trailing partial block has no hash
 * and is never cached.
 */
export function TokenMap({
  tokens,
  blockSize,
  hashes,
  showId,
}: {
  tokens: RenderToken[];
  blockSize: number;
  hashes: string[];
  showId: boolean;
}) {
  const t = useT();
  const k = t.playground.inspector;
  const size = Math.max(1, blockSize);
  const rows: { index: number; tokens: RenderToken[]; hash: string | null; start: number }[] = [];
  for (let start = 0; start < tokens.length; start += size) {
    const index = start / size;
    const slice = tokens.slice(start, start + size);
    rows.push({ index, tokens: slice, hash: slice.length === size ? (hashes[index] ?? null) : null, start });
  }
  return (
    <div className="border-y-[1.5px] border-ink">
      {rows.map((row) => {
        const partial = row.tokens.length < size;
        return (
          <div
            key={row.index}
            className={cn(
              "grid grid-cols-[8.5rem_minmax(0,1fr)] gap-x-4 py-2",
              row.index > 0 && "border-t border-rule",
            )}
          >
            <div className="flex flex-col gap-0.5 pt-0.5">
              <span className="cond text-xs font-semibold text-ink">
                {partial ? k.tail : k.block(row.index)}
                <span className="ml-1.5 font-sans font-normal text-ink-3 tabular-nums">
                  {row.start}–{row.start + row.tokens.length - 1}
                </span>
              </span>
              {row.hash ? (
                <span className="font-mono text-2xs break-all text-ink-2" title={k.hashTitle}>
                  {row.hash}
                </span>
              ) : (
                <span className="text-2xs leading-snug text-ink-3">{k.tailNote(size)}</span>
              )}
            </div>
            <div className={cn("flex flex-wrap content-start gap-[3px]", partial && "rounded-xs border border-dashed border-rule-2 p-1")}>
              {row.tokens.map((token, offset) => (
                <TokenChip key={row.start + offset} token={token} showId={showId} index={row.start + offset} />
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
