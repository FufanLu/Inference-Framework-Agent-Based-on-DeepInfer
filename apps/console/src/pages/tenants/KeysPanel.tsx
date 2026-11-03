import { Check, Copy, KeyRound } from "lucide-react";
import { useState } from "react";
import { useCreateKey, useKeys, useRevokeKey } from "@/api/queries";
import type { CreatedKey } from "@/api/types";
import { EmptyState, ErrorState, Skeleton } from "@/components/States";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Field";
import { StatusMark } from "@/components/ui/StatusMark";
import { useT } from "@/i18n";
import { fmtStamp } from "@/lib/format";
import { getSettings, saveSettings, useSettings } from "@/state/settings";

/** The secret of a key that was just created: shown once, copy or lose it. */
function SecretOnce({ created, onDismiss }: { created: CreatedKey; onDismiss: () => void }) {
  const t = useT();
  const k = t.tenants.keysPanel;
  const settings = useSettings();
  const [copied, setCopied] = useState<"idle" | "ok" | "failed">("idle");
  const inUse = settings.apiKey === created.key;
  return (
    <div role="alert" className="flex flex-col gap-3 border-y-[1.5px] border-warn py-3">
      <div>
        <p className="font-semibold text-ink">{k.secretTitle}</p>
        <p className="mt-0.5 text-xs text-ink-2">{k.secretBody}</p>
      </div>
      <code className="block overflow-x-auto border border-ink bg-sheet px-3 py-2 font-mono text-sm break-all select-all">{created.key}</code>
      <div className="flex flex-wrap items-center gap-2">
        <Button
          size="sm"
          variant="primary"
          icon={copied === "ok" ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
          onClick={async () => {
            try {
              await navigator.clipboard.writeText(created.key);
              setCopied("ok");
            } catch {
              setCopied("failed");
            }
          }}
        >
          {copied === "ok" ? k.copied : k.copy}
        </Button>
        <Button size="sm" icon={<KeyRound className="size-3.5" />} disabled={inUse} onClick={() => saveSettings({ ...getSettings(), apiKey: created.key })}>
          {inUse ? k.inUse : k.useInPlayground}
        </Button>
        <Button size="sm" variant="ghost" onClick={onDismiss}>
          {k.dismiss}
        </Button>
        {copied === "failed" ? <span className="text-xs text-bad">{k.copyFailed}</span> : null}
      </div>
    </div>
  );
}

/** API keys of one tenant: create (secret shown once), list, revoke. */
export function KeysPanel({ tenant }: { tenant: string }) {
  const t = useT();
  const k = t.tenants.keysPanel;
  const keys = useKeys(tenant);
  const create = useCreateKey();
  const revoke = useRevokeKey();
  const [name, setName] = useState("");
  const [created, setCreated] = useState<CreatedKey | null>(null);
  const [confirming, setConfirming] = useState<string | null>(null);

  return (
    <div className="flex flex-col gap-5">
      <p className="text-sm text-ink-2">{k.description}</p>
      <form
        className="flex flex-wrap items-end gap-2"
        onSubmit={(event) => {
          event.preventDefault();
          create.mutate(
            { tenant, name: name.trim() || "default" },
            {
              onSuccess: (key) => {
                setCreated(key);
                setName("");
              },
            },
          );
        }}
      >
        <label className="flex flex-col gap-1.5">
          <span className="cond text-xs font-semibold text-ink-2">{k.name}</span>
          <Input value={name} onChange={(event) => setName(event.target.value)} placeholder={k.namePlaceholder} className="w-56" autoComplete="off" />
        </label>
        <Button type="submit" variant="primary" busy={create.isPending}>
          {k.create}
        </Button>
      </form>
      {create.error ? <ErrorState error={create.error} what={k.failed} compact /> : null}
      {created ? <SecretOnce created={created} onDismiss={() => setCreated(null)} /> : null}

      {keys.error ? (
        <ErrorState error={keys.error} what={k.failed} compact />
      ) : !keys.data ? (
        <Skeleton className="h-24 w-full" />
      ) : keys.data.keys.length === 0 ? (
        <div className="border-y-[1.5px] border-ink">
          <EmptyState compact title={k.none} className="px-1" />
        </div>
      ) : (
        <table className="table-3l">
          <thead>
            <tr>
              <th>{k.columns.name}</th>
              <th>{k.columns.prefix}</th>
              <th>{k.columns.created}</th>
              <th>{k.columns.state}</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {keys.data.keys.map((key) => (
              <tr key={key.id}>
                <td className="text-sm">{key.name}</td>
                <td className="font-mono text-xs">{key.prefix}</td>
                <td className="font-mono text-xs text-ink-2">{fmtStamp(key.created_at)}</td>
                <td>
                  {key.revoked_at ? (
                    <StatusMark state="down" tone="muted" label={k.revoked(fmtStamp(key.revoked_at))} />
                  ) : (
                    <StatusMark state="live" tone="ok" label={k.live} />
                  )}
                </td>
                <td className="text-right">
                  {key.revoked_at ? null : confirming === key.id ? (
                    <span className="inline-flex flex-wrap items-center justify-end gap-2">
                      <span className="text-xs text-ink-2">{k.confirmRevoke}</span>
                      <Button
                        size="sm"
                        variant="danger"
                        busy={revoke.isPending}
                        onClick={() => revoke.mutate(key.id, { onSettled: () => setConfirming(null) })}
                      >
                        {k.confirm}
                      </Button>
                      <Button size="sm" variant="ghost" onClick={() => setConfirming(null)}>
                        {t.common.cancel}
                      </Button>
                    </span>
                  ) : (
                    <Button size="sm" variant="ghost" onClick={() => setConfirming(key.id)}>
                      {k.revoke}
                    </Button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {revoke.error ? <ErrorState error={revoke.error} what={k.failed} compact /> : null}
    </div>
  );
}
