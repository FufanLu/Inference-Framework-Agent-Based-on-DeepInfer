import type { ReactNode } from "react";
import { isGatewayError } from "@/api/client";
import { useT } from "@/i18n";
import { useSettings } from "@/state/settings";
import { openSettings } from "@/state/ui";
import { cn } from "@/lib/cn";
import { Button } from "./ui/Button";

/** A shell command the reader can copy. */
export function Command({ children }: { children: string }) {
  return (
    <code className="block overflow-x-auto rounded-xs border border-rule bg-wash px-3 py-2 font-mono text-xs leading-relaxed whitespace-pre text-ink">
      {children}
    </code>
  );
}

interface EmptyStateProps {
  title: string;
  children?: ReactNode;
  action?: ReactNode;
  className?: string;
  compact?: boolean;
}

/** No data yet: say why, and how to produce some. */
export function EmptyState({ title, children, action, className, compact }: EmptyStateProps) {
  return (
    <div className={cn("flex flex-col gap-2.5 text-sm", compact ? "py-4" : "py-10", className)}>
      <p className="font-semibold text-ink">{title}</p>
      {children ? <div className="flex max-w-[68ch] flex-col gap-2.5 text-ink-2">{children}</div> : null}
      {action ? <div className="pt-1">{action}</div> : null}
    </div>
  );
}

export const GATEWAY_COMMAND = "cargo run -p lev-gateway -- --worker http://127.0.0.1:50051";
export const DEV_STACK_COMMAND = "scripts/dev.sh            # 2 prefill + 2 decode sim workers and a gateway on :8080";

/** Explain a failed gateway call: network, 401, 503 or anything else. */
export function ErrorState({ error, what, compact, className }: { error: unknown; what?: string; compact?: boolean; className?: string }) {
  const t = useT();
  const { gatewayUrl } = useSettings();
  const title = (text: string) => (what ? t.states.withContext(what, text) : text);
  if (isGatewayError(error)) {
    if (error.kind === "network") {
      return (
        <EmptyState
          compact={compact}
          className={className}
          title={title(t.states.networkTitle)}
          action={
            <Button size="sm" onClick={openSettings}>
              {t.states.changeUrl}
            </Button>
          }
        >
          <p>
            {t.states.networkBody} <code className="font-mono text-ink">{gatewayUrl}</code>
            {t.states.networkBodyAfter}
          </p>
          <Command>{GATEWAY_COMMAND}</Command>
        </EmptyState>
      );
    }
    if (error.unauthorized) {
      return (
        <EmptyState
          compact={compact}
          className={className}
          title={title(t.states.unauthorizedTitle)}
          action={
            <Button size="sm" variant="primary" onClick={openSettings}>
              {t.states.enterToken}
            </Button>
          }
        >
          <p>{t.states.unauthorizedBody}</p>
        </EmptyState>
      );
    }
    if (error.noWorkers) {
      return (
        <EmptyState compact={compact} className={className} title={title(t.states.noWorkersTitle)}>
          <p>{t.states.noWorkersBody(error.code ?? "no_available_worker", error.message)}</p>
          <p>{t.states.noWorkersStart}</p>
          <Command>{DEV_STACK_COMMAND}</Command>
        </EmptyState>
      );
    }
    return (
      <EmptyState compact={compact} className={className} title={title(t.states.httpTitle(error.status))}>
        <p>
          {error.message}
          {error.code ? <code className="ml-1.5 font-mono text-xs text-ink-3">{error.code}</code> : null}
        </p>
      </EmptyState>
    );
  }
  return (
    <EmptyState compact={compact} className={className} title={title(t.states.genericTitle)}>
      <p>{error instanceof Error ? error.message : String(error)}</p>
    </EmptyState>
  );
}

/** Placeholder bars while the first response is in flight. */
export function Skeleton({ className }: { className?: string }) {
  return <div className={cn("animate-pulse rounded-xs bg-wash", className)} aria-hidden />;
}
