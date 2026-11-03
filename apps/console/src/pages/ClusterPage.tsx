import { useMemo, useState } from "react";
import { useLiveData } from "@/api/live";
import { useAddWorker, useRouterSnapshot, useWorkers } from "@/api/queries";
import type { RouteEvent } from "@/api/types";
import { PageHeader } from "@/components/PageHeader";
import { Command, EmptyState, ErrorState, Skeleton } from "@/components/States";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Field";
import { useT } from "@/i18n";
import { useNow } from "@/lib/useNow";
import { usePrefersReducedMotion } from "@/state/ui";
import { DecisionPanel } from "./cluster/DecisionPanel";
import { ClusterDiagram } from "./cluster/Diagram";
import { RouterControls } from "./cluster/RouterControls";

const DEV_STACK = `scripts/dev.sh             # 2 prefill + 2 decode sim workers, gateway on :8080
scripts/dev.sh unified 4   # or 4 unified workers`;

export function ClusterPage() {
  const t = useT();
  const router = useRouterSnapshot(1_000);
  const workers = useWorkers(5_000);
  const live = useLiveData();
  const reducedMotion = usePrefersReducedMotion();
  const [pinned, setPinned] = useState<RouteEvent | null>(null);
  const now = useNow(1_000);

  const info = useMemo(() => new Map((workers.data?.workers ?? []).map((w) => [w.id, w])), [workers.data]);
  const pending = useMemo(() => {
    if (!workers.data) return [];
    const connected = new Set(workers.data.workers.map((w) => w.endpoint.replace(/\/+$/, "")));
    return workers.data.endpoints.filter((endpoint) => !connected.has(endpoint.replace(/\/+$/, "")));
  }, [workers.data]);
  const recentRoutes = useMemo(() => live.routes.filter((route) => route.at_ms > now - 10_000).length, [live.routes, now]);

  const snapshot = router.data;

  return (
    <div className="flex flex-col gap-6">
      <PageHeader title={t.cluster.title} description={t.cluster.description} actions={<RouterControls snapshot={snapshot} />} />

      {router.error ? (
        <ErrorState error={router.error} what="/admin/router" />
      ) : !snapshot ? (
        <div className="grid gap-8 xl:grid-cols-[minmax(0,1fr)_minmax(340px,400px)]">
          <Skeleton className="h-[420px] w-full" />
          <Skeleton className="h-[320px] w-full" />
        </div>
      ) : (
        <div className="grid items-start gap-x-8 gap-y-10 xl:grid-cols-[minmax(0,1fr)_minmax(340px,400px)]">
          {snapshot.workers.length === 0 && pending.length === 0 ? (
            <EmptyState title={t.cluster.emptyTitle}>
              <p>{t.cluster.emptyBody}</p>
              <Command>{DEV_STACK}</Command>
              <p>{t.cluster.emptyAfter}</p>
              <AddWorker />
            </EmptyState>
          ) : (
            <ClusterDiagram
              snapshot={snapshot}
              info={info}
              pending={pending}
              reducedMotion={reducedMotion}
              recentRoutes={recentRoutes}
              footer={
                <figcaption className="border-t border-rule pt-3">
                  <AddWorker />
                </figcaption>
              }
            />
          )}
          <DecisionPanel routes={live.routes} stages={live.stages} pinned={pinned} onPin={setPinned} />
        </div>
      )}
    </div>
  );
}

function AddWorker() {
  const t = useT();
  const [endpoint, setEndpoint] = useState("");
  const add = useAddWorker();
  const valid = /^https?:\/\/[^\s/]+:\d+\/?$/.test(endpoint.trim());
  return (
    <form
      className="flex flex-wrap items-center gap-2"
      onSubmit={(event) => {
        event.preventDefault();
        if (valid) add.mutate(endpoint.trim(), { onSuccess: () => setEndpoint("") });
      }}
    >
      <label htmlFor="worker-endpoint" className="text-xs text-ink-3">
        {t.cluster.addWorker}
      </label>
      <Input
        id="worker-endpoint"
        value={endpoint}
        onChange={(event) => setEndpoint(event.target.value)}
        placeholder="http://127.0.0.1:50055"
        className="h-7 w-60 font-mono text-xs"
        spellCheck={false}
      />
      <Button size="sm" type="submit" disabled={!valid} busy={add.isPending}>
        {t.cluster.add}
      </Button>
      {add.isSuccess ? <span className="text-xs text-ok">{t.cluster.added(add.data.endpoint)}</span> : null}
      {add.error ? (
        <span role="alert" className="text-xs text-bad">
          {add.error.message}
        </span>
      ) : null}
    </form>
  );
}
