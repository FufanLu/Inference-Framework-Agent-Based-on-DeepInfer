import { Plus } from "lucide-react";
import { useMemo, useState } from "react";
import { useTenants, useUsage } from "@/api/queries";
import type { Tenant, UsageTotals } from "@/api/types";
import { PageHeader, Section } from "@/components/PageHeader";
import { EmptyState, ErrorState, Skeleton } from "@/components/States";
import { Button } from "@/components/ui/Button";
import { Drawer } from "@/components/ui/Drawer";
import { Select } from "@/components/ui/Field";
import { Segmented } from "@/components/ui/Segmented";
import { StatusMark } from "@/components/ui/StatusMark";
import { useT, type Messages } from "@/i18n";
import { cn } from "@/lib/cn";
import { fmtInt, fmtPct } from "@/lib/format";
import { KeysPanel } from "./tenants/KeysPanel";
import { TenantForm } from "./tenants/TenantForm";
import { UsageChart, UsageLegend } from "./tenants/UsageChart";
import { dayRange, totalsByDay } from "./tenants/usage";

const DAYS = [
  { value: "7", label: "7" },
  { value: "30", label: "30" },
  { value: "90", label: "90" },
] as const;

type Panel = { kind: "create" } | { kind: "edit"; tenant: Tenant } | { kind: "keys"; tenant: Tenant } | null;

function Limit({ value, t }: { value: number | null; t: Messages }) {
  return value === null ? <span className="text-ink-3">{t.tenants.unlimited}</span> : <>{fmtInt(value)}</>;
}

/** Month-to-date billable tokens against the monthly quota. */
function QuotaBar({ usage, quota, t }: { usage: UsageTotals; quota: number | null; t: Messages }) {
  const used = usage.prompt_tokens + usage.completion_tokens;
  if (quota === null) {
    return (
      <div className="flex flex-col gap-1">
        <span className="text-xs tabular-nums">{t.tenants.tokensUsed(fmtInt(used))}</span>
        <span className="text-2xs text-ink-3">{t.tenants.noQuota}</span>
      </div>
    );
  }
  const ratio = quota > 0 ? used / quota : 0;
  const tone = ratio >= 1 ? "bg-bad" : ratio >= 0.9 ? "bg-warn" : "bg-ink";
  return (
    <div className="flex flex-col gap-1">
      <div
        role="meter"
        aria-valuemin={0}
        aria-valuemax={quota}
        aria-valuenow={Math.min(used, quota)}
        aria-label={t.tenants.quotaUse(fmtInt(used), fmtInt(quota))}
        className="relative h-2 w-full border border-ink-3/70"
      >
        <div className={cn("absolute inset-y-0 left-0", tone)} style={{ width: `${Math.min(100, ratio * 100)}%` }} />
      </div>
      <span className="text-2xs text-ink-2 tabular-nums">
        {t.tenants.quotaUse(fmtInt(used), fmtInt(quota))} · {fmtPct(ratio)}
      </span>
    </div>
  );
}

export function TenantsPage() {
  const t = useT();
  const k = t.tenants;
  const tenants = useTenants();
  const [panel, setPanel] = useState<Panel>(null);
  const [filter, setFilter] = useState<string>("all");
  const [days, setDays] = useState(30);
  const usage = useUsage(filter === "all" ? null : filter, days);

  const rows = useMemo(() => tenants.data?.tenants ?? [], [tenants.data]);
  const tenantIds = useMemo(() => {
    const ids = new Set(rows.map((row) => row.tenant.id));
    for (const row of usage.data?.usage ?? []) ids.add(row.tenant_id);
    return [...ids].sort();
  }, [rows, usage.data]);
  const series = useMemo(() => totalsByDay(usage.data?.usage ?? [], dayRange(days)), [usage.data, days]);
  const totals = series.reduce(
    (sum, day) => ({
      requests: sum.requests + day.requests,
      uncached: sum.uncached + day.uncached,
      cached: sum.cached + day.cached,
      completion: sum.completion + day.completion,
    }),
    { requests: 0, uncached: 0, cached: 0, completion: 0 },
  );

  const panelTitle =
    panel?.kind === "create" ? k.form.createTitle : panel?.kind === "edit" ? k.form.editTitle(panel.tenant.id) : panel?.kind === "keys" ? k.keysPanel.title(panel.tenant.id) : "";

  return (
    <div className="flex flex-col gap-8">
      <PageHeader
        title={k.title}
        description={k.description}
        actions={
          <Button variant="primary" icon={<Plus className="size-3.5" />} onClick={() => setPanel({ kind: "create" })}>
            {k.newTenant}
          </Button>
        }
      />

      <Section title={k.tableCaption}>
        {tenants.error ? (
          <ErrorState error={tenants.error} what={k.readError} />
        ) : !tenants.data ? (
          <Skeleton className="h-40 w-full" />
        ) : rows.length === 0 ? (
          <div className="border-y-[1.5px] border-ink">
            <EmptyState
              title={k.emptyTitle}
              className="px-1"
              action={
                <Button size="sm" variant="primary" onClick={() => setPanel({ kind: "create" })}>
                  {k.newTenant}
                </Button>
              }
            >
              <p>{k.emptyBody}</p>
            </EmptyState>
          </div>
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="table-3l min-w-[920px]">
                <thead>
                  <tr>
                    <th>{k.id}</th>
                    <th>{k.name}</th>
                    <th className="num">{k.rpm}</th>
                    <th className="num">{k.tpm}</th>
                    <th className="num">{k.quota}</th>
                    <th className="num">{k.requests}</th>
                    <th className="w-[24%]">{k.mtd}</th>
                    <th>{k.state}</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {rows.map(({ tenant, usage_this_month: mtd }) => (
                    <tr key={tenant.id}>
                      <td className="font-mono text-xs font-semibold">{tenant.id}</td>
                      <td className="text-sm">{tenant.name}</td>
                      <td className="num text-xs">
                        <Limit value={tenant.rpm} t={t} />
                      </td>
                      <td className="num text-xs">
                        <Limit value={tenant.tpm} t={t} />
                      </td>
                      <td className="num text-xs">
                        <Limit value={tenant.monthly_token_quota} t={t} />
                      </td>
                      <td className="num text-xs">{fmtInt(mtd.requests)}</td>
                      <td>
                        <QuotaBar usage={mtd} quota={tenant.monthly_token_quota} t={t} />
                      </td>
                      <td>
                        {tenant.disabled ? (
                          <StatusMark state="down" tone="bad" label={k.disabled} />
                        ) : (
                          <StatusMark state="live" tone="ok" label={k.active} />
                        )}
                      </td>
                      <td className="text-right whitespace-nowrap">
                        <Button size="sm" variant="ghost" onClick={() => setPanel({ kind: "edit", tenant })}>
                          {k.edit}
                        </Button>
                        <Button size="sm" variant="ghost" onClick={() => setPanel({ kind: "keys", tenant })}>
                          {k.keys}
                        </Button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="mt-2 text-xs text-ink-3">{k.quotaNote}</p>
          </>
        )}
      </Section>

      <Section
        title={k.usage.title}
        caption={k.usage.caption(days)}
        aside={
          <>
            <label className="flex items-center gap-2 text-xs text-ink-3">
              {k.usage.tenant}
              <Select value={filter} onChange={(event) => setFilter(event.target.value)} className="h-7 w-44 text-xs">
                <option value="all">{k.usage.allTenants}</option>
                {tenantIds.map((id) => (
                  <option key={id} value={id}>
                    {id}
                  </option>
                ))}
              </Select>
            </label>
            <span className="text-xs text-ink-3">{k.usage.days}</span>
            <Segmented label={k.usage.days} size="sm" value={String(days)} options={DAYS} onChange={(value) => setDays(Number(value))} />
          </>
        }
      >
        {usage.error ? (
          <ErrorState error={usage.error} what={k.usage.readError} />
        ) : !usage.data ? (
          <Skeleton className="h-56 w-full" />
        ) : (
          <div className="flex flex-col gap-3 border-t-[1.5px] border-ink pt-3">
            <UsageChart data={series} />
            <div className="flex flex-wrap items-center justify-between gap-3">
              <UsageLegend />
              <p className="text-xs text-ink-2 tabular-nums">
                {k.usage.totals}: {k.usage.requests(fmtInt(totals.requests))} · {k.usage.uncached} {fmtInt(totals.uncached)} · {k.usage.cached}{" "}
                {fmtInt(totals.cached)} · {k.usage.completion} {fmtInt(totals.completion)}
              </p>
            </div>
            {totals.requests === 0 ? <p className="text-sm text-ink-3">{k.usage.empty}</p> : null}
          </div>
        )}
      </Section>

      <Drawer
        open={panel !== null}
        onOpenChange={(open) => {
          if (!open) setPanel(null);
        }}
        title={panelTitle}
        description={panel?.kind === "keys" ? panel.tenant.name : panel ? k.form.description : undefined}
        width="min(620px, 92vw)"
      >
        {panel?.kind === "create" ? <TenantForm tenant={null} onDone={() => setPanel(null)} /> : null}
        {panel?.kind === "edit" ? <TenantForm key={panel.tenant.id} tenant={panel.tenant} onDone={() => setPanel(null)} /> : null}
        {panel?.kind === "keys" ? <KeysPanel tenant={panel.tenant.id} /> : null}
      </Drawer>
    </div>
  );
}
