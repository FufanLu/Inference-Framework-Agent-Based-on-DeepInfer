import { NavLink } from "react-router";
import { Bot, Building2, FlaskConical, Languages, LayoutGrid, Moon, Network, ScrollText, Settings, Sun, SunMoon } from "lucide-react";
import type { ComponentType } from "react";
import { liveHub, useLiveState, type LiveReason } from "@/api/live";
import { useGatewayHealth, useOverview } from "@/api/queries";
import { isGatewayError } from "@/api/client";
import { setLocale, useLocale, useT, type Messages } from "@/i18n";
import { setTheme, useSettings, useTheme, type ThemePreference } from "@/state/settings";
import { openSettings } from "@/state/ui";
import { fmtInt } from "@/lib/format";
import { useNow } from "@/lib/useNow";
import { cn } from "@/lib/cn";
import { Mark } from "./Mark";
import { StatusMark } from "./ui/StatusMark";

type NavKey = "overview" | "cluster" | "playground" | "requests" | "agent" | "tenants";

const NAV: { to: string; key: NavKey; icon: ComponentType<{ className?: string; strokeWidth?: number }> }[] = [
  { to: "/", key: "overview", icon: LayoutGrid },
  { to: "/cluster", key: "cluster", icon: Network },
  { to: "/playground", key: "playground", icon: FlaskConical },
  { to: "/requests", key: "requests", icon: ScrollText },
  { to: "/agent", key: "agent", icon: Bot },
  { to: "/tenants", key: "tenants", icon: Building2 },
];

const THEME_NEXT: Record<ThemePreference, ThemePreference> = { system: "light", light: "dark", dark: "system" };

function reasonText(reason: LiveReason, t: Messages): string {
  switch (reason.kind) {
    case "http":
      return t.conn.reasons.http(reason.status);
    default:
      return t.conn.reasons[reason.kind];
  }
}

const footButton =
  "inline-flex h-7 items-center gap-1.5 rounded-sm px-2 text-xs text-ink-2 transition-colors duration-150 hover:bg-rule/60 hover:text-ink";

export function Sidebar() {
  const t = useT();
  const locale = useLocale();
  const overview = useOverview();
  const health = useGatewayHealth();
  const live = useLiveState();
  const theme = useTheme();
  const { gatewayUrl } = useSettings();
  const now = useNow(1000);

  const host = gatewayUrl.replace(/^https?:\/\//, "");
  const gateway = health.isPending
    ? ({ state: "idle", tone: "muted", label: t.conn.checking } as const)
    : health.data?.ok
      ? ({ state: "live", tone: "ok", label: t.conn.online } as const)
      : ({ state: "down", tone: "bad", label: health.data ? `HTTP ${health.data.status}` : t.conn.offline } as const);

  const retryIn = live.retryAt ? Math.max(0, Math.ceil((live.retryAt - now) / 1000)) : 0;
  const stream =
    live.status === "open"
      ? ({ state: "live", tone: "ok", label: t.conn.connected } as const)
      : live.status === "retrying"
        ? ({ state: "stale", tone: "warn", label: t.conn.retrying(retryIn) } as const)
        : live.status === "unauthorized"
          ? ({ state: "down", tone: "bad", label: t.conn.needToken } as const)
          : ({ state: "idle", tone: "muted", label: t.conn.connecting } as const);

  const workers = overview.data ? `${overview.data.healthy_workers}/${overview.data.workers}` : null;
  const overviewError = overview.error && isGatewayError(overview.error) ? overview.error : null;
  const themeName = t.theme[theme];

  return (
    <div className="shrink-0 border-r border-rule bg-wash">
    <aside className="sticky top-0 flex h-dvh w-[220px] flex-col xl:w-[232px]">
      <div className="flex items-center gap-2.5 px-5 pt-5 pb-4">
        <Mark className="size-7 text-ink" />
        <div className="leading-none">
          <div className="cond text-[1.0625rem] font-bold tracking-[0.02em] text-ink">{t.app.name}</div>
          <div className="mt-1 text-2xs text-ink-3">{t.app.tagline}</div>
        </div>
      </div>

      <nav aria-label={t.nav.label} className="mt-2 flex flex-col px-3">
        {NAV.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === "/"}
            className={({ isActive }) =>
              cn(
                "group flex h-9 items-center gap-2.5 rounded-sm px-2.5 text-sm transition-colors duration-150",
                isActive ? "bg-ink text-paper" : "text-ink-2 hover:bg-rule/60 hover:text-ink",
              )
            }
          >
            <item.icon className="size-4 shrink-0" strokeWidth={1.75} />
            <span className="flex-1 truncate">{t.nav[item.key]}</span>
            {item.key === "cluster" && workers ? (
              <span className="font-mono text-2xs tabular-nums opacity-80" title={t.nav.healthyTitle}>
                {workers}
              </span>
            ) : null}
          </NavLink>
        ))}
      </nav>

      <div className="mt-auto flex flex-col gap-3 border-t border-rule px-5 py-4 text-xs">
        <dl className="grid grid-cols-[auto_1fr] items-center gap-x-3 gap-y-2">
          <dt className="text-ink-3">{t.conn.gateway}</dt>
          <dd className="min-w-0">
            <StatusMark {...gateway} />
          </dd>
          <dt className="sr-only">{t.conn.address}</dt>
          <dd className="col-span-2 -mt-1 truncate font-mono text-2xs text-ink-3" title={gatewayUrl}>
            {host}
          </dd>
          <dt className="text-ink-3">{t.conn.stream}</dt>
          <dd className="flex min-w-0 items-center justify-between gap-2">
            <StatusMark {...stream} />
            {live.status === "retrying" || live.status === "unauthorized" ? (
              <button
                type="button"
                className="text-2xs text-ink-2 underline decoration-rule-2 hover:text-ink"
                onClick={() => (live.status === "unauthorized" ? openSettings() : liveHub.restart())}
              >
                {live.status === "unauthorized" ? t.conn.openSettings : t.conn.reconnect}
              </button>
            ) : null}
          </dd>
          {overview.data ? (
            <>
              <dt className="text-ink-3">{t.conn.version}</dt>
              <dd className="flex flex-wrap gap-x-1 font-mono text-2xs text-ink-2" title={overview.data.fingerprint}>
                <span className="whitespace-nowrap">v{overview.data.version} ·</span>
                <span className="whitespace-nowrap">{overview.data.fingerprint.replace(/^fp_lev_/, "")}</span>
              </dd>
            </>
          ) : overviewError?.unauthorized ? (
            <>
              <dt className="text-ink-3">{t.conn.admin}</dt>
              <dd className="text-bad">{t.conn.adminUnauthorized}</dd>
            </>
          ) : null}
        </dl>
        {live.status === "open" ? (
          <p className="text-2xs text-ink-3 tabular-nums">{t.conn.received(fmtInt(liveHub.getData().received))}</p>
        ) : live.reason ? (
          <p className="text-2xs text-ink-3">{reasonText(live.reason, t)}</p>
        ) : null}
        <div className="-mx-2 flex flex-wrap items-center gap-x-0.5 gap-y-1 pt-1">
          <button type="button" onClick={openSettings} className={footButton}>
            <Settings className="size-3.5" strokeWidth={1.75} />
            {t.conn.settings}
          </button>
          <button type="button" onClick={() => setTheme(THEME_NEXT[theme])} className={footButton} title={t.theme.cycle(themeName)}>
            {theme === "system" ? (
              <SunMoon className="size-3.5" strokeWidth={1.75} />
            ) : theme === "light" ? (
              <Sun className="size-3.5" strokeWidth={1.75} />
            ) : (
              <Moon className="size-3.5" strokeWidth={1.75} />
            )}
            {themeName}
          </button>
          <button
            type="button"
            onClick={() => setLocale(locale === "en" ? "zh" : "en")}
            className={footButton}
            title={t.locale.toggleTitle}
            lang={locale === "en" ? "zh-CN" : "en"}
          >
            <Languages className="size-3.5" strokeWidth={1.75} />
            {t.locale.toggle}
          </button>
        </div>
      </div>
    </aside>
    </div>
  );
}
