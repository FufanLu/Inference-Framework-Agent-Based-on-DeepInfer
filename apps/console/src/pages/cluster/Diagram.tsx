import { forwardRef, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { liveHub } from "@/api/live";
import type { Plan, Role, RouterSnapshot, WorkerInfoEntry, WorkerView } from "@/api/types";
import { KvCells } from "@/components/KvCells";
import { RoleTag } from "@/components/ui/Badge";
import { StatusMark } from "@/components/ui/StatusMark";
import { useT, type Messages } from "@/i18n";
import { cn } from "@/lib/cn";
import { fmtInt, fmtPct, fmtTokens } from "@/lib/format";
import { workerLineState } from "@/lib/workers";
import { PacketEngine } from "./packets";

const ROLE_ORDER: Role[] = ["prefill", "decode", "unified"];
const NOTE_KEY = { prefill: "prefillNote", decode: "decodeNote", unified: "unifiedNote" } as const;

const REQUEST_LEG_MS = 650;
const KV_LEG_MS = 520;
const HIGHLIGHT_MS = 900;
/** Without any stage event from this gateway, start the KV leg this long after the first leg lands. */
const STAGE_FALLBACK_MS = 400;
/** Where a net enters a worker block, from its top edge. */
const PIN_Y = 21;

interface Net {
  key: string;
  d: string;
  kind: "request" | "kv";
  /** Nets the router can use in the current mode. */
  inUse: boolean;
  healthy: boolean;
  workers: string[];
  /** Where the net enters its worker, for the strike mark of an unhealthy net. */
  end: { x: number; y: number };
}

interface Geometry {
  width: number;
  height: number;
  gateway: { x: number; y: number };
  nets: Net[];
}

/** A disaggregated request between its route event and its KV hand-off. */
interface Flight {
  prefill: string;
  decode: string;
  landed: boolean;
  prefillDone: boolean;
  launched: boolean;
  created: number;
  fallback: number | null;
}

export interface DiagramProps {
  snapshot: RouterSnapshot;
  info: Map<string, WorkerInfoEntry>;
  /** Endpoints the gateway supervises that have not answered GetInfo. */
  pending: string[];
  reducedMotion: boolean;
  /** Route events seen in the last 10 s, for the gateway block. */
  recentRoutes: number;
  footer?: ReactNode;
}

export function ClusterDiagram({ snapshot, info, pending, reducedMotion, recentRoutes, footer }: DiagramProps) {
  const t = useT();
  const containerRef = useRef<HTMLDivElement | null>(null);
  const gatewayRef = useRef<HTMLDivElement | null>(null);
  const columnRef = useRef<HTMLDivElement | null>(null);
  const cardRefs = useRef(new Map<string, HTMLDivElement>());
  const pathRefs = useRef(new Map<string, SVGPathElement>());
  const highlightLayer = useRef<SVGGElement | null>(null);
  const packetLayer = useRef<SVGGElement | null>(null);
  const engine = useRef<PacketEngine | null>(null);
  const flights = useRef(new Map<string, Flight>());
  const stageSeen = useRef(false);
  const [geometry, setGeometry] = useState<Geometry | null>(null);
  const [hovered, setHovered] = useState<string | null>(null);

  const pools = useMemo(() => {
    const byRole = new Map<Role, WorkerView[]>();
    for (const worker of snapshot.workers) {
      const list = byRole.get(worker.role) ?? [];
      list.push(worker);
      byRole.set(worker.role, list);
    }
    return ROLE_ORDER.flatMap((role) => {
      const workers = byRole.get(role);
      return workers && workers.length > 0 ? [{ role, workers }] : [];
    });
  }, [snapshot.workers]);

  const healthyIn = (role: Role) => snapshot.workers.some((w) => w.role === role && w.healthy);
  const pdActive = snapshot.disaggregation && healthyIn("prefill") && healthyIn("decode");
  const unifiedUp = healthyIn("unified");
  const layoutKey = snapshot.workers.map((w) => `${w.id}:${w.role}:${w.healthy ? 1 : 0}`).join("|") + `|${pdActive}|${unifiedUp}`;

  // ------------------------------------------------------------ geometry
  const measure = useCallback(() => {
    const container = containerRef.current;
    const gateway = gatewayRef.current;
    const column = columnRef.current;
    if (!container || !gateway || !column) return;
    const box = container.getBoundingClientRect();
    const g = gateway.getBoundingClientRect();
    const col = column.getBoundingClientRect();
    const gx = g.right - box.left;
    const gy = g.top + g.height / 2 - box.top;
    const trunkX = Math.round(gx + (col.left - g.right) / 2);
    const handoffX = Math.round(col.right - box.left + 14);
    const nets: Net[] = [];
    const anchor = (id: string) => {
      const card = cardRefs.current.get(id);
      if (!card) return null;
      const r = card.getBoundingClientRect();
      return { left: r.left - box.left, right: r.right - box.left, y: Math.round(r.top - box.top + PIN_Y) };
    };
    for (const worker of snapshot.workers) {
      const a = anchor(worker.id);
      if (!a) continue;
      // Without a unified pool, prefill and decode workers run requests end to end.
      const inUse = worker.role === "unified" ? !pdActive : pdActive || !unifiedUp;
      nets.push({
        key: `g>${worker.id}`,
        d: `M${gx},${gy} H${trunkX} V${a.y} H${a.left}`,
        kind: "request",
        inUse,
        healthy: worker.healthy,
        workers: [worker.id],
        end: { x: a.left, y: a.y },
      });
    }
    const prefill = snapshot.workers.filter((w) => w.role === "prefill");
    const decode = snapshot.workers.filter((w) => w.role === "decode");
    for (const p of prefill) {
      const pa = anchor(p.id);
      if (!pa) continue;
      for (const d of decode) {
        const da = anchor(d.id);
        if (!da) continue;
        nets.push({
          key: `kv>${p.id}>${d.id}`,
          d: `M${pa.right},${pa.y + 6} H${handoffX} V${da.y + 6} H${da.right}`,
          kind: "kv",
          inUse: pdActive,
          healthy: p.healthy && d.healthy,
          workers: [p.id, d.id],
          end: { x: da.right, y: da.y + 6 },
        });
      }
    }
    setGeometry({ width: box.width, height: box.height, gateway: { x: gx, y: gy }, nets });
  }, [snapshot.workers, pdActive, unifiedUp]);

  useLayoutEffect(() => {
    measure();
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver(() => measure());
    observer.observe(container);
    if (columnRef.current) observer.observe(columnRef.current);
    return () => observer.disconnect();
  }, [layoutKey, measure]);

  // ------------------------------------------------------------- packets
  useEffect(() => {
    if (!packetLayer.current) return;
    engine.current = new PacketEngine(packetLayer.current);
    const pending = flights.current;
    return () => {
      engine.current?.dispose();
      engine.current = null;
      for (const flight of pending.values()) if (flight.fallback !== null) window.clearTimeout(flight.fallback);
      pending.clear();
    };
  }, []);

  const flashCard = useCallback((id: string) => {
    const card = cardRefs.current.get(id);
    if (!card) return;
    card.classList.remove("is-hit");
    // Restart the transition.
    void card.offsetWidth;
    card.classList.add("is-hit");
    window.setTimeout(() => card.classList.remove("is-hit"), 700);
  }, []);

  const highlight = useCallback(
    (key: string) => {
      const source = pathRefs.current.get(key);
      const layer = highlightLayer.current;
      if (!source || !layer) return;
      const clone = document.createElementNS("http://www.w3.org/2000/svg", "path");
      clone.setAttribute("d", source.getAttribute("d") ?? "");
      clone.setAttribute("fill", "none");
      clone.setAttribute("stroke", "var(--route)");
      clone.setAttribute("stroke-width", "2");
      if (key.startsWith("kv>")) clone.setAttribute("stroke-dasharray", "5 3");
      layer.appendChild(clone);
      if (reducedMotion) {
        window.setTimeout(() => clone.remove(), 1000);
        return;
      }
      clone.style.transition = `opacity ${HIGHLIGHT_MS}ms cubic-bezier(0.16, 1, 0.3, 1)`;
      clone.style.opacity = "1";
      requestAnimationFrame(() => {
        clone.style.opacity = "0";
      });
      window.setTimeout(() => clone.remove(), HIGHLIGHT_MS + 100);
    },
    [reducedMotion],
  );

  useEffect(() => {
    const all = flights.current;

    /** The KV leg: prefill → decode, once the first leg landed and prefill is done. */
    const launchKv = (id: string) => {
      const flight = all.get(id);
      if (!flight || flight.launched) return;
      flight.launched = true;
      if (flight.fallback !== null) window.clearTimeout(flight.fallback);
      all.delete(id);
      const key = `kv>${flight.prefill}>${flight.decode}`;
      const path = pathRefs.current.get(key);
      highlight(key);
      if (reducedMotion || !path || !engine.current) {
        flashCard(flight.decode);
        return;
      }
      engine.current.launch(path, KV_LEG_MS, "kv", () => flashCard(flight.decode));
    };

    const landFirst = (id: string, worker: string, plan: Plan) => {
      flashCard(worker);
      if (plan.kind !== "disaggregated") return;
      const flight = all.get(id);
      if (!flight) return;
      flight.landed = true;
      if (flight.prefillDone) launchKv(id);
      else if (!stageSeen.current) flight.fallback = window.setTimeout(() => launchKv(id), STAGE_FALLBACK_MS);
    };

    const sweep = window.setInterval(() => {
      const cutoff = performance.now() - 15_000;
      for (const [id, flight] of all) {
        if (flight.created < cutoff) {
          if (flight.fallback !== null) window.clearTimeout(flight.fallback);
          all.delete(id);
        }
      }
    }, 5_000);

    const unsubscribe = liveHub.onEvent((event) => {
      if (event.type === "stage") {
        stageSeen.current = true;
        const flight = all.get(event.request_id);
        if (!flight) return;
        if (event.stage === "prefill_done") {
          flight.prefillDone = true;
          if (flight.landed) launchKv(event.request_id);
        }
        return;
      }
      if (event.type !== "route") return;
      const plan = event.plan;
      const first = plan.kind === "unified" ? plan.worker.worker_id : plan.prefill.worker_id;
      if (plan.kind === "disaggregated") {
        all.set(event.request_id, {
          prefill: plan.prefill.worker_id,
          decode: plan.decode.worker_id,
          landed: false,
          prefillDone: false,
          launched: false,
          created: performance.now(),
          fallback: null,
        });
      }
      const firstKey = `g>${first}`;
      const firstPath = pathRefs.current.get(firstKey);
      highlight(firstKey);
      if (reducedMotion || !firstPath || !engine.current) {
        landFirst(event.request_id, first, plan);
        return;
      }
      engine.current.launch(firstPath, REQUEST_LEG_MS, "request", () => landFirst(event.request_id, first, plan));
    });
    return () => {
      unsubscribe();
      window.clearInterval(sweep);
    };
  }, [flashCard, highlight, reducedMotion]);

  const related = (net: Net) => hovered === null || net.workers.includes(hovered);

  return (
    <figure className="flex flex-col gap-3">
      <div
        ref={containerRef}
        className="relative grid grid-cols-[minmax(204px,236px)_44px_minmax(0,1fr)_30px] items-start"
        onPointerLeave={() => setHovered(null)}
      >
        <div className="flex flex-col gap-6 self-center py-4">
          <GatewayBlock ref={gatewayRef} snapshot={snapshot} pdActive={pdActive} recentRoutes={recentRoutes} t={t} />
          <FigureKey t={t} reducedMotion={reducedMotion} />
        </div>
        <div aria-hidden />
        <div ref={columnRef} className="flex min-w-0 flex-col gap-4">
          {pools.map(({ role, workers }) => {
            const active = role === "unified" ? !pdActive && unifiedUp : pdActive || !unifiedUp;
            return (
              <PoolFrame key={role} role={role} workers={workers} active={active} t={t}>
                {workers.map((worker) => (
                  <WorkerBlock
                    key={worker.id}
                    ref={(element) => {
                      if (element) cardRefs.current.set(worker.id, element);
                      else cardRefs.current.delete(worker.id);
                    }}
                    worker={worker}
                    info={info.get(worker.id)}
                    dimmed={hovered !== null && hovered !== worker.id && !isPartner(geometry, hovered, worker.id)}
                    onHover={setHovered}
                    t={t}
                  />
                ))}
              </PoolFrame>
            );
          })}
          {pending.length > 0 ? (
            <div className="flex flex-col gap-1.5 border-[0.75px] border-rule-2 px-3 py-2.5">
              <p className="cond text-xs font-semibold text-ink-2">{t.cluster.pendingTitle}</p>
              {pending.map((endpoint) => (
                <p key={endpoint} className="flex flex-wrap items-center gap-2 text-xs text-ink-3">
                  <StatusMark state="idle" tone="muted" />
                  <span className="font-mono">{endpoint}</span>
                  <span>{t.cluster.pendingNote}</span>
                </p>
              ))}
            </div>
          ) : null}
        </div>
        <div aria-hidden />

        {geometry ? (
          <svg className="pointer-events-none absolute inset-0 z-20 overflow-visible" width={geometry.width} height={geometry.height} aria-hidden>
            <defs>
              <marker id="arrow-net" viewBox="0 0 8 8" refX="7.5" refY="4" markerWidth="7" markerHeight="7" orient="auto">
                <path d="M0,0.5 L8,4 L0,7.5 z" fill="var(--ink-3)" />
              </marker>
              <marker id="arrow-idle" viewBox="0 0 8 8" refX="7.5" refY="4" markerWidth="7" markerHeight="7" orient="auto">
                <path d="M0,0.5 L8,4 L0,7.5 z" fill="var(--rule-2)" />
              </marker>
            </defs>
            <g>
              {geometry.nets.map((net) => {
                const strong = net.inUse && net.healthy;
                return (
                  <path
                    key={net.key}
                    ref={(element) => {
                      if (element) pathRefs.current.set(net.key, element);
                      else pathRefs.current.delete(net.key);
                    }}
                    d={net.d}
                    fill="none"
                    stroke={strong ? "var(--ink-3)" : "var(--rule-2)"}
                    strokeWidth={strong ? 1.25 : 1}
                    strokeDasharray={net.kind === "kv" ? "5 3" : net.healthy && !net.inUse ? "1 3" : undefined}
                    markerEnd={strong ? "url(#arrow-net)" : "url(#arrow-idle)"}
                    opacity={related(net) ? 1 : 0.18}
                    style={{ transition: "opacity 150ms ease-out" }}
                  />
                );
              })}
            </g>
            <g aria-hidden>
              {geometry.nets
                .filter((net) => !net.healthy)
                .map((net) => {
                  const x = net.kind === "kv" ? net.end.x + 9 : net.end.x - 12;
                  return (
                    <line key={`strike-${net.key}`} x1={x - 4} y1={net.end.y + 5} x2={x + 4} y2={net.end.y - 5} stroke="var(--bad)" strokeWidth="1.5" opacity={related(net) ? 1 : 0.18} />
                  );
                })}
            </g>
            <g ref={highlightLayer} />
            <rect x={geometry.gateway.x - 3} y={geometry.gateway.y - 3} width="6" height="6" fill="var(--sheet)" stroke="var(--ink)" strokeWidth="1.25" />
            <g ref={packetLayer} />
          </svg>
        ) : null}
      </div>
      {footer}
    </figure>
  );
}

function isPartner(geometry: Geometry | null, hovered: string, other: string) {
  return geometry?.nets.some((net) => net.kind === "kv" && net.workers.includes(hovered) && net.workers.includes(other)) ?? false;
}

// ---------------------------------------------------------------- blocks

const GatewayBlock = forwardRef<HTMLDivElement, { snapshot: RouterSnapshot; pdActive: boolean; recentRoutes: number; t: Messages }>(
  function GatewayBlock({ snapshot, pdActive, recentRoutes, t }, ref) {
    const k = t.cluster.gateway;
    const rows: [string, ReactNode][] = [
      [
        k.protocols,
        <span className="inline-flex flex-wrap justify-end gap-x-1">
          {["chat", "responses", "messages"].map((name, index, all) => (
            <span key={name} className="whitespace-nowrap">
              {name}
              {index < all.length - 1 ? " ·" : ""}
            </span>
          ))}
        </span>,
      ],
      [k.render, <span className="whitespace-nowrap">{k.renderValue}</span>],
      [k.policy, <span className="font-mono">{snapshot.policy}</span>],
      [k.index, <span className="font-mono">{snapshot.index_mode}</span>],
      [k.pd, pdActive ? k.pdActive : snapshot.disaggregation ? k.pdPartial : k.pdOff],
      [k.blocks, fmtInt(snapshot.indexed_blocks)],
    ];
    return (
      <div ref={ref} className="relative z-10 border-[1.5px] border-ink bg-sheet">
        <div className="flex items-baseline justify-between gap-2 border-b border-ink px-3 pt-2 pb-1.5">
          <span className="cond-tight text-md font-bold">lev-gateway</span>
          <span className="cond text-2xs text-ink-3">+ router</span>
        </div>
        <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 px-3 py-2 text-xs">
          {rows.map(([label, value]) => (
            <div key={label} className="contents">
              <dt className="text-ink-3">{label}</dt>
              <dd className="min-w-0 text-right break-words text-ink">{value}</dd>
            </div>
          ))}
        </dl>
        <div className="border-t border-rule px-3 py-1.5 text-2xs text-ink-3 tabular-nums">
          <span className={cn("font-semibold", recentRoutes > 0 ? "text-route" : "text-ink-2")}>{fmtInt(recentRoutes)}</span> {k.routed}
        </div>
      </div>
    );
  },
);

function PoolFrame({ role, workers, active, children, t }: { role: Role; workers: WorkerView[]; active: boolean; children: ReactNode; t: Messages }) {
  const k = t.cluster.pool;
  const running = workers.reduce((sum, w) => sum + w.load.running, 0);
  const waiting = workers.reduce((sum, w) => sum + w.load.waiting, 0);
  const healthy = workers.filter((w) => w.healthy).length;
  return (
    <section aria-label={k[role]} className={cn("relative mt-2.5 border-[0.75px] px-3 pt-4 pb-3", active ? "border-ink-3" : "border-rule-2")}>
      <header className="mb-2 flex flex-wrap items-baseline justify-between gap-x-3 gap-y-0.5">
        <h3
          className={cn(
            "cond absolute -top-[11px] left-2.5 border-[0.75px] bg-paper px-2 text-sm leading-5 font-bold text-ink",
            active ? "border-ink-3" : "border-rule-2",
          )}
        >
          {k[role]}
        </h3>
        <span className="text-2xs text-ink-3">{k[NOTE_KEY[role]]}</span>
        <span className="text-2xs text-ink-3 tabular-nums">
          {k.stats(healthy, workers.length, running, waiting)}
          {active ? "" : k.inactive}
        </span>
      </header>
      <div className="flex flex-col gap-2">{children}</div>
    </section>
  );
}

interface WorkerBlockProps {
  worker: WorkerView;
  info: WorkerInfoEntry | undefined;
  dimmed: boolean;
  onHover: (id: string | null) => void;
  t: Messages;
}

const WorkerBlock = forwardRef<HTMLDivElement, WorkerBlockProps>(function WorkerBlock({ worker, info, dimmed, onHover, t }, ref) {
  const k = t.cluster.block;
  const line = workerLineState(worker, t);
  const reported = worker.load_age_ms !== null;
  const usedBlocks = worker.total_blocks > 0 && reported ? worker.total_blocks - worker.load.free_blocks : null;
  const profile = info?.labels.profile;
  const sourceTitle = worker.index_source === "events" ? k.events : worker.index_source === "shadow" ? k.shadow : k.indexOff;
  return (
    <div
      ref={ref}
      onPointerEnter={() => onHover(worker.id)}
      className={cn(
        "worker-card relative z-10 border bg-sheet px-3 pt-2 pb-2 transition-[opacity,border-color,box-shadow] duration-200",
        worker.healthy ? "border-ink" : "border-ctrl opacity-80",
        dimmed && "opacity-45",
      )}
    >
      <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1">
        <div className="flex min-w-0 items-center gap-2">
          <span className={cn("truncate font-mono text-sm font-semibold", !worker.healthy && "text-ink-3 line-through")}>{worker.id}</span>
          <RoleTag role={worker.role} />
          <StatusMark {...line} />
        </div>
        <span className="truncate text-2xs text-ink-3" title={worker.endpoint}>
          {info ? (
            <>
              <span className="font-mono text-ink-2">{info.backend}</span>
              {profile ? <span className="font-mono"> · {profile}</span> : null}
            </>
          ) : (
            <span className="font-mono">{worker.endpoint.replace(/^https?:\/\//, "")}</span>
          )}
        </span>
      </div>

      <div className="mt-1.5 flex items-center gap-3">
        <span className="cond w-6 text-2xs font-semibold text-ink-3">{k.kv}</span>
        <KvCells
          usage={reported ? worker.load.kv_usage : null}
          cached={worker.total_blocks > 0 && worker.index_source !== "off" ? worker.gpu_blocks / worker.total_blocks : null}
          className="flex-1"
        />
        <span className="w-12 text-right text-xs tabular-nums">{reported ? fmtPct(worker.load.kv_usage, worker.load.kv_usage < 0.1 ? 1 : 0) : "—"}</span>
      </div>
      <p className="mt-0.5 flex flex-wrap items-center gap-x-3 gap-y-0.5 pl-9 text-2xs text-ink-3 tabular-nums">
        <span className="inline-flex items-center gap-1 whitespace-nowrap">
          <span aria-hidden className="inline-block size-2 bg-ink" />
          {usedBlocks !== null ? k.used(fmtInt(usedBlocks), fmtInt(worker.total_blocks)) : k.total(fmtInt(worker.total_blocks))}
        </span>
        <span className="inline-flex items-center gap-1 whitespace-nowrap" title={k.indexViewTitle}>
          <span aria-hidden className="inline-block size-2 border border-ink-3 bg-ink-3/60" />
          {k.indexView(fmtInt(worker.gpu_blocks), fmtInt(worker.cpu_blocks))}
        </span>
      </p>

      <dl className="mt-1.5 grid grid-cols-3 gap-x-3 gap-y-1 border-t border-rule pt-1 text-xs tabular-nums xl:grid-cols-6">
        <Metric label={k.running} value={reported ? fmtInt(worker.load.running) : "—"} />
        <Metric label={k.waiting} value={reported ? fmtInt(worker.load.waiting) : "—"} />
        <Metric label={k.inflight} value={fmtInt(worker.inflight)} title={k.inflightTitle} />
        <Metric label={k.pending} value={`${fmtTokens(worker.pending_prefill_tokens)} tok`} title={k.pendingTitle} />
        <Metric label={k.dispatched} value={fmtInt(worker.dispatched)} />
        <Metric label={k.indexSource} value={worker.index_source} title={sourceTitle} />
      </dl>
    </div>
  );
});

/** The figure's key, set beside the diagram: line forms, the pulse, the cells. */
function FigureKey({ t, reducedMotion }: { t: Messages; reducedMotion: boolean }) {
  const k = t.cluster.legend;
  const line = (dash?: string, color = "var(--ink-3)") => (
    <svg width="26" height="8" aria-hidden className="shrink-0">
      <line x1="1" y1="4" x2="25" y2="4" stroke={color} strokeWidth="1.5" strokeDasharray={dash} />
    </svg>
  );
  const rows: [ReactNode, string][] = [
    [line(), k.used],
    [line("1 3", "var(--rule-2)"), k.unused],
    [
      <svg width="26" height="10" aria-hidden className="shrink-0">
        <line x1="1" y1="5" x2="25" y2="5" stroke="var(--rule-2)" strokeWidth="1.5" />
        <line x1="9" y1="9" x2="17" y2="1" stroke="var(--bad)" strokeWidth="1.5" />
      </svg>,
      k.down,
    ],
    [line("5 3"), k.kv],
    [<span aria-hidden className="mx-[9px] inline-block size-[7px] shrink-0 bg-route" />, k.pulse],
    [<span aria-hidden className="inline-block h-3 w-[26px] shrink-0 border-2 border-route" />, k.hit],
    [
      <span aria-hidden className="inline-flex w-[26px] shrink-0 gap-[2px]">
        <span className="h-2 flex-1 bg-ink" />
        <span className="h-2 flex-1 border border-ink-3 bg-ink-3/60" />
        <span className="h-2 flex-1 border border-rule-2" />
      </span>,
      k.cells,
    ],
  ];
  return (
    <section aria-label={k.title} className="flex flex-col gap-1.5 text-2xs leading-4 text-ink-2">
      <h3 className="cond text-xs font-bold text-ink">{k.title}</h3>
      <ul className="flex flex-col gap-1">
        {rows.map(([swatch, text]) => (
          <li key={text} className="flex items-center gap-2">
            {swatch}
            <span>{text}</span>
          </li>
        ))}
      </ul>
      <p className="mt-0.5 text-ink-3">
        {k.hover}
        {reducedMotion ? t.cluster.reducedMotion : ""}
      </p>
    </section>
  );
}

function Metric({ label, value, title }: { label: string; value: ReactNode; title?: string }) {
  return (
    <div className="min-w-0" title={title}>
      <dt className="cond truncate text-2xs text-ink-3">{label}</dt>
      <dd className="truncate text-ink">{value}</dd>
    </div>
  );
}
