import { useSyncExternalStore } from "react";
import { getSettings, subscribeSettings } from "@/state/settings";
import { SseParser, gatewayUrl } from "./client";
import type { LiveEvent, RequestDoneEvent, RouteEvent, StageEvent, WorkerEvent } from "./types";

/**
 * One shared connection to `GET /admin/live`.
 *
 * EventSource cannot send an Authorization header, so the stream is read with
 * fetch. The hub reconnects with exponential backoff, treats 45 s without a
 * byte as a dead connection (the gateway sends a heartbeat every 15 s), and
 * stops retrying on 401 until the settings change.
 */

export type LiveStatus = "connecting" | "open" | "retrying" | "unauthorized" | "stopped";

/** Why the last connection failed; translated in the UI. */
export type LiveReason =
  | { kind: "network" }
  | { kind: "unauthorized" }
  | { kind: "closed" }
  | { kind: "heartbeat" }
  | { kind: "http"; status: number };

export interface LiveState {
  status: LiveStatus;
  /** Why the last connection failed. */
  reason: LiveReason | null;
  /** When the next attempt starts (epoch ms), while retrying. */
  retryAt: number | null;
  /** When the current connection opened (epoch ms). */
  openedAt: number | null;
  /** When the hub first connected in this page view (epoch ms). */
  firstOpenedAt: number | null;
  /** Bytes or heartbeats seen last (epoch ms). */
  lastSeenAt: number | null;
  attempts: number;
}

export interface LiveData {
  /** Routing decisions, newest last. */
  routes: RouteEvent[];
  /** Completed requests, newest last. */
  done: RequestDoneEvent[];
  /** Disaggregated stage reports, newest last. */
  stages: StageEvent[];
  /** Worker health changes, newest last. */
  workers: WorkerEvent[];
  /** Total events received in this page view. */
  received: number;
}

const ROUTE_CAP = 400;
const DONE_CAP = 6000;
const WORKER_CAP = 120;
const STAGE_CAP = 800;
const WATCHDOG_MS = 45_000;
const SNAPSHOT_INTERVAL_MS = 250;

type Listener = () => void;
type EventListener = (event: LiveEvent) => void;

class LiveHub {
  private state: LiveState = {
    status: "stopped",
    reason: null,
    retryAt: null,
    openedAt: null,
    firstOpenedAt: null,
    lastSeenAt: null,
    attempts: 0,
  };
  private routes: RouteEvent[] = [];
  private done: RequestDoneEvent[] = [];
  private workerEvents: WorkerEvent[] = [];
  private stages: StageEvent[] = [];
  private received = 0;
  private data: LiveData = { routes: [], done: [], stages: [], workers: [], received: 0 };

  private controller: AbortController | null = null;
  private generation = 0;
  private retryTimer: ReturnType<typeof setTimeout> | null = null;
  private watchdog: ReturnType<typeof setTimeout> | null = null;
  private snapshotTimer: ReturnType<typeof setTimeout> | null = null;
  private started = false;
  private connectionKey = "";

  private readonly stateListeners = new Set<Listener>();
  private readonly dataListeners = new Set<Listener>();
  private readonly eventListeners = new Set<EventListener>();

  start() {
    if (this.started) return;
    this.started = true;
    this.connectionKey = keyOf();
    subscribeSettings(() => {
      const key = keyOf();
      if (key !== this.connectionKey) {
        this.connectionKey = key;
        this.restart();
      }
    });
    this.connect();
  }

  /** Reconnect now, e.g. after the settings changed or on user request. */
  restart() {
    this.teardown();
    this.state = { ...this.state, attempts: 0 };
    this.connect();
  }

  private teardown() {
    this.generation += 1;
    this.controller?.abort();
    this.controller = null;
    if (this.retryTimer) clearTimeout(this.retryTimer);
    if (this.watchdog) clearTimeout(this.watchdog);
    this.retryTimer = null;
    this.watchdog = null;
  }

  private setState(patch: Partial<LiveState>) {
    this.state = { ...this.state, ...patch };
    for (const listener of this.stateListeners) listener();
  }

  private async connect() {
    const generation = ++this.generation;
    const controller = new AbortController();
    this.controller = controller;
    this.setState({ status: "connecting", retryAt: null });

    const headers: Record<string, string> = { Accept: "text/event-stream" };
    const { adminToken } = getSettings();
    if (adminToken) headers.Authorization = `Bearer ${adminToken}`;

    let response: Response;
    try {
      response = await fetch(gatewayUrl("/admin/live"), { headers, signal: controller.signal, cache: "no-store" });
    } catch {
      if (generation === this.generation) this.scheduleRetry({ kind: "network" });
      return;
    }
    if (generation !== this.generation) return;
    if (response.status === 401) {
      this.setState({ status: "unauthorized", reason: { kind: "unauthorized" }, openedAt: null });
      return;
    }
    if (!response.ok || !response.body) {
      this.scheduleRetry({ kind: "http", status: response.status });
      return;
    }

    const now = Date.now();
    this.setState({
      status: "open",
      reason: null,
      openedAt: now,
      firstOpenedAt: this.state.firstOpenedAt ?? now,
      lastSeenAt: now,
      attempts: 0,
    });
    this.armWatchdog(generation, controller);

    const parser = new SseParser(
      (_event, data) => this.handle(data),
      () => this.setState({ lastSeenAt: Date.now() }),
    );
    const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
    try {
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        if (generation !== this.generation) return;
        this.armWatchdog(generation, controller);
        parser.push(value);
      }
    } catch {
      // Aborted by teardown or the watchdog, or the connection dropped.
    }
    if (generation === this.generation) this.scheduleRetry({ kind: "closed" });
  }

  private armWatchdog(generation: number, controller: AbortController) {
    if (this.watchdog) clearTimeout(this.watchdog);
    this.watchdog = setTimeout(() => {
      if (generation !== this.generation) return;
      controller.abort();
      this.scheduleRetry({ kind: "heartbeat" });
    }, WATCHDOG_MS);
  }

  private scheduleRetry(reason: LiveReason) {
    if (this.retryTimer) return;
    this.generation += 1;
    this.controller?.abort();
    if (this.watchdog) clearTimeout(this.watchdog);
    this.watchdog = null;
    const attempts = this.state.attempts + 1;
    const base = Math.min(15_000, 1_000 * 2 ** (attempts - 1));
    const delay = Math.round(base * (0.8 + Math.random() * 0.4));
    this.setState({ status: "retrying", reason, retryAt: Date.now() + delay, openedAt: null, attempts });
    this.retryTimer = setTimeout(() => {
      this.retryTimer = null;
      void this.connect();
    }, delay);
  }

  private handle(data: string) {
    let event: LiveEvent;
    try {
      event = JSON.parse(data) as LiveEvent;
    } catch {
      return;
    }
    this.received += 1;
    this.state = { ...this.state, lastSeenAt: Date.now() };
    switch (event.type) {
      case "route":
        push(this.routes, event, ROUTE_CAP);
        break;
      case "request_done":
        push(this.done, event, DONE_CAP);
        break;
      case "stage":
        push(this.stages, event, STAGE_CAP);
        break;
      case "worker":
        push(this.workerEvents, event, WORKER_CAP);
        break;
      default:
        return;
    }
    for (const listener of this.eventListeners) listener(event);
    this.scheduleSnapshot();
  }

  private scheduleSnapshot() {
    if (this.snapshotTimer) return;
    this.snapshotTimer = setTimeout(() => {
      this.snapshotTimer = null;
      this.data = {
        routes: this.routes.slice(),
        done: this.done.slice(),
        stages: this.stages.slice(),
        workers: this.workerEvents.slice(),
        received: this.received,
      };
      for (const listener of this.dataListeners) listener();
    }, SNAPSHOT_INTERVAL_MS);
  }

  // ------------------------------------------------------- subscriptions

  getState = () => this.state;
  getData = () => this.data;

  subscribeState = (listener: Listener) => {
    this.stateListeners.add(listener);
    return () => {
      this.stateListeners.delete(listener);
    };
  };

  subscribeData = (listener: Listener) => {
    this.dataListeners.add(listener);
    return () => {
      this.dataListeners.delete(listener);
    };
  };

  /** Every event as it arrives, unthrottled (for animation). */
  onEvent(listener: EventListener) {
    this.eventListeners.add(listener);
    return () => {
      this.eventListeners.delete(listener);
    };
  }
}

function push<T>(list: T[], item: T, cap: number) {
  list.push(item);
  if (list.length > cap) list.splice(0, list.length - cap);
}

function keyOf() {
  const { gatewayUrl: url, adminToken } = getSettings();
  return `${url}\u0000${adminToken}`;
}

export const liveHub = new LiveHub();

export function useLiveState(): LiveState {
  return useSyncExternalStore(liveHub.subscribeState, liveHub.getState, liveHub.getState);
}

export function useLiveData(): LiveData {
  return useSyncExternalStore(liveHub.subscribeData, liveHub.getData, liveHub.getData);
}
