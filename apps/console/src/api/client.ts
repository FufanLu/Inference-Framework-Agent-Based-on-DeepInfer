import { getT } from "@/i18n";
import { getSettings } from "@/state/settings";
import type {
  AnthropicErrorBody,
  ApiKeyInfo,
  CreatedKey,
  DailyUsage,
  DetokenizeResult,
  ModelsResponse,
  OpenAiErrorBody,
  Overview,
  ProtocolName,
  RenderResult,
  ReplayResult,
  RequestDetail,
  RequestsResponse,
  RouterSnapshot,
  RouterUpdate,
  StatsResult,
  Tenant,
  TenantsResponse,
  TenantUpsert,
  WorkersResponse,
} from "./types";

/** Which credential a request carries. */
export type AuthKind = "admin" | "api" | "none";

export type GatewayErrorKind = "network" | "http" | "parse" | "aborted";

/** A failed gateway call, with enough detail to explain it in the UI. */
export class GatewayError extends Error {
  readonly kind: GatewayErrorKind;
  readonly status: number | null;
  /** Machine-readable `error.code` (OpenAI) or `error.type` (Anthropic). */
  readonly code: string | null;
  readonly url: string;
  /** Seconds from the `Retry-After` header of a 429. */
  readonly retryAfter: number | null;

  constructor(
    kind: GatewayErrorKind,
    message: string,
    url: string,
    status: number | null = null,
    code: string | null = null,
    retryAfter: number | null = null,
  ) {
    super(message);
    this.name = "GatewayError";
    this.kind = kind;
    this.status = status;
    this.code = code;
    this.url = url;
    this.retryAfter = retryAfter;
  }

  get unauthorized() {
    return this.status === 401;
  }

  get noWorkers() {
    return this.status === 503;
  }
}

export function isGatewayError(error: unknown): error is GatewayError {
  return error instanceof GatewayError;
}

/**
 * JSON.parse that keeps u64 `seed` values exact: they are quoted before
 * parsing, so every `seed` field arrives as a decimal string.
 */
export function parseGatewayJson<T>(text: string): T {
  const quoted = text.replace(/("seed"\s*:\s*)(\d+)/g, '$1"$2"');
  return JSON.parse(quoted) as T;
}

function authHeader(kind: AuthKind): Record<string, string> {
  const { adminToken, apiKey } = getSettings();
  const token = kind === "admin" ? adminToken : kind === "api" ? apiKey : "";
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export function gatewayUrl(path: string): string {
  return `${getSettings().gatewayUrl}${path}`;
}

/** Seconds from a `Retry-After` header (delta-seconds or an HTTP date). */
export function parseRetryAfter(value: string | null): number | null {
  if (!value) return null;
  const seconds = Number(value);
  if (Number.isFinite(seconds)) return Math.max(0, Math.ceil(seconds));
  const date = Date.parse(value);
  return Number.isNaN(date) ? null : Math.max(0, Math.ceil((date - Date.now()) / 1000));
}

/** Pull a readable message out of an error body in any protocol's format. */
export function errorFromBody(text: string): { message: string; code: string | null } {
  try {
    const body = JSON.parse(text) as Partial<OpenAiErrorBody & AnthropicErrorBody>;
    if (body.error && typeof body.error === "object") {
      const error = body.error as { message?: string; code?: string | null; type?: string };
      return { message: error.message ?? text, code: error.code ?? error.type ?? null };
    }
  } catch {
    // Not JSON: plain-text bodies such as /readyz.
  }
  return { message: text.trim() || getT().states.emptyBody, code: null };
}

interface RequestOptions {
  method?: "GET" | "POST" | "PUT" | "DELETE";
  body?: unknown;
  auth?: AuthKind;
  signal?: AbortSignal;
}

async function send(path: string, options: RequestOptions): Promise<Response> {
  const url = gatewayUrl(path);
  const headers: Record<string, string> = { Accept: "application/json", ...authHeader(options.auth ?? "none") };
  if (options.body !== undefined) headers["Content-Type"] = "application/json";
  try {
    return await fetch(url, {
      method: options.method ?? "GET",
      headers,
      body: options.body === undefined ? undefined : JSON.stringify(options.body),
      signal: options.signal,
      cache: "no-store",
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new GatewayError("aborted", getT().states.aborted, url);
    }
    throw new GatewayError("network", getT().states.networkError(getSettings().gatewayUrl), url);
  }
}

async function failure(response: Response): Promise<GatewayError> {
  const { message, code } = errorFromBody(await response.text());
  return new GatewayError("http", message, response.url, response.status, code, parseRetryAfter(response.headers.get("retry-after")));
}

async function requestJson<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const response = await send(path, options);
  if (!response.ok) throw await failure(response);
  const text = await response.text();
  try {
    return parseGatewayJson<T>(text);
  } catch {
    throw new GatewayError("parse", getT().states.badJson, response.url, response.status);
  }
}

async function requestEmpty(path: string, options: RequestOptions = {}): Promise<void> {
  const response = await send(path, options);
  if (!response.ok) throw await failure(response);
}

async function requestText(path: string, options: RequestOptions = {}): Promise<{ ok: boolean; status: number; text: string }> {
  const response = await send(path, options);
  return { ok: response.ok, status: response.status, text: await response.text() };
}

function query(params: Record<string, string | number | null | undefined>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) if (value !== null && value !== undefined && value !== "") search.set(key, String(value));
  const text = search.toString();
  return text ? `?${text}` : "";
}

export const api = {
  // --------------------------------------------------------------- admin
  overview: (signal?: AbortSignal) => requestJson<Overview>("/admin/overview", { auth: "admin", signal }),
  stats: (windowS: number, signal?: AbortSignal) =>
    requestJson<StatsResult>(`/admin/stats${query({ window_s: windowS })}`, { auth: "admin", signal }),
  router: (signal?: AbortSignal) => requestJson<RouterSnapshot>("/admin/router", { auth: "admin", signal }),
  updateRouter: (update: RouterUpdate) =>
    requestJson<RouterSnapshot>("/admin/router", { method: "PUT", body: update, auth: "admin" }),
  workers: (signal?: AbortSignal) => requestJson<WorkersResponse>("/admin/workers", { auth: "admin", signal }),
  addWorker: (endpoint: string) =>
    requestJson<{ endpoint: string }>("/admin/workers", { method: "POST", body: { endpoint }, auth: "admin" }),
  requests: (limit: number, signal?: AbortSignal) =>
    requestJson<RequestsResponse>(`/admin/requests${query({ limit })}`, { auth: "admin", signal }),
  request: (id: string, signal?: AbortSignal) =>
    requestJson<RequestDetail>(`/admin/requests/${encodeURIComponent(id)}`, { auth: "admin", signal }),
  replay: (id: string) =>
    requestJson<ReplayResult>(`/admin/requests/${encodeURIComponent(id)}/replay`, { method: "POST", auth: "admin" }),

  // ------------------------------------------------------- control plane
  tenants: (signal?: AbortSignal) => requestJson<TenantsResponse>("/admin/tenants", { auth: "admin", signal }),
  upsertTenant: (tenant: TenantUpsert) =>
    requestJson<{ tenant: Tenant }>("/admin/tenants", { method: "POST", body: tenant, auth: "admin" }),
  createKey: (tenantId: string, name: string) =>
    requestJson<CreatedKey>(`/admin/tenants/${encodeURIComponent(tenantId)}/keys`, { method: "POST", body: { name }, auth: "admin" }),
  keys: (tenant: string | null, signal?: AbortSignal) =>
    requestJson<{ keys: ApiKeyInfo[] }>(`/admin/keys${query({ tenant })}`, { auth: "admin", signal }),
  revokeKey: (id: string) => requestEmpty(`/admin/keys/${encodeURIComponent(id)}`, { method: "DELETE", auth: "admin" }),
  usage: (tenant: string | null, days: number, signal?: AbortSignal) =>
    requestJson<{ usage: DailyUsage[] }>(`/admin/usage${query({ tenant, days })}`, { auth: "admin", signal }),

  // -------------------------------------------------------------- public
  models: (signal?: AbortSignal) => requestJson<ModelsResponse>("/v1/models", { auth: "api", signal }),
  render: (protocol: ProtocolName, body: unknown, signal?: AbortSignal) =>
    requestJson<RenderResult>(`/v1/leviathan/render${query({ protocol })}`, { method: "POST", body, auth: "api", signal }),
  detokenize: (ids: number[], signal?: AbortSignal) =>
    requestJson<DetokenizeResult>("/v1/leviathan/detokenize", { method: "POST", body: { ids }, auth: "api", signal }),
  healthz: (signal?: AbortSignal) => requestText("/healthz", { signal }),
  readyz: (signal?: AbortSignal) => requestText("/readyz", { signal }),
};

// ------------------------------------------------------------- streaming

/** One server-sent event frame as it arrived on the wire. */
export interface SseFrame {
  /** `event:` field, when the protocol names its events (Responses, Messages). */
  event: string | null;
  data: string;
  /** Milliseconds since the request was sent. */
  t: number;
}

/**
 * Incremental SSE parser: feed decoded text, get complete frames back.
 * Comment lines (`: heartbeat`) are reported through `onComment`.
 */
export class SseParser {
  private buffer = "";
  private readonly onFrame: (event: string | null, data: string) => void;
  private readonly onComment: (() => void) | undefined;

  constructor(onFrame: (event: string | null, data: string) => void, onComment?: () => void) {
    this.onFrame = onFrame;
    this.onComment = onComment;
  }

  push(chunk: string) {
    this.buffer += chunk.replace(/\r\n?/g, "\n");
    let boundary = this.buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const block = this.buffer.slice(0, boundary);
      this.buffer = this.buffer.slice(boundary + 2);
      this.parseBlock(block);
      boundary = this.buffer.indexOf("\n\n");
    }
  }

  private parseBlock(block: string) {
    let event: string | null = null;
    const data: string[] = [];
    let comment = false;
    for (const line of block.split("\n")) {
      if (line.startsWith(":")) {
        comment = true;
        continue;
      }
      const colon = line.indexOf(":");
      const field = colon === -1 ? line : line.slice(0, colon);
      let value = colon === -1 ? "" : line.slice(colon + 1);
      if (value.startsWith(" ")) value = value.slice(1);
      if (field === "event") event = value;
      else if (field === "data") data.push(value);
    }
    if (data.length > 0) this.onFrame(event, data.join("\n"));
    else if (comment) this.onComment?.();
  }
}

export interface StreamError {
  message: string;
  code: string | null;
  /** Seconds from `Retry-After`, on a 429. */
  retryAfter: number | null;
}

export interface StreamResult {
  status: number;
  requestId: string | null;
  /** Error when the gateway refused the request before streaming. */
  error: StreamError | null;
  /** Wall time until the response headers arrived. */
  headersMs: number;
}

/**
 * POST a streaming inference request and report every SSE frame. Resolves
 * when the stream ends; rejects with GatewayError on transport failure.
 */
export async function streamInference(
  path: string,
  body: unknown,
  onFrame: (frame: SseFrame) => void,
  signal?: AbortSignal,
): Promise<StreamResult> {
  const started = performance.now();
  const response = await send(path, { method: "POST", body, auth: "api", signal });
  const headersMs = performance.now() - started;
  const requestId = response.headers.get("x-request-id");
  if (!response.ok || !response.body) {
    const text = await response.text();
    return {
      status: response.status,
      requestId,
      error: { ...errorFromBody(text), retryAfter: parseRetryAfter(response.headers.get("retry-after")) },
      headersMs,
    };
  }
  const parser = new SseParser((event, data) => onFrame({ event, data, t: performance.now() - started }));
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      parser.push(value);
    }
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new GatewayError("aborted", getT().states.aborted, response.url);
    }
    throw new GatewayError("network", getT().states.streamBroken, response.url);
  } finally {
    reader.releaseLock();
  }
  return { status: response.status, requestId, error: null, headersMs };
}

export interface ProbeResult {
  reachable: boolean;
  health: string | null;
  admin: "ok" | "unauthorized" | "error" | "unreachable";
  adminDetail: string | null;
  version: string | null;
}

/** Check a gateway URL and admin token before saving them. */
export async function probeGateway(baseUrl: string, adminToken: string): Promise<ProbeResult> {
  const base = baseUrl.trim().replace(/\/+$/, "");
  const result: ProbeResult = { reachable: false, health: null, admin: "unreachable", adminDetail: null, version: null };
  try {
    const health = await fetch(`${base}/healthz`, { cache: "no-store", signal: AbortSignal.timeout(4000) });
    result.reachable = true;
    result.health = health.ok ? (await health.text()).trim() : `HTTP ${health.status}`;
  } catch {
    return result;
  }
  try {
    const overview = await fetch(`${base}/admin/overview`, {
      cache: "no-store",
      headers: adminToken ? { Authorization: `Bearer ${adminToken}` } : {},
      signal: AbortSignal.timeout(4000),
    });
    if (overview.status === 401) {
      result.admin = "unauthorized";
      result.adminDetail = errorFromBody(await overview.text()).message;
    } else if (overview.ok) {
      result.admin = "ok";
      const body = parseGatewayJson<{ version?: string }>(await overview.text());
      result.version = body.version ?? null;
    } else {
      result.admin = "error";
      result.adminDetail = `HTTP ${overview.status}`;
    }
  } catch {
    result.admin = "error";
    result.adminDetail = "—";
  }
  return result;
}
