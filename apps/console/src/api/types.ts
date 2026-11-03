// JSON shapes of the Leviathan gateway API, mirrored from the Rust sources:
//   crates/lev-router/src/lib.rs      RouterSnapshot, WorkerView, Plan, CandidateScore
//   crates/lev-gateway/src/api.rs     routes and ad-hoc JSON bodies
//   crates/lev-gateway/src/events.rs  LiveEvent (SSE payloads of /admin/live)
//   crates/lev-gateway/src/audit.rs   AuditEntry
//
// u64 seeds can exceed Number.MAX_SAFE_INTEGER, so the client keeps them as
// decimal strings (see `parseGatewayJson`).

export type Role = "unified" | "prefill" | "decode";
export type Policy = "round_robin" | "least_load" | "p2c" | "cache_aware";
export type IndexMode = "exact" | "approximate" | "off";
export type IndexSource = "events" | "shadow" | "off";
export type ProtocolName = "chat_completions" | "responses" | "messages";
export type Outcome = "ok" | "error" | "rejected" | "cancelled";

export const POLICIES: readonly Policy[] = ["round_robin", "least_load", "p2c", "cache_aware"];
export const INDEX_MODES: readonly IndexMode[] = ["exact", "approximate", "off"];
export const PROTOCOLS: readonly ProtocolName[] = ["chat_completions", "responses", "messages"];
export const OUTCOMES: readonly Outcome[] = ["ok", "error", "rejected", "cancelled"];

/** `WorkerLoad`: the latest load report of a worker. */
export interface WorkerLoad {
  running: number;
  waiting: number;
  kv_usage: number;
  free_blocks: number;
  prefill_tokens_per_s: number;
  decode_tokens_per_s: number;
  spec_accept_rate: number;
}

/** `WorkerView`: a worker as the router sees it. */
export interface WorkerView {
  id: string;
  endpoint: string;
  model: string;
  role: Role;
  healthy: boolean;
  block_size: number;
  total_blocks: number;
  index_source: IndexSource;
  load: WorkerLoad;
  /** Milliseconds since the last load report; null before the first one. */
  load_age_ms: number | null;
  inflight: number;
  pending_prefill_tokens: number;
  dispatched: number;
  gpu_blocks: number;
  cpu_blocks: number;
}

/** `RouterSnapshot`: GET /admin/router and the body of PUT /admin/router. */
export interface RouterSnapshot {
  policy: Policy;
  index_mode: IndexMode;
  disaggregation: boolean;
  session_affinity: boolean;
  indexed_blocks: number;
  workers: WorkerView[];
}

/** Body of PUT /admin/router. */
export interface RouterUpdate {
  policy?: Policy;
  index_mode?: IndexMode;
  disaggregation?: boolean;
}

/** `Target`: a worker chosen for one stage of a request. */
export interface Target {
  worker_id: string;
  endpoint: string;
  cached_tokens: number;
}

/** `Plan`, serialized with `#[serde(tag = "kind")]`. */
export type Plan =
  | { kind: "unified"; worker: Target }
  | { kind: "disaggregated"; prefill: Target; decode: Target };

/** `CandidateScore`: how one candidate scored for the prefill/unified stage. */
export interface CandidateScore {
  worker_id: string;
  role: Role;
  cached_tokens: number;
  sequences: number;
  kv_usage: number;
  /** Estimated time to first token, microseconds. */
  est_ttft_us: number;
  chosen: boolean;
}

/** GET /admin/overview */
export interface Overview {
  version: string;
  uptime_s: number;
  fingerprint: string;
  models: string[];
  workers: number;
  healthy_workers: number;
  policy: Policy;
  index_mode: IndexMode;
  disaggregation: boolean;
  indexed_blocks: number;
}

/** One entry of GET /admin/workers. */
export interface WorkerInfoEntry {
  id: string;
  endpoint: string;
  backend: string;
  model: string;
  role: Role | null;
  block_size: number;
  context_limit: number;
  total_blocks: number;
  supports_kv_events: boolean;
  deterministic: boolean;
  labels: Record<string, string>;
  router: WorkerView | null;
}

/** GET /admin/workers */
export interface WorkersResponse {
  /** Every endpoint the gateway supervises, connected or not. */
  endpoints: string[];
  /** Workers that answered GetInfo. */
  workers: WorkerInfoEntry[];
}

/** `AuditEntry` (fields with `#[serde(skip)]` are not present). */
export interface AuditEntry {
  request_id: string;
  response_id: string;
  protocol: ProtocolName;
  model: string;
  tenant: string;
  received_at_ms: number;
  policy: string | null;
  plan: Plan | null;
  worker_id: string | null;
  prefill_worker_id: string | null;
  attempts: number;
  prompt_tokens: number;
  cached_tokens: number;
  completion_tokens: number;
  /** u64, kept as a decimal string. */
  seed: string;
  ttft_ms: number | null;
  tpot_ms: number | null;
  e2e_ms: number;
  outcome: Outcome;
  finish_reason: string | null;
  error: string | null;
  spec_draft_tokens: number;
  spec_accepted_tokens: number;
  /** Stages measured from arrival; absent in entries from older gateways. */
  timeline?: TimelineStage[];
}

/** One step of a request's timeline (`pipeline::Stage`), relative to arrival. */
export interface TimelineStage {
  /** auth, parse, convert, render, tokenize, dispatch, worker.queue, worker.prefill, decode, respond */
  name: string;
  start_ms: number;
  duration_ms: number;
}

/** GET /admin/requests */
export interface RequestsResponse {
  requests: AuditEntry[];
}

/** GET /admin/requests/{id} */
export interface RequestDetail {
  request: AuditEntry;
  prompt_token_ids: number[];
  output_token_ids: number[];
  output_text: string | null;
}

/** POST /admin/requests/{id}/replay */
export interface ReplayResult {
  request_id: string;
  replay_id: string;
  original_worker: string | null;
  replay_worker: string;
  seed: string;
  identical: boolean;
  /** Index of the first output token that differs; null when identical. */
  first_divergence: number | null;
  original_tokens: number;
  replay_tokens: number;
  /** e.g. `FINISH_REASON_STOP`. */
  finish_reason: string | null;
  original_text: string | null;
  replay_text: string | null;
  original_token_ids?: number[];
  replay_token_ids?: number[];
  original_tokens_view?: RenderToken[] | null;
  replay_tokens_view?: RenderToken[] | null;
}

/** POST /v1/leviathan/detokenize */
export interface DetokenizeResult {
  model: string;
  text: string;
  tokens: RenderToken[];
}

/** GET /admin/stats?window_s=N: KPIs over requests finished in the window, from the audit log. */
export interface StatsResult {
  window_s: number;
  n: number;
  source: string;
  requests_per_s: number;
  outcomes: Partial<Record<Outcome, number>>;
  ttft_ms: { p50: number | null; p90: number | null; p99: number | null };
  tpot_ms: { p50: number | null; p90: number | null };
  e2e_ms: { p50: number | null; p99: number | null };
  prompt_tokens: number;
  cached_tokens: number;
  cache_hit_rate: number | null;
  output_tokens_per_s: number;
  spec_accept_rate: number | null;
  per_worker: Record<string, number>;
  per_policy: Record<string, number>;
}

/** One token of the Prompt Inspector. */
export interface RenderToken {
  id: number;
  text: string;
  special: boolean;
}

/** POST /v1/leviathan/render */
export interface RenderResult {
  protocol: ProtocolName;
  model: string;
  thinking_mode: boolean;
  stream: boolean;
  prompt: string;
  /** Byte ranges of control tokens in `prompt` (UTF-8 offsets). */
  special_ranges: { start: number; end: number; text: string }[];
  prompt_tokens: number;
  context_limit: number;
  max_tokens: number;
  temperature: number;
  top_p: number;
  seed: string;
  constraint: { kind: string; schema: string } | null;
  hints: Record<string, string>;
  block_size: number;
  block_hashes: string[];
  tokens: RenderToken[];
  /** Gateway-side stages of the render. */
  timeline?: TimelineStage[];
}

/** GET /v1/models */
export interface ModelsResponse {
  object: "list";
  data: {
    id: string;
    object: "model";
    owned_by: string;
    aliases: string[];
    context_length: number;
    healthy_workers: number;
  }[];
}

// ------------------------------------------------------------ live events

export interface RouteEvent {
  type: "route";
  request_id: string;
  protocol: ProtocolName;
  model: string;
  policy: Policy;
  plan: Plan;
  prompt_tokens: number;
  candidates: CandidateScore[];
  affinity: boolean;
  attempt: number;
  at_ms: number;
}

export interface RequestDoneEvent {
  type: "request_done";
  request_id: string;
  model: string;
  tenant: string;
  worker_id: string | null;
  outcome: Outcome;
  prompt_tokens: number;
  cached_tokens: number;
  completion_tokens: number;
  ttft_ms: number | null;
  e2e_ms: number;
  at_ms: number;
  // Added in 2f9cc67; optional so that older gateways still parse.
  protocol?: ProtocolName;
  tpot_ms?: number | null;
  spec_draft_tokens?: number;
  spec_accepted_tokens?: number;
  finish_reason?: string | null;
  prefill_worker_id?: string | null;
  policy?: string | null;
}

/** A stage of a disaggregated request finished. */
export interface StageEvent {
  type: "stage";
  request_id: string;
  /** `prefill_done`: the prefill worker returned its KV hand-off; `decode_ready`: the decode worker loaded it. */
  stage: "prefill_done" | "decode_ready";
  worker_id: string;
  duration_ms: number;
  transferred_tokens: number;
  at_ms: number;
}

export interface WorkerEvent {
  type: "worker";
  worker_id: string;
  endpoint: string;
  /** `healthy` or `unhealthy`. */
  status: string;
  detail: string;
  at_ms: number;
}

export type LiveEvent = RouteEvent | RequestDoneEvent | StageEvent | WorkerEvent;

// ------------------------------------------------------------ control plane

/** `lev_control::Tenant` (limits flattened; null means unlimited). */
export interface Tenant {
  id: string;
  name: string;
  rpm: number | null;
  tpm: number | null;
  monthly_token_quota: number | null;
  disabled: boolean;
}

/** `lev_control::UsageTotals` */
export interface UsageTotals {
  requests: number;
  prompt_tokens: number;
  cached_tokens: number;
  completion_tokens: number;
}

/** GET /admin/tenants */
export interface TenantsResponse {
  tenants: { tenant: Tenant; usage_this_month: UsageTotals }[];
}

/** Body of POST /admin/tenants. */
export interface TenantUpsert {
  id: string;
  name: string;
  rpm?: number | null;
  tpm?: number | null;
  monthly_token_quota?: number | null;
  disabled?: boolean;
}

/** `lev_control::ApiKeyInfo`: never the secret. */
export interface ApiKeyInfo {
  id: string;
  tenant_id: string;
  name: string;
  /** Displayable start of the key, e.g. `lev-3fA9xQ…`. */
  prefix: string;
  created_at: string;
  revoked_at: string | null;
}

/** 201 from POST /admin/tenants/{id}/keys: the only response with the secret. */
export interface CreatedKey extends ApiKeyInfo {
  key: string;
}

/** `lev_control::DailyUsage` (totals flattened). */
export interface DailyUsage extends UsageTotals {
  /** UTC day, `YYYY-MM-DD`. */
  day: string;
  tenant_id: string;
  model: string;
}

/** OpenAI error body: `{"error": {"message", "type", "param", "code"}}`. */
export interface OpenAiErrorBody {
  error: { message: string; type: string; param: unknown; code: string | null };
}

/** Anthropic error body: `{"type": "error", "error": {"type", "message"}}`. */
export interface AnthropicErrorBody {
  type: "error";
  error: { type: string; message: string };
}
