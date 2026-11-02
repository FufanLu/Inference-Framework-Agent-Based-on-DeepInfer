//! The request path: protocol request → recipe conversion → V4.1 prompt →
//! token IDs → router → worker(s) → token stream → recipe stream processor →
//! protocol response (SSE or JSON).

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::Json;
use axum::body::Body;
use axum::http::{HeaderMap, HeaderValue, header};
use axum::response::{IntoResponse, Response};
use deepseek_recipe::request::{ConversationRequest, OutputConstraint, ProtocolRequest};
use deepseek_recipe::response::{Accumulate, ProtocolResponse};
use deepseek_recipe::stream::{
    ChunkGenerator, InferenceChunk, InferenceFinishReason, InferenceTokenLogprob, PromptUsage,
    StreamError, StreamProcessor,
};
use deepseek_recipe_core::messages::InputMessage;
use futures::{Stream, StreamExt};
use lev_proto::pb;
use lev_router::{Plan, Policy, RouteDecision, RouteRequest, Ticket};
use parking_lot::Mutex;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tonic::{Code, Status, Streaming};
use uuid::Uuid;

use axum::http::StatusCode;
use chrono::Utc;
use lev_control::{AuthError, Tenant, Throttle, UsageRecord};

use crate::audit::{AuditEntry, ReplayInputs};
use crate::error::{ApiError, ErrorKind};
use crate::events::{LiveEvent, now_ms};
use crate::model::ServedModel;
use crate::protocol::ProtocolInfo;
use crate::state::AppState;

/// The response chunk type of protocol request `T`.
pub type ProtocolChunk<T> =
    <<<T as ProtocolRequest>::Response as ProtocolResponse>::ChunkGenerator as ChunkGenerator>::Chunk;

/// One step of a request's timeline, relative to its arrival.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Stage {
    pub name: &'static str,
    pub start_ms: f64,
    pub duration_ms: f64,
}

fn stage(name: &'static str, origin: Instant, start: Instant, end: Instant) -> Stage {
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    Stage {
        name,
        start_ms: ms(start.saturating_duration_since(origin)),
        duration_ms: ms(end.saturating_duration_since(start)),
    }
}

/// Fields read from the raw body before protocol parsing.
#[derive(Debug, Default, Deserialize)]
struct Envelope {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    user: Option<String>,
    #[serde(default)]
    prompt_cache_key: Option<String>,
    #[serde(default)]
    metadata: Option<serde_json::Value>,
}

impl Envelope {
    fn session_key(&self, headers: &HeaderMap) -> Option<String> {
        headers
            .get("x-session-id")
            .and_then(|value| value.to_str().ok())
            .map(str::to_string)
            .or_else(|| self.prompt_cache_key.clone())
            .or_else(|| self.user.clone())
            .or_else(|| {
                self.metadata
                    .as_ref()
                    .and_then(|metadata| metadata.get("user_id"))
                    .and_then(|user| user.as_str())
                    .map(str::to_string)
            })
    }
}

/// Resolve the tenant of a request from its API key
/// (`Authorization: Bearer …` or Anthropic's `x-api-key`).
pub async fn authenticate(state: &AppState, headers: &HeaderMap) -> Result<Tenant, ApiError> {
    let key = headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .or_else(|| {
            headers
                .get("x-api-key")
                .and_then(|value| value.to_str().ok())
        });
    state.control.authenticate(key).await.map_err(|error| {
        let (status, kind, code) = match &error {
            AuthError::Missing | AuthError::Invalid => (
                StatusCode::UNAUTHORIZED,
                ErrorKind::Authentication,
                "invalid_api_key",
            ),
            AuthError::Disabled(_) => (
                StatusCode::FORBIDDEN,
                ErrorKind::PermissionDenied,
                "tenant_disabled",
            ),
            AuthError::Unavailable(_) => (
                StatusCode::SERVICE_UNAVAILABLE,
                ErrorKind::Overloaded,
                "control_plane_unavailable",
            ),
        };
        ApiError::new(status, kind, code, error.to_string())
    })
}

/// Admit a request under its tenant's rate limits and quota.
pub async fn admit(state: &AppState, tenant: &Tenant) -> Result<(), ApiError> {
    state.control.admit(tenant).await.map_err(|throttle| {
        metrics::counter!(
            "lev_throttled_total",
            "tenant" => tenant.id.clone(),
            "reason" => match throttle {
                Throttle::Requests { .. } => "rpm",
                Throttle::Tokens { .. } => "tpm",
                Throttle::Quota { .. } => "quota",
            },
        )
        .increment(1);
        let code = if matches!(throttle, Throttle::Quota { .. }) {
            "insufficient_quota"
        } else {
            "rate_limit_exceeded"
        };
        let mut error = ApiError::new(
            StatusCode::TOO_MANY_REQUESTS,
            ErrorKind::RateLimited,
            code,
            throttle.to_string(),
        );
        error.retry_after = throttle.retry_after();
        error
    })
}

/// A request that has been converted, rendered and tokenized.
pub struct Prepared {
    pub request_id: Uuid,
    pub response_id: String,
    pub protocol: &'static str,
    pub model: Arc<ServedModel>,
    pub tenant: Tenant,
    pub session_key: Option<String>,
    pub request: ConversationRequest,
    pub generate: pb::GenerateRequest,
    pub received_at: Instant,
    pub received_at_ms: u64,
    /// Gateway-side stages so far (authentication, parsing, conversion,
    /// rendering, tokenization).
    pub timeline: Vec<Stage>,
}

/// Parse, convert, render and tokenize a protocol request.
pub fn prepare<T>(
    state: &AppState,
    headers: &HeaderMap,
    body: &[u8],
    tenant: Tenant,
    received_at: Instant,
) -> Result<Prepared, ApiError>
where
    T: ProtocolRequest + ProtocolInfo + DeserializeOwned,
{
    let mut timeline = Vec::with_capacity(8);
    let start = Instant::now();
    timeline.push(stage("auth", received_at, received_at, start));
    let envelope: Envelope = serde_json::from_slice(body)
        .map_err(|error| ApiError::bad_request("invalid_json", format!("invalid JSON: {error}")))?;
    let model_name = envelope
        .model
        .as_deref()
        .ok_or_else(|| ApiError::bad_request("missing_model", "`model` is required"))?;
    let model = state.models.resolve(model_name).ok_or_else(|| {
        ApiError::new(
            axum::http::StatusCode::NOT_FOUND,
            ErrorKind::NotFound,
            "model_not_found",
            format!("model `{model_name}` is not served by this gateway"),
        )
    })?;
    let parsed: T = serde_json::from_slice(body).map_err(|error| {
        ApiError::bad_request("invalid_request_error", format!("invalid request: {error}"))
    })?;
    let parsed_at = Instant::now();
    timeline.push(stage("parse", received_at, start, parsed_at));
    let request = parsed.convert(model.options.clone())?;
    let converted_at = Instant::now();
    timeline.push(stage("convert", received_at, parsed_at, converted_at));
    let rendered = model.render(&request.conversation);
    let rendered_at = Instant::now();
    timeline.push(stage("render", received_at, converted_at, rendered_at));
    if !rendered.image_sources.is_empty() {
        return Err(ApiError::bad_request(
            "unsupported_feature",
            "image input is not enabled on this deployment",
        ));
    }
    let prompt_token_ids = model
        .encode(&rendered)
        .map_err(|error| ApiError::internal(format!("tokenization failed: {error}")))?;
    let max_tokens =
        request.resolve_max_tokens(prompt_token_ids.len() as u32, model.context_limit)?;
    timeline.push(stage("tokenize", received_at, rendered_at, Instant::now()));

    let request_id = Uuid::now_v7();
    let generate = generate_request(
        &request,
        &model,
        &request_id,
        prompt_token_ids,
        max_tokens,
        &tenant.id,
    );
    Ok(Prepared {
        request_id,
        response_id: T::response_id(&request_id),
        protocol: T::NAME,
        session_key: envelope.session_key(headers),
        model,
        tenant,
        request,
        generate,
        received_at,
        received_at_ms: now_ms(),
        timeline,
    })
}

fn generate_request(
    request: &ConversationRequest,
    model: &ServedModel,
    request_id: &Uuid,
    prompt_token_ids: Vec<u32>,
    max_tokens: u32,
    tenant: &str,
) -> pb::GenerateRequest {
    let options = &request.inference_options;
    // Every request gets a seed so that it can be replayed.
    let seed = options.seed.unwrap_or_else(|| rand::random::<u64>() >> 1);
    let conversation = &request.conversation;
    let mut hints = std::collections::HashMap::new();
    hints.insert(
        "thinking".to_string(),
        conversation.thinking_mode.to_string(),
    );
    hints.insert(
        "last_role".to_string(),
        last_role(conversation.messages.last()).to_string(),
    );
    if let Some(effort) = conversation.reasoning_effort {
        hints.insert(
            "effort".to_string(),
            format!("{effort:?}").to_ascii_lowercase(),
        );
    }
    pb::GenerateRequest {
        request_id: request_id.to_string(),
        prompt_token_ids,
        sampling: Some(pb::SamplingParams {
            temperature: options.temperature.unwrap_or(1.0),
            top_p: options.top_p.unwrap_or(1.0),
            max_tokens,
            seed,
            logprobs: options.logprobs,
            top_logprobs: options.top_logprobs.map(u32::from).unwrap_or(0),
            stop_token_ids: vec![model.eos_token_id],
            ignore_eos: false,
        }),
        constraint: options.output_constraint.as_ref().map(constraint),
        stage: pb::Role::Unified as i32,
        kv_transfer: None,
        tenant_id: tenant.to_string(),
        priority: 0,
        hints,
    }
}

fn last_role(message: Option<&InputMessage>) -> &'static str {
    match message {
        Some(InputMessage::System { .. }) => "system",
        Some(InputMessage::User { .. }) => "user",
        Some(InputMessage::Assistant { .. }) => "assistant",
        Some(InputMessage::Tool { .. }) => "tool",
        Some(_) => "other",
        None => "none",
    }
}

fn constraint(constraint: &OutputConstraint) -> pb::OutputConstraint {
    use pb::output_constraint::Kind;
    let kind = match constraint.kind() {
        "json" => Kind::Json,
        "regex" => Kind::Regex,
        "tool_call" => Kind::ToolCall,
        "tool_call_required" => Kind::ToolCallRequired,
        _ => Kind::Unspecified,
    };
    pb::OutputConstraint {
        kind: kind as i32,
        schema: constraint.schema(),
    }
}

/// What the dispatcher needs to run a request.
#[derive(Debug, Clone, Copy)]
pub struct DispatchInput<'a> {
    pub request_id: &'a str,
    pub protocol: &'static str,
    pub model: &'a str,
    pub generate: &'a pb::GenerateRequest,
    pub session_key: Option<&'a str>,
}

/// A request whose prefill is done and whose output stream is open.
pub struct Dispatched {
    pub ready: pb::Ready,
    pub stream: Streaming<pb::GenerateResponse>,
    pub ticket: Ticket,
    pub worker_id: String,
    pub prefill_worker_id: Option<String>,
    pub plan: Plan,
    pub policy: Policy,
    pub attempts: u32,
}

enum AttemptError {
    /// Try another worker.
    Retry {
        worker_id: String,
        detail: String,
        unhealthy: bool,
    },
    Fatal(ApiError),
}

fn classify(worker_id: &str, status: Status) -> AttemptError {
    let detail = format!("worker {worker_id}: {}", status.message());
    match status.code() {
        Code::InvalidArgument | Code::OutOfRange | Code::FailedPrecondition => {
            AttemptError::Fatal(ApiError::bad_request("worker_rejected", detail))
        }
        Code::Unavailable | Code::Unknown | Code::Cancelled | Code::DeadlineExceeded => {
            AttemptError::Retry {
                worker_id: worker_id.to_string(),
                detail,
                unhealthy: matches!(status.code(), Code::Unavailable | Code::Unknown),
            }
        }
        _ => AttemptError::Retry {
            worker_id: worker_id.to_string(),
            detail,
            unhealthy: false,
        },
    }
}

/// Route a request and open its output stream, retrying on other workers
/// while no output has been produced.
pub async fn dispatch(state: &AppState, input: DispatchInput<'_>) -> Result<Dispatched, ApiError> {
    let max_attempts = state.config.dispatch.max_attempts.max(1);
    let timeout = Duration::from_millis(state.config.dispatch.ready_timeout_ms);
    let mut exclude: Vec<String> = Vec::new();
    let mut last_error: Option<ApiError> = None;
    for attempt in 1..=max_attempts {
        let decision = match state.router.route(RouteRequest {
            model: input.model,
            prompt: &input.generate.prompt_token_ids,
            session_key: input.session_key,
            exclude: &exclude,
        }) {
            Ok(decision) => decision,
            Err(error) => {
                return Err(last_error.unwrap_or_else(|| ApiError::overloaded(error.to_string())));
            }
        };
        let RouteDecision {
            plan,
            policy,
            prompt_tokens,
            candidates,
            affinity,
            mut ticket,
        } = decision;
        metrics::counter!(
            "lev_route_decisions_total",
            "policy" => policy.as_str(),
            "plan" => plan_kind(&plan),
        )
        .increment(1);
        state.events.publish(LiveEvent::Route {
            request_id: input.request_id.to_string(),
            protocol: input.protocol,
            model: input.model.to_string(),
            policy,
            plan: plan.clone(),
            prompt_tokens,
            candidates,
            affinity,
            attempt,
            at_ms: now_ms(),
        });

        let result = match &plan {
            Plan::Unified { worker } => {
                run_stage(state, &worker.worker_id, input.generate.clone(), timeout)
                    .await
                    .map(|(ready, stream)| (ready, stream, worker.worker_id.clone(), None))
            }
            Plan::Disaggregated { prefill, decode } => {
                run_disaggregated(
                    state,
                    input.generate,
                    &prefill.worker_id,
                    &decode.worker_id,
                    timeout,
                )
                .await
            }
        };
        match result {
            Ok((ready, stream, worker_id, prefill_worker_id)) => {
                ticket.prefill_done();
                return Ok(Dispatched {
                    ready,
                    stream,
                    ticket,
                    worker_id,
                    prefill_worker_id,
                    plan,
                    policy,
                    attempts: attempt,
                });
            }
            Err(AttemptError::Retry {
                worker_id,
                detail,
                unhealthy,
            }) => {
                metrics::counter!("lev_dispatch_retries_total").increment(1);
                tracing::warn!(request_id = input.request_id, worker = %worker_id, %detail, attempt, "dispatch attempt failed");
                if unhealthy {
                    state.pool.report_failure(&worker_id, &detail);
                }
                exclude.push(worker_id);
                last_error = Some(ApiError::upstream(detail));
            }
            Err(AttemptError::Fatal(error)) => return Err(error),
        }
    }
    Err(last_error.unwrap_or_else(|| ApiError::overloaded("no worker accepted the request")))
}

fn plan_kind(plan: &Plan) -> &'static str {
    match plan {
        Plan::Unified { .. } => "unified",
        Plan::Disaggregated { .. } => "disaggregated",
    }
}

/// Send one stage to a worker and wait for its `Ready`.
async fn run_stage(
    state: &AppState,
    worker_id: &str,
    request: pb::GenerateRequest,
    timeout: Duration,
) -> Result<(pb::Ready, Streaming<pb::GenerateResponse>), AttemptError> {
    let Some(entry) = state.pool.get(worker_id) else {
        return Err(AttemptError::Retry {
            worker_id: worker_id.to_string(),
            detail: format!("worker {worker_id} is not connected"),
            unhealthy: true,
        });
    };
    let mut client = entry.client;
    let attempt = async {
        let mut stream = client
            .generate(request)
            .await
            .map_err(|status| classify(worker_id, status))?
            .into_inner();
        loop {
            match stream.message().await {
                Ok(Some(message)) => match message.event {
                    Some(pb::generate_response::Event::Ready(ready)) => return Ok((ready, stream)),
                    Some(pb::generate_response::Event::Finished(finished)) => {
                        return Err(AttemptError::Retry {
                            worker_id: worker_id.to_string(),
                            detail: format!(
                                "worker {worker_id} finished before prefill completed: {}",
                                finished.error
                            ),
                            unhealthy: false,
                        });
                    }
                    Some(pb::generate_response::Event::Tokens(_)) => {
                        return Err(AttemptError::Fatal(ApiError::upstream(format!(
                            "worker {worker_id} sent tokens before Ready"
                        ))));
                    }
                    None => {}
                },
                Ok(None) => {
                    return Err(AttemptError::Retry {
                        worker_id: worker_id.to_string(),
                        detail: format!("worker {worker_id} closed the stream before Ready"),
                        unhealthy: false,
                    });
                }
                Err(status) => return Err(classify(worker_id, status)),
            }
        }
    };
    match tokio::time::timeout(timeout, attempt).await {
        Ok(result) => result,
        Err(_) => Err(AttemptError::Retry {
            worker_id: worker_id.to_string(),
            detail: format!("worker {worker_id} did not finish prefill within {timeout:?}"),
            unhealthy: false,
        }),
    }
}

/// Prefill on one worker, hand its KV to another, decode there.
async fn run_disaggregated(
    state: &AppState,
    generate: &pb::GenerateRequest,
    prefill_worker: &str,
    decode_worker: &str,
    timeout: Duration,
) -> Result<
    (
        pb::Ready,
        Streaming<pb::GenerateResponse>,
        String,
        Option<String>,
    ),
    AttemptError,
> {
    let mut prefill_request = generate.clone();
    prefill_request.stage = pb::Role::Prefill as i32;
    let (prefill_ready, mut prefill_stream) =
        run_stage(state, prefill_worker, prefill_request, timeout).await?;
    let Some(kv_transfer) = prefill_ready.kv_transfer.clone() else {
        return Err(AttemptError::Fatal(ApiError::upstream(format!(
            "prefill worker {prefill_worker} returned no KV hand-off"
        ))));
    };
    // The prefill stage ends right after Ready; read its Finished so that the
    // worker does not see the hand-off as a cancellation.
    let _ = tokio::time::timeout(Duration::from_secs(5), async {
        while let Ok(Some(message)) = prefill_stream.message().await {
            if matches!(
                message.event,
                Some(pb::generate_response::Event::Finished(_))
            ) {
                break;
            }
        }
    })
    .await;

    let mut decode_request = generate.clone();
    decode_request.stage = pb::Role::Decode as i32;
    decode_request.kv_transfer = Some(kv_transfer);
    let (decode_ready, stream) = run_stage(state, decode_worker, decode_request, timeout).await?;
    let ready = pb::Ready {
        prompt_tokens: prefill_ready.prompt_tokens,
        // A cache hit is what the prefill worker did not recompute; tokens the
        // decode worker received by transfer are not hits.
        cached_tokens: prefill_ready.cached_tokens,
        queue_time_us: prefill_ready.queue_time_us + decode_ready.queue_time_us,
        prefill_time_us: prefill_ready.prefill_time_us + decode_ready.prefill_time_us,
        kv_transfer: None,
    };
    Ok((
        ready,
        stream,
        decode_worker.to_string(),
        Some(prefill_worker.to_string()),
    ))
}

#[derive(Debug, Default)]
struct RecordState {
    worker_id: Option<String>,
    prefill_worker_id: Option<String>,
    plan: Option<serde_json::Value>,
    policy: Option<&'static str>,
    attempts: u32,
    prompt_tokens: u32,
    cached_tokens: u32,
    first_token_at: Option<Instant>,
    last_token_at: Option<Instant>,
    output_token_ids: Vec<u32>,
    completion_tokens: u32,
    draft_tokens: u64,
    accepted_tokens: u64,
    finish_reason: Option<String>,
    error: Option<ApiError>,
    completed: bool,
    timeline: Vec<Stage>,
}

/// Everything observed about one request. Finalized (metrics, audit log,
/// live event) when the last reference drops, which also covers clients that
/// disconnect mid-stream.
pub struct RequestRecord {
    state: AppState,
    request_id: String,
    response_id: String,
    protocol: &'static str,
    model: String,
    tenant: Tenant,
    received_at: Instant,
    received_at_ms: u64,
    replay: ReplayInputs,
    inner: Mutex<RecordState>,
}

impl RequestRecord {
    pub fn new(state: &AppState, prepared: &Prepared) -> Arc<Self> {
        Arc::new(Self {
            state: state.clone(),
            request_id: prepared.request_id.to_string(),
            response_id: prepared.response_id.clone(),
            protocol: prepared.protocol,
            model: prepared.model.name.clone(),
            tenant: prepared.tenant.clone(),
            received_at: prepared.received_at,
            received_at_ms: prepared.received_at_ms,
            replay: ReplayInputs {
                model: prepared.model.name.clone(),
                generate: prepared.generate.clone(),
            },
            inner: Mutex::new(RecordState {
                prompt_tokens: prepared.generate.prompt_token_ids.len() as u32,
                timeline: prepared.timeline.clone(),
                ..RecordState::default()
            }),
        })
    }

    fn on_dispatched(&self, dispatched: &Dispatched, dispatch_started: Instant) {
        let now = Instant::now();
        let mut inner = self.inner.lock();
        let origin = self.received_at;
        // The worker's own account of the request: queueing, then prefill.
        let prefill = Duration::from_micros(dispatched.ready.prefill_time_us);
        let queue = Duration::from_micros(dispatched.ready.queue_time_us);
        let prefill_start = now
            .checked_sub(prefill)
            .unwrap_or(now)
            .max(dispatch_started);
        let queue_start = prefill_start
            .checked_sub(queue)
            .unwrap_or(prefill_start)
            .max(dispatch_started);
        inner
            .timeline
            .push(stage("dispatch", origin, dispatch_started, now));
        if !queue.is_zero() {
            inner
                .timeline
                .push(stage("worker.queue", origin, queue_start, prefill_start));
        }
        inner
            .timeline
            .push(stage("worker.prefill", origin, prefill_start, now));
        inner.worker_id = Some(dispatched.worker_id.clone());
        inner.prefill_worker_id = dispatched.prefill_worker_id.clone();
        inner.plan = serde_json::to_value(&dispatched.plan).ok();
        inner.policy = Some(dispatched.policy.as_str());
        inner.attempts = dispatched.attempts;
        inner.prompt_tokens = dispatched.ready.prompt_tokens;
        inner.cached_tokens = dispatched.ready.cached_tokens;
    }

    fn on_tokens(&self, batch: &pb::TokenBatch) {
        let now = Instant::now();
        let mut inner = self.inner.lock();
        if !batch.token_ids.is_empty() {
            inner.first_token_at.get_or_insert(now);
            inner.last_token_at = Some(now);
        }
        inner.output_token_ids.extend_from_slice(&batch.token_ids);
        inner.completion_tokens += batch.token_ids.len() as u32;
        inner.draft_tokens += u64::from(batch.draft_tokens);
        inner.accepted_tokens += u64::from(batch.accepted_tokens);
    }

    fn on_finish(&self, finished: &pb::Finished) {
        let reason = pb::FinishReason::try_from(finished.reason)
            .map(|reason| {
                reason
                    .as_str_name()
                    .trim_start_matches("FINISH_REASON_")
                    .to_ascii_lowercase()
            })
            .unwrap_or_else(|_| "unknown".to_string());
        self.inner.lock().finish_reason = Some(reason);
    }

    pub fn fail(&self, error: ApiError) {
        let mut inner = self.inner.lock();
        if inner.error.is_none() {
            inner.error = Some(error);
        }
    }

    pub fn error(&self) -> Option<ApiError> {
        self.inner.lock().error.clone()
    }

    pub fn complete(&self) {
        self.inner.lock().completed = true;
    }
}

impl Drop for RequestRecord {
    fn drop(&mut self) {
        let mut inner = std::mem::take(&mut *self.inner.lock());
        let now = Instant::now();
        if let (Some(first), Some(last)) = (inner.first_token_at, inner.last_token_at) {
            inner
                .timeline
                .push(stage("decode", self.received_at, first, last));
            inner
                .timeline
                .push(stage("respond", self.received_at, last, now));
        }
        let outcome: &'static str = if inner.error.is_some() {
            if inner.worker_id.is_none() {
                "rejected"
            } else {
                "error"
            }
        } else if inner.completed {
            "ok"
        } else {
            "cancelled"
        };
        let ttft = inner
            .first_token_at
            .map(|at| at.duration_since(self.received_at));
        let e2e = now.duration_since(self.received_at);
        let tpot = match (inner.first_token_at, inner.last_token_at) {
            (Some(first), Some(last)) if inner.completion_tokens > 1 => {
                Some(last.duration_since(first) / (inner.completion_tokens - 1))
            }
            _ => None,
        };

        let model = self.model.clone();
        metrics::counter!(
            "lev_requests_total",
            "protocol" => self.protocol,
            "model" => model.clone(),
            "outcome" => outcome,
        )
        .increment(1);
        metrics::histogram!("lev_e2e_seconds", "model" => model.clone()).record(e2e.as_secs_f64());
        if let Some(ttft) = ttft {
            metrics::histogram!("lev_ttft_seconds", "model" => model.clone())
                .record(ttft.as_secs_f64());
        }
        if let Some(tpot) = tpot {
            metrics::histogram!("lev_tpot_seconds", "model" => model.clone())
                .record(tpot.as_secs_f64());
        }
        let tenant = self.tenant.id.clone();
        metrics::counter!("lev_prompt_tokens_total", "model" => model.clone(), "tenant" => tenant.clone())
            .increment(u64::from(inner.prompt_tokens));
        metrics::counter!("lev_cached_tokens_total", "model" => model.clone(), "tenant" => tenant.clone())
            .increment(u64::from(inner.cached_tokens));
        metrics::counter!("lev_completion_tokens_total", "model" => model.clone(), "tenant" => tenant)
            .increment(u64::from(inner.completion_tokens));
        metrics::counter!("lev_spec_draft_tokens_total", "model" => model.clone())
            .increment(inner.draft_tokens);
        metrics::counter!("lev_spec_accepted_tokens_total", "model" => model)
            .increment(inner.accepted_tokens);

        let ms = |duration: Duration| duration.as_secs_f64() * 1e3;
        self.state.events.publish(LiveEvent::RequestDone {
            request_id: self.request_id.clone(),
            model: self.model.clone(),
            tenant: self.tenant.id.clone(),
            worker_id: inner.worker_id.clone(),
            outcome,
            prompt_tokens: inner.prompt_tokens,
            cached_tokens: inner.cached_tokens,
            completion_tokens: inner.completion_tokens,
            ttft_ms: ttft.map(ms),
            e2e_ms: ms(e2e),
            at_ms: now_ms(),
        });
        if inner.worker_id.is_some()
            && let Ok(runtime) = tokio::runtime::Handle::try_current()
        {
            let control = Arc::clone(&self.state.control);
            let tenant = self.tenant.clone();
            let record = UsageRecord {
                tenant_id: tenant.id.clone(),
                request_id: self.request_id.clone(),
                model: self.model.clone(),
                prompt_tokens: u64::from(inner.prompt_tokens),
                cached_tokens: u64::from(inner.cached_tokens),
                completion_tokens: u64::from(inner.completion_tokens),
                at: Utc::now(),
            };
            runtime.spawn(async move { control.settle(&tenant, record).await });
        }
        let seed = self
            .replay
            .generate
            .sampling
            .as_ref()
            .map_or(0, |sampling| sampling.seed);
        self.state.audit.push(AuditEntry {
            request_id: self.request_id.clone(),
            response_id: self.response_id.clone(),
            protocol: self.protocol,
            model: self.model.clone(),
            tenant: self.tenant.id.clone(),
            received_at_ms: self.received_at_ms,
            policy: inner.policy.map(str::to_string),
            plan: inner.plan,
            worker_id: inner.worker_id,
            prefill_worker_id: inner.prefill_worker_id,
            attempts: inner.attempts,
            prompt_tokens: inner.prompt_tokens,
            cached_tokens: inner.cached_tokens,
            completion_tokens: inner.completion_tokens,
            seed,
            ttft_ms: ttft.map(ms),
            tpot_ms: tpot.map(ms),
            e2e_ms: ms(e2e),
            outcome,
            finish_reason: inner.finish_reason,
            error: inner.error.map(|error| error.message),
            spec_draft_tokens: inner.draft_tokens,
            spec_accepted_tokens: inner.accepted_tokens,
            replay: Some(self.replay.clone()),
            output_token_ids: inner.output_token_ids,
            timeline: inner.timeline,
        });
    }
}

fn recipe_logprob(logprob: &pb::TokenLogprob) -> InferenceTokenLogprob {
    InferenceTokenLogprob {
        logprob: logprob.logprob,
        top_logprobs: logprob
            .top_token_ids
            .iter()
            .copied()
            .zip(logprob.top_logprobs.iter().copied())
            .collect(),
    }
}

/// The worker's output as recipe inference chunks.
fn inference_stream(
    dispatched: Dispatched,
    record: Arc<RequestRecord>,
    fingerprint: String,
) -> impl Stream<Item = InferenceChunk> + Send + 'static {
    async_stream::stream! {
        let Dispatched { ready, mut stream, ticket, .. } = dispatched;
        // Holds the request's load on its workers until the stream ends.
        let _ticket = ticket;
        yield InferenceChunk::Ready {
            system_fingerprint: Some(fingerprint),
            prompt_usage: PromptUsage {
                prompt_tokens: ready.prompt_tokens as usize,
                prompt_cache_hit_tokens: ready.cached_tokens as usize,
            },
        };
        loop {
            match stream.message().await {
                Ok(Some(message)) => match message.event {
                    Some(pb::generate_response::Event::Tokens(batch)) => {
                        record.on_tokens(&batch);
                        if batch.token_ids.is_empty() {
                            continue;
                        }
                        let logprobs = (!batch.logprobs.is_empty())
                            .then(|| batch.logprobs.iter().map(recipe_logprob).collect());
                        yield InferenceChunk::Tokens { token_ids: batch.token_ids, logprobs };
                    }
                    Some(pb::generate_response::Event::Finished(finished)) => {
                        record.on_finish(&finished);
                        let reason = match pb::FinishReason::try_from(finished.reason) {
                            Ok(pb::FinishReason::Stop) => Some(InferenceFinishReason::Stop),
                            Ok(pb::FinishReason::Length) => Some(InferenceFinishReason::Length),
                            Ok(pb::FinishReason::ContentFilter) => Some(InferenceFinishReason::ContentFilter),
                            _ => None,
                        };
                        match reason {
                            Some(finish_reason) => yield InferenceChunk::Finish { finish_reason },
                            None => record.fail(ApiError::upstream(if finished.error.is_empty() {
                                "the worker aborted the request".to_string()
                            } else {
                                finished.error
                            })),
                        }
                        break;
                    }
                    _ => {}
                },
                Ok(None) => {
                    record.fail(ApiError::upstream("the worker closed the stream before finishing"));
                    break;
                }
                Err(status) => {
                    record.fail(ApiError::upstream(format!("worker stream failed: {}", status.message())));
                    break;
                }
            }
        }
    }
}

/// Dispatch a prepared request and answer in its protocol.
pub async fn respond<T>(state: AppState, prepared: Prepared) -> Response
where
    T: ProtocolRequest + ProtocolInfo,
    ProtocolChunk<T>: Serialize + Send,
{
    let record = RequestRecord::new(&state, &prepared);
    let request_id = prepared.request_id.to_string();
    let dispatch_started = Instant::now();
    let dispatched = match dispatch(
        &state,
        DispatchInput {
            request_id: &request_id,
            protocol: prepared.protocol,
            model: &prepared.model.name,
            generate: &prepared.generate,
            session_key: prepared.session_key.as_deref(),
        },
    )
    .await
    {
        Ok(dispatched) => dispatched,
        Err(error) => {
            record.fail(error.clone());
            return with_request_id(T::error_response(&error), &request_id);
        }
    };
    record.on_dispatched(&dispatched, dispatch_started);

    let Prepared {
        request,
        model,
        response_id,
        ..
    } = prepared;
    let streaming = request.stream;
    let generator = T::chunk_generator(&request, response_id.clone(), model.name.clone());
    let processor = StreamProcessor::new(generator, request.parsing_options)
        .with_tokenizer(Arc::clone(&model.tokenizer));
    let inference = inference_stream(dispatched, Arc::clone(&record), state.fingerprint.clone());
    let protocol_stream = processor.into_protocol_stream(inference);
    let response = if streaming {
        sse_response::<T>(protocol_stream, record)
    } else {
        json_response::<T>(protocol_stream, record, response_id, model.name.clone()).await
    };
    with_request_id(response, &request_id)
}

fn with_request_id(mut response: Response, request_id: &str) -> Response {
    if let Ok(value) = HeaderValue::from_str(request_id) {
        response.headers_mut().insert("x-request-id", value);
    }
    response
}

fn sse_response<T>(
    stream: impl Stream<Item = Result<ProtocolChunk<T>, StreamError>> + Send + 'static,
    record: Arc<RequestRecord>,
) -> Response
where
    T: ProtocolRequest + ProtocolInfo,
    ProtocolChunk<T>: Serialize + Send,
{
    let body = async_stream::stream! {
        let mut stream = Box::pin(stream);
        while let Some(item) = stream.next().await {
            match item {
                Ok(chunk) => match serde_json::to_string(&chunk) {
                    Ok(data) => {
                        let frame = match <T::Response as ProtocolResponse>::sse_event_name(&chunk) {
                            Some(name) => format!("event: {name}\ndata: {data}\n\n"),
                            None => format!("data: {data}\n\n"),
                        };
                        yield Ok::<String, std::convert::Infallible>(frame);
                    }
                    Err(error) => {
                        record.fail(ApiError::internal(format!("serializing a chunk failed: {error}")));
                        break;
                    }
                },
                Err(error) => {
                    record.fail(ApiError::internal(format!("output processing failed: {error}")));
                    break;
                }
            }
        }
        match record.error() {
            Some(error) => yield Ok(T::stream_error_frame(&error)),
            None => {
                if let Some(done) = <T::Response as ProtocolResponse>::sse_done_data() {
                    yield Ok(format!("data: {done}\n\n"));
                }
                record.complete();
            }
        }
    };
    let mut response = Response::new(Body::from_stream(body));
    let headers = response.headers_mut();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/event-stream; charset=utf-8"),
    );
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    headers.insert("x-accel-buffering", HeaderValue::from_static("no"));
    response
}

async fn json_response<T>(
    stream: impl Stream<Item = Result<ProtocolChunk<T>, StreamError>> + Send + 'static,
    record: Arc<RequestRecord>,
    response_id: String,
    model: String,
) -> Response
where
    T: ProtocolRequest + ProtocolInfo,
    ProtocolChunk<T>: Serialize + Send,
{
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0);
    let mut response = T::Response::new(response_id, model, created, 0, 0);
    let mut stream = Box::pin(stream);
    while let Some(item) = stream.next().await {
        match item {
            Ok(chunk) => response.accumulate(chunk),
            Err(error) => {
                record.fail(ApiError::internal(format!(
                    "output processing failed: {error}"
                )));
                break;
            }
        }
    }
    if let Some(error) = record.error() {
        return T::error_response(&error);
    }
    record.complete();
    Json(response).into_response()
}
