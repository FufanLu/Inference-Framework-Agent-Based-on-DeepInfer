//! HTTP routes.

use std::convert::Infallible;
use std::time::Duration;

use axum::Json;
use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use deepseek_recipe::anthropic::MessagesRequest;
use deepseek_recipe::openai::{ChatCompletionRequest, ResponsesRequest};
use deepseek_recipe::request::ProtocolRequest;
use lev_proto::pb;
use lev_router::{IndexMode, Policy};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::json;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::BroadcastStream;
use tower_http::cors::CorsLayer;

use crate::error::{ApiError, ErrorKind};
use crate::pipeline::{self, DispatchInput, ProtocolChunk};
use crate::protocol::ProtocolInfo;
use crate::state::AppState;

pub fn router(state: AppState) -> Router {
    Router::new()
        .route(
            "/v1/chat/completions",
            post(protocol_handler::<ChatCompletionRequest>),
        )
        .route("/v1/responses", post(protocol_handler::<ResponsesRequest>))
        .route("/v1/messages", post(protocol_handler::<MessagesRequest>))
        .route("/v1/models", get(list_models))
        .route("/v1/leviathan/render", post(render))
        .route("/healthz", get(|| async { "ok" }))
        .route("/readyz", get(ready))
        .route("/metrics", get(metrics))
        .route("/admin/overview", get(overview))
        .route("/admin/router", get(router_snapshot).put(update_router))
        .route("/admin/workers", get(list_workers).post(add_worker))
        .route("/admin/requests", get(list_requests))
        .route("/admin/requests/{id}", get(get_request))
        .route("/admin/requests/{id}/replay", post(replay_request))
        .route("/admin/live", get(live))
        .route("/admin/tenants", get(list_tenants).post(upsert_tenant))
        .route("/admin/tenants/{id}/keys", post(create_key))
        .route("/admin/keys", get(list_keys))
        .route("/admin/keys/{id}", axum::routing::delete(revoke_key))
        .route("/admin/usage", get(usage))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

async fn protocol_handler<T>(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response
where
    T: ProtocolRequest + ProtocolInfo + DeserializeOwned,
    ProtocolChunk<T>: serde::Serialize + Send,
{
    let received_at = std::time::Instant::now();
    let tenant = match pipeline::authenticate(&state, &headers).await {
        Ok(tenant) => tenant,
        Err(error) => return T::error_response(&error),
    };
    if let Err(error) = pipeline::admit(&state, &tenant).await {
        return T::error_response(&error);
    }
    match pipeline::prepare::<T>(&state, &headers, &body, tenant, received_at) {
        Ok(prepared) => pipeline::respond::<T>(state, prepared).await,
        Err(error) => {
            metrics::counter!(
                "lev_requests_total",
                "protocol" => T::NAME,
                "model" => "unknown",
                "outcome" => "invalid",
            )
            .increment(1);
            T::error_response(&error)
        }
    }
}

async fn list_models(State(state): State<AppState>) -> Json<serde_json::Value> {
    let snapshot = state.router.snapshot();
    let data: Vec<_> = state
        .models
        .all()
        .iter()
        .map(|model| {
            let workers = snapshot
                .workers
                .iter()
                .filter(|worker| worker.model == model.name && worker.healthy)
                .count();
            json!({
                "id": model.name,
                "object": "model",
                "owned_by": "deepseek",
                "aliases": model.aliases,
                "context_length": model.context_limit,
                "healthy_workers": workers,
            })
        })
        .collect();
    Json(json!({ "object": "list", "data": data }))
}

async fn ready(State(state): State<AppState>) -> Response {
    let healthy = state
        .router
        .snapshot()
        .workers
        .iter()
        .filter(|worker| worker.healthy)
        .count();
    if healthy > 0 {
        (StatusCode::OK, format!("ready: {healthy} healthy workers")).into_response()
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "no healthy workers").into_response()
    }
}

async fn metrics(State(state): State<AppState>) -> Response {
    match &state.metrics {
        Some(handle) => (
            [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
            handle.render(),
        )
            .into_response(),
        None => (StatusCode::NOT_FOUND, "metrics are disabled").into_response(),
    }
}

// ---------------------------------------------------------------- inspector

#[derive(Debug, Deserialize)]
struct RenderQuery {
    #[serde(default)]
    protocol: Option<String>,
}

/// Prompt Inspector: show exactly what the model will see for a request.
async fn render(
    State(state): State<AppState>,
    Query(query): Query<RenderQuery>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let tenant = match pipeline::authenticate(&state, &headers).await {
        Ok(tenant) => tenant,
        Err(error) => return (error.status, Json(error.openai_body())).into_response(),
    };
    let result = match query.protocol.as_deref().unwrap_or("chat_completions") {
        "chat_completions" => inspect::<ChatCompletionRequest>(&state, &headers, &body, tenant),
        "responses" => inspect::<ResponsesRequest>(&state, &headers, &body, tenant),
        "messages" => inspect::<MessagesRequest>(&state, &headers, &body, tenant),
        other => Err(ApiError::bad_request(
            "unknown_protocol",
            format!("unknown protocol `{other}`"),
        )),
    };
    match result {
        Ok(value) => Json(value).into_response(),
        Err(error) => (error.status, Json(error.openai_body())).into_response(),
    }
}

fn inspect<T>(
    state: &AppState,
    headers: &HeaderMap,
    body: &[u8],
    tenant: lev_control::Tenant,
) -> Result<serde_json::Value, ApiError>
where
    T: ProtocolRequest + ProtocolInfo + DeserializeOwned,
{
    let prepared = pipeline::prepare::<T>(state, headers, body, tenant, std::time::Instant::now())?;
    let model = &prepared.model;
    let rendered = model.render(&prepared.request.conversation);
    let special: Vec<_> = rendered
        .special_ranges
        .iter()
        .map(|range| {
            json!({
                "start": range.start,
                "end": range.end,
                "text": &rendered.prompt[range.clone()],
            })
        })
        .collect();
    let tokenizer = &model.tokenizer;
    let added = tokenizer.get_added_vocabulary().get_added_tokens_decoder();
    let ids = &prepared.generate.prompt_token_ids;
    let tokens: Vec<_> = ids
        .iter()
        .map(|&id| {
            json!({
                "id": id,
                "text": tokenizer.decode(&[id], false).unwrap_or_default(),
                "special": added.contains_key(&id),
            })
        })
        .collect();
    let sampling = prepared.generate.sampling.clone().unwrap_or_default();
    let constraint = prepared.generate.constraint.as_ref().map(|constraint| {
        json!({
            "kind": pb::output_constraint::Kind::try_from(constraint.kind)
                .map(|kind| kind.as_str_name())
                .unwrap_or("KIND_UNSPECIFIED"),
            "schema": constraint.schema,
        })
    });
    let snapshot = state.router.snapshot();
    let block_size = snapshot
        .workers
        .iter()
        .find(|worker| worker.model == model.name)
        .map_or(64, |worker| worker.block_size);
    let block_hashes: Vec<String> = lev_router::hashing::block_hashes(ids, block_size)
        .iter()
        .map(|hash| format!("{hash:016x}"))
        .collect();
    Ok(json!({
        "protocol": T::NAME,
        "model": model.name,
        "thinking_mode": prepared.request.conversation.thinking_mode,
        "stream": prepared.request.stream,
        "prompt": rendered.prompt,
        "special_ranges": special,
        "prompt_tokens": ids.len(),
        "context_limit": model.context_limit,
        "max_tokens": sampling.max_tokens,
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
        "seed": sampling.seed,
        "constraint": constraint,
        "hints": prepared.generate.hints,
        "block_size": block_size,
        "block_hashes": block_hashes,
        "tokens": tokens,
        "timeline": prepared.timeline,
    }))
}

// ------------------------------------------------------------------- admin

fn authorize_admin(state: &AppState, headers: &HeaderMap) -> Result<(), ApiError> {
    let Some(token) = &state.config.auth.admin_token else {
        return Ok(());
    };
    let presented = headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "));
    if presented == Some(token.as_str()) {
        Ok(())
    } else {
        Err(ApiError::new(
            StatusCode::UNAUTHORIZED,
            ErrorKind::Authentication,
            "invalid_admin_token",
            "admin endpoints require the admin token",
        ))
    }
}

fn admin_error(error: ApiError) -> Response {
    (error.status, Json(error.openai_body())).into_response()
}

async fn overview(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(error) = authorize_admin(&state, &headers) {
        return admin_error(error);
    }
    let snapshot = state.router.snapshot();
    let healthy = snapshot
        .workers
        .iter()
        .filter(|worker| worker.healthy)
        .count();
    Json(json!({
        "version": env!("CARGO_PKG_VERSION"),
        "uptime_s": state.started_at.elapsed().as_secs(),
        "fingerprint": state.fingerprint,
        "models": state.models.all().iter().map(|model| &model.name).collect::<Vec<_>>(),
        "workers": snapshot.workers.len(),
        "healthy_workers": healthy,
        "policy": snapshot.policy,
        "index_mode": snapshot.index_mode,
        "disaggregation": snapshot.disaggregation,
        "indexed_blocks": snapshot.indexed_blocks,
    }))
    .into_response()
}

async fn router_snapshot(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(error) = authorize_admin(&state, &headers) {
        return admin_error(error);
    }
    Json(state.router.snapshot()).into_response()
}

#[derive(Debug, Deserialize)]
struct RouterUpdate {
    #[serde(default)]
    policy: Option<Policy>,
    #[serde(default)]
    index_mode: Option<IndexMode>,
    #[serde(default)]
    disaggregation: Option<bool>,
}

/// Switch the routing policy, index mode or disaggregation at runtime.
async fn update_router(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(update): Json<RouterUpdate>,
) -> Response {
    if let Err(error) = authorize_admin(&state, &headers) {
        return admin_error(error);
    }
    if let Some(policy) = update.policy {
        state.router.set_policy(policy);
    }
    if let Some(mode) = update.index_mode {
        state.router.set_index_mode(mode);
    }
    if let Some(enabled) = update.disaggregation {
        state.router.set_disaggregation(enabled);
    }
    tracing::info!(?update, "router configuration changed");
    Json(state.router.snapshot()).into_response()
}

async fn list_workers(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(error) = authorize_admin(&state, &headers) {
        return admin_error(error);
    }
    let snapshot = state.router.snapshot();
    let workers: Vec<_> = state
        .pool
        .all()
        .into_iter()
        .map(|entry| {
            let view = snapshot.workers.iter().find(|view| view.id == entry.id);
            json!({
                "id": entry.id,
                "endpoint": entry.endpoint,
                "backend": entry.info.backend,
                "model": entry.info.model,
                "role": view.map(|view| view.role),
                "block_size": entry.info.block_size,
                "context_limit": entry.info.context_limit,
                "total_blocks": entry.info.total_blocks,
                "supports_kv_events": entry.info.supports_kv_events,
                "deterministic": entry.info.deterministic,
                "labels": entry.info.labels,
                "router": view,
            })
        })
        .collect();
    Json(json!({ "endpoints": state.pool.endpoints(), "workers": workers })).into_response()
}

#[derive(Debug, Deserialize)]
struct AddWorker {
    endpoint: String,
}

async fn add_worker(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<AddWorker>,
) -> Response {
    if let Err(error) = authorize_admin(&state, &headers) {
        return admin_error(error);
    }
    match state.pool.add_endpoint(&body.endpoint) {
        Ok(()) => (
            StatusCode::ACCEPTED,
            Json(json!({ "endpoint": body.endpoint })),
        )
            .into_response(),
        Err(error) => admin_error(ApiError::bad_request("invalid_endpoint", error.to_string())),
    }
}

#[derive(Debug, Deserialize)]
struct ListQuery {
    #[serde(default = "default_limit")]
    limit: usize,
}

fn default_limit() -> usize {
    50
}

async fn list_requests(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<ListQuery>,
) -> Response {
    if let Err(error) = authorize_admin(&state, &headers) {
        return admin_error(error);
    }
    let entries = state
        .audit
        .recent(query.limit.min(state.config.audit.capacity));
    let entries: Vec<&crate::audit::AuditEntry> = entries.iter().map(|entry| &**entry).collect();
    Json(json!({ "requests": entries })).into_response()
}

async fn get_request(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if let Err(error) = authorize_admin(&state, &headers) {
        return admin_error(error);
    }
    let Some(entry) = state.audit.get(&id) else {
        return admin_error(ApiError::new(
            StatusCode::NOT_FOUND,
            ErrorKind::NotFound,
            "request_not_found",
            format!("request `{id}` is not in the audit log"),
        ));
    };
    let output_text = state
        .models
        .resolve(&entry.model)
        .and_then(|model| model.tokenizer.decode(&entry.output_token_ids, false).ok());
    let prompt_token_ids = entry
        .replay
        .as_ref()
        .map(|replay| replay.generate.prompt_token_ids.clone())
        .unwrap_or_default();
    Json(json!({
        "request": &*entry,
        "prompt_token_ids": prompt_token_ids,
        "output_token_ids": entry.output_token_ids,
        "output_text": output_text,
    }))
    .into_response()
}

/// Re-run a request with its recorded inputs (token IDs, sampling, seed,
/// constraint) and compare the output token by token.
async fn replay_request(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if let Err(error) = authorize_admin(&state, &headers) {
        return admin_error(error);
    }
    let Some(entry) = state.audit.get(&id) else {
        return admin_error(ApiError::new(
            StatusCode::NOT_FOUND,
            ErrorKind::NotFound,
            "request_not_found",
            format!("request `{id}` is not in the audit log"),
        ));
    };
    let Some(replay) = entry.replay.clone() else {
        return admin_error(ApiError::bad_request(
            "not_replayable",
            "the request has no recorded inputs",
        ));
    };
    let mut generate = replay.generate;
    let replay_id = format!("replay-{}", uuid::Uuid::now_v7());
    generate.request_id = replay_id.clone();
    let dispatched = match pipeline::dispatch(
        &state,
        DispatchInput {
            request_id: &replay_id,
            protocol: entry.protocol,
            model: &replay.model,
            generate: &generate,
            session_key: None,
        },
    )
    .await
    {
        Ok(dispatched) => dispatched,
        Err(error) => return admin_error(error),
    };
    let worker_id = dispatched.worker_id.clone();
    let mut stream = dispatched.stream;
    let mut tokens = Vec::new();
    let mut finish = None;
    while let Ok(Some(message)) = stream.message().await {
        match message.event {
            Some(pb::generate_response::Event::Tokens(batch)) => tokens.extend(batch.token_ids),
            Some(pb::generate_response::Event::Finished(finished)) => {
                finish = Some(finished.reason);
                break;
            }
            _ => {}
        }
    }
    drop(dispatched.ticket);
    let original = &entry.output_token_ids;
    let first_divergence = original
        .iter()
        .zip(&tokens)
        .position(|(a, b)| a != b)
        .or_else(|| (original.len() != tokens.len()).then(|| original.len().min(tokens.len())));
    let decode = |ids: &[u32]| {
        state
            .models
            .resolve(&replay.model)
            .and_then(|model| model.tokenizer.decode(ids, false).ok())
    };
    Json(json!({
        "request_id": entry.request_id,
        "replay_id": replay_id,
        "original_worker": entry.worker_id,
        "replay_worker": worker_id,
        "seed": entry.seed,
        "identical": first_divergence.is_none(),
        "first_divergence": first_divergence,
        "original_tokens": original.len(),
        "replay_tokens": tokens.len(),
        "finish_reason": finish.and_then(|reason| pb::FinishReason::try_from(reason).ok()).map(|reason| reason.as_str_name()),
        "original_text": decode(original),
        "replay_text": decode(&tokens),
    }))
    .into_response()
}

/// Server-sent live events: routing decisions, completions, worker status.
async fn live(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(error) = authorize_admin(&state, &headers) {
        return admin_error(error);
    }
    let events = BroadcastStream::new(state.events.subscribe()).filter_map(|event| {
        let event = event.ok()?;
        let data = serde_json::to_string(&event).ok()?;
        Some(Ok::<_, Infallible>(format!("data: {data}\n\n")))
    });
    let heartbeat =
        tokio_stream::wrappers::IntervalStream::new(tokio::time::interval(Duration::from_secs(15)))
            .map(|_| Ok::<_, Infallible>(": heartbeat\n\n".to_string()));
    let body = Body::from_stream(events.merge(heartbeat));
    let mut response = Response::new(body);
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/event-stream; charset=utf-8"),
    );
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    response
}

// ------------------------------------------------------------- tenants

async fn list_tenants(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(error) = authorize_admin(&state, &headers) {
        return admin_error(error);
    }
    match state.control.tenants_with_usage().await {
        Ok(rows) => {
            let tenants: Vec<_> = rows
                .into_iter()
                .map(|(tenant, usage)| json!({ "tenant": tenant, "usage_this_month": usage }))
                .collect();
            Json(json!({ "tenants": tenants })).into_response()
        }
        Err(error) => admin_error(ApiError::internal(error.to_string())),
    }
}

async fn upsert_tenant(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(tenant): Json<lev_control::Tenant>,
) -> Response {
    if let Err(error) = authorize_admin(&state, &headers) {
        return admin_error(error);
    }
    if tenant.id.is_empty() || tenant.id == "anonymous" {
        return admin_error(ApiError::bad_request("invalid_tenant", "invalid tenant id"));
    }
    match state.control.upsert_tenant(&tenant).await {
        Ok(()) => Json(json!({ "tenant": tenant })).into_response(),
        Err(error) => admin_error(ApiError::internal(error.to_string())),
    }
}

#[derive(Debug, Deserialize)]
struct CreateKey {
    #[serde(default = "default_key_name")]
    name: String,
}

fn default_key_name() -> String {
    "default".to_string()
}

/// Create an API key; the secret appears only in this response.
async fn create_key(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    body: Option<Json<CreateKey>>,
) -> Response {
    if let Err(error) = authorize_admin(&state, &headers) {
        return admin_error(error);
    }
    let name = body.map_or_else(default_key_name, |Json(body)| body.name);
    match state.control.create_key(&id, &name).await {
        Ok(created) => (StatusCode::CREATED, Json(created)).into_response(),
        Err(error) => admin_error(ApiError::bad_request("invalid_tenant", error.to_string())),
    }
}

#[derive(Debug, Deserialize)]
struct KeysQuery {
    #[serde(default)]
    tenant: Option<String>,
}

async fn list_keys(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<KeysQuery>,
) -> Response {
    if let Err(error) = authorize_admin(&state, &headers) {
        return admin_error(error);
    }
    match state
        .control
        .store()
        .list_keys(query.tenant.as_deref())
        .await
    {
        Ok(keys) => Json(json!({ "keys": keys })).into_response(),
        Err(error) => admin_error(ApiError::internal(error.to_string())),
    }
}

async fn revoke_key(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<uuid::Uuid>,
) -> Response {
    if let Err(error) = authorize_admin(&state, &headers) {
        return admin_error(error);
    }
    match state.control.revoke_key(id).await {
        Ok(true) => StatusCode::NO_CONTENT.into_response(),
        Ok(false) => admin_error(ApiError::new(
            StatusCode::NOT_FOUND,
            ErrorKind::NotFound,
            "key_not_found",
            format!("no live key `{id}`"),
        )),
        Err(error) => admin_error(ApiError::internal(error.to_string())),
    }
}

#[derive(Debug, Deserialize)]
struct UsageQuery {
    #[serde(default)]
    tenant: Option<String>,
    #[serde(default = "default_days")]
    days: i64,
}

fn default_days() -> i64 {
    30
}

/// Daily usage per tenant and model from the ledger.
async fn usage(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<UsageQuery>,
) -> Response {
    if let Err(error) = authorize_admin(&state, &headers) {
        return admin_error(error);
    }
    let since = chrono::Utc::now() - chrono::Duration::days(query.days.clamp(1, 366));
    match state
        .control
        .store()
        .daily_usage(query.tenant.as_deref(), since)
        .await
    {
        Ok(rows) => Json(json!({ "usage": rows })).into_response(),
        Err(error) => admin_error(ApiError::internal(error.to_string())),
    }
}
