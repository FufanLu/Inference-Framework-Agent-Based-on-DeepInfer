//! Asynchronous jobs: absorb bursts in a queue, run them as capacity frees up.
//!
//! Interactive requests go straight to the engines; a burst of them queues
//! inside the engines and every request's TTFT pays for it. A job is accepted
//! at once (`202`), stored as `queued`, and put on the job queue (Kafka, or an
//! in-process channel). Runners take jobs off the queue only while the
//! cluster has room, measured with the router's own TTFT model
//! ([`lev_router::Router::min_queue_delay`]), and never more than
//! `jobs.max_in_flight` at once. Interactive traffic is never queued behind
//! jobs: when it loads the cluster, runners pause and the backlog waits in
//! the queue instead of in the engines.
//!
//! A job runs through the same pipeline as an HTTP request (recipe
//! conversion, routing, metering, audit), non-streaming, as the tenant that
//! submitted it.

#[cfg(feature = "kafka")]
pub mod kafka;
pub mod offsets;
pub mod queue;

use std::sync::Arc;
use std::sync::atomic::{AtomicI32, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use axum::http::HeaderMap;
use axum::response::Response;
use deepseek_recipe::anthropic::MessagesRequest;
use deepseek_recipe::openai::{ChatCompletionRequest, ResponsesRequest};
use deepseek_recipe::request::ProtocolRequest;
use lev_control::{JobOutcome, JobRecord, JobStatus, Tenant};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use tokio::sync::Notify;
use uuid::Uuid;

use crate::config::JobsConfig;
use crate::error::ApiError;
use crate::events::now_ms;
use crate::pipeline::{self, ProtocolChunk};
use crate::protocol::ProtocolInfo;
use crate::state::AppState;
pub use queue::{Endpoint, JobMessage, JobQueue, MemoryQueue, QueueFull};

/// How often a waiting runner re-checks the cluster's headroom.
const HEADROOM_POLL: Duration = Duration::from_millis(10);
/// Largest protocol response kept as a job result.
const MAX_RESPONSE_BYTES: usize = 32 << 20;

/// One job in a submission.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JobSubmission {
    /// `/v1/chat/completions`, `/v1/responses` or `/v1/messages`.
    pub endpoint: String,
    /// The protocol request. `stream` is ignored: jobs return one response.
    pub body: Value,
    #[serde(default)]
    pub custom_id: Option<String>,
}

/// The job queue and the runner's counters.
#[derive(Debug)]
pub struct Jobs {
    pub queue: Arc<dyn JobQueue>,
    config: JobsConfig,
    in_flight: AtomicUsize,
    /// Runners waiting for headroom right now (0 or 1 per gateway).
    waiting_for_headroom: AtomicUsize,
    /// Tunable at runtime (`PUT /admin/jobs`).
    max_in_flight: AtomicUsize,
    max_queue_delay_us: AtomicU64,
    priority: AtomicI32,
    /// Signalled when a job finishes or the limits change.
    slot_freed: Notify,
}

/// Runtime changes to the runner's limits.
#[derive(Debug, Clone, Copy, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunnerUpdate {
    pub max_in_flight: Option<usize>,
    pub max_queue_delay_ms: Option<f64>,
    pub priority: Option<i32>,
}

impl Jobs {
    pub async fn connect(config: &JobsConfig) -> anyhow::Result<Self> {
        let queue: Arc<dyn JobQueue> = match config.queue.as_str() {
            "memory" | "" => Arc::new(MemoryQueue::new(config.memory_capacity)),
            #[cfg(feature = "kafka")]
            "kafka" => Arc::new(kafka::KafkaQueue::connect(&config.kafka, config.run).await?),
            #[cfg(not(feature = "kafka"))]
            "kafka" => anyhow::bail!("this gateway was built without the `kafka` feature"),
            other => anyhow::bail!("unknown job queue `{other}`: use `memory` or `kafka`"),
        };
        Ok(Self {
            queue,
            config: config.clone(),
            in_flight: AtomicUsize::new(0),
            waiting_for_headroom: AtomicUsize::new(0),
            max_in_flight: AtomicUsize::new(config.max_in_flight.max(1)),
            max_queue_delay_us: AtomicU64::new(config.max_queue_delay_ms.saturating_mul(1_000)),
            priority: AtomicI32::new(config.priority),
            slot_freed: Notify::new(),
        })
    }

    pub fn max_in_flight(&self) -> usize {
        self.max_in_flight.load(Ordering::Relaxed)
    }

    pub fn max_queue_delay(&self) -> Duration {
        Duration::from_micros(self.max_queue_delay_us.load(Ordering::Relaxed))
    }

    pub fn priority(&self) -> i32 {
        self.priority.load(Ordering::Relaxed)
    }

    /// Change the runner's limits; takes effect for the next job.
    pub fn update(&self, update: RunnerUpdate) {
        if let Some(priority) = update.priority {
            self.priority.store(priority, Ordering::Relaxed);
        }
        if let Some(max) = update.max_in_flight {
            self.max_in_flight.store(max.max(1), Ordering::Relaxed);
        }
        if let Some(ms) = update.max_queue_delay_ms {
            self.max_queue_delay_us
                .store((ms.max(0.0) * 1e3) as u64, Ordering::Relaxed);
        }
        self.slot_freed.notify_waiters();
    }

    /// Wait until fewer than `max_in_flight` jobs run, then take a slot.
    async fn take_slot(&self) {
        loop {
            let notified = self.slot_freed.notified();
            let current = self.in_flight.load(Ordering::Acquire);
            if current < self.max_in_flight()
                && self
                    .in_flight
                    .compare_exchange(current, current + 1, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
            {
                metrics::gauge!("lev_jobs_in_flight").increment(1.0);
                return;
            }
            // Re-check now and then: a limit change notifies, but cheaply
            // covering a missed wakeup costs nothing.
            let _ = tokio::time::timeout(Duration::from_millis(100), notified).await;
        }
    }

    fn release_slot(&self) {
        self.in_flight.fetch_sub(1, Ordering::AcqRel);
        metrics::gauge!("lev_jobs_in_flight").decrement(1.0);
        self.slot_freed.notify_waiters();
    }

    pub fn config(&self) -> &JobsConfig {
        &self.config
    }

    pub fn in_flight(&self) -> usize {
        self.in_flight.load(Ordering::Relaxed)
    }
}

/// The public view of a job.
pub fn job_json(job: &JobRecord) -> Value {
    json!({
        "id": job.id,
        "object": "leviathan.job",
        "status": job.status,
        "endpoint": job.endpoint,
        "custom_id": job.custom_id,
        "attempts": job.attempts,
        "request_id": job.request_id,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "queue_wait_ms": job.started_at.map(|at| (at - job.created_at).num_milliseconds()),
        "response": job.response_status.map(|status| json!({
            "status_code": status,
            "body": job.response,
        })),
        "error": job.error,
    })
}

/// Accept jobs: validate, store as queued, enqueue.
pub async fn submit(
    state: &AppState,
    tenant: &Tenant,
    submissions: Vec<JobSubmission>,
) -> Result<Vec<JobRecord>, ApiError> {
    let Some(jobs) = state.jobs.as_ref() else {
        return Err(ApiError::new(
            axum::http::StatusCode::NOT_FOUND,
            crate::error::ErrorKind::NotFound,
            "jobs_disabled",
            "asynchronous jobs are not enabled on this gateway",
        ));
    };
    if submissions.is_empty() {
        return Err(ApiError::bad_request("empty_batch", "no jobs to submit"));
    }
    if submissions.len() > jobs.config.max_batch {
        return Err(ApiError::bad_request(
            "batch_too_large",
            format!(
                "{} jobs in one submission; the limit is {}",
                submissions.len(),
                jobs.config.max_batch
            ),
        ));
    }
    let mut records = Vec::with_capacity(submissions.len());
    let mut messages = Vec::with_capacity(submissions.len());
    let submitted_at_ms = now_ms();
    for (index, submission) in submissions.into_iter().enumerate() {
        let endpoint = Endpoint::parse(&submission.endpoint).ok_or_else(|| {
            ApiError::bad_request(
                "invalid_endpoint",
                format!(
                    "job {index}: endpoint `{}` is not one of /v1/chat/completions, \
                     /v1/responses, /v1/messages",
                    submission.endpoint
                ),
            )
        })?;
        let model = submission
            .body
            .get("model")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                ApiError::bad_request("missing_model", format!("job {index}: `model` is required"))
            })?;
        if state.models.resolve(model).is_none() {
            return Err(ApiError::new(
                axum::http::StatusCode::NOT_FOUND,
                crate::error::ErrorKind::NotFound,
                "model_not_found",
                format!("job {index}: model `{model}` is not served by this gateway"),
            ));
        }
        let id = Uuid::now_v7();
        records.push(JobRecord::queued(
            id,
            &tenant.id,
            submission.custom_id,
            endpoint.as_str(),
        ));
        messages.push(JobMessage {
            id,
            tenant_id: tenant.id.clone(),
            endpoint,
            body: submission.body,
            submitted_at_ms,
        });
    }
    pipeline::admit_job(state, tenant).await?;
    let store = state.control.store();
    store
        .insert_jobs(&records)
        .await
        .map_err(|error| ApiError::overloaded(format!("storing jobs failed: {error}")))?;
    if let Err(error) = jobs.queue.enqueue(&messages).await {
        // Not queued: they will never run.
        let outcome = JobOutcome {
            status: JobStatus::Failed,
            request_id: None,
            response_status: None,
            response: None,
            error: Some(format!("enqueueing failed: {error}")),
        };
        for record in &records {
            let _ = store.finish_job(record.id, &outcome).await;
        }
        let mut api_error = ApiError::overloaded(error.to_string());
        if error.downcast_ref::<QueueFull>().is_some() {
            api_error.retry_after = Some(Duration::from_secs(5));
        }
        return Err(api_error);
    }
    metrics::counter!("lev_jobs_submitted_total", "tenant" => tenant.id.clone())
        .increment(records.len() as u64);
    Ok(records)
}

/// The runner: take jobs off the queue while the cluster has room.
pub async fn run(state: AppState) {
    let Some(jobs) = state.jobs.clone() else {
        return;
    };
    let config = jobs.config.clone();
    tracing::info!(
        queue = jobs.queue.backend(),
        max_in_flight = jobs.max_in_flight(),
        max_queue_delay_ms = config.max_queue_delay_ms,
        "job runner started"
    );
    loop {
        let delivery = match jobs.queue.next().await {
            Ok(delivery) => delivery,
            Err(error) => {
                tracing::warn!(%error, "reading the job queue failed");
                tokio::time::sleep(Duration::from_secs(1)).await;
                continue;
            }
        };
        // Holding one delivery while every slot is busy is a prefetch of one.
        jobs.take_slot().await;
        let slot = Slot(Arc::clone(&jobs));
        let job = Arc::new(delivery.job.clone());
        let claimed = match claim(&state, job.id).await {
            Some(claimed) => claimed,
            None => {
                // Finished already (a redelivery) or unknown to the store.
                delivery.ack();
                continue;
            }
        };
        let tenant = match state.control.tenant(&job.tenant_id).await {
            Ok(Some(tenant)) if !tenant.disabled => tenant,
            Ok(_) => {
                finish(
                    &state,
                    &claimed,
                    failed(format!("tenant `{}` is unknown or disabled", job.tenant_id)),
                )
                .await;
                delivery.ack();
                continue;
            }
            Err(error) => {
                tracing::warn!(%error, job = %job.id, "looking up the job's tenant failed");
                finish(
                    &state,
                    &claimed,
                    failed(format!("tenant lookup failed: {error}")),
                )
                .await;
                delivery.ack();
                continue;
            }
        };
        if claimed.attempts > config.max_attempts.max(1) {
            finish(
                &state,
                &claimed,
                failed(format!("gave up after {} attempts", claimed.attempts - 1)),
            )
            .await;
            delivery.ack();
            continue;
        }

        let model = job
            .body
            .get("model")
            .and_then(Value::as_str)
            .and_then(|name| state.models.resolve(name))
            .map(|model| model.name.clone());
        wait_for_headroom(&state, &jobs, model.as_deref()).await;

        // Poll the request once before looking at the next job: by then it
        // has been tokenized and routed, so the next headroom check sees its
        // load.
        let queue_wait = claimed
            .started_at
            .map(|at| (at - claimed.created_at).to_std().unwrap_or_default())
            .unwrap_or_default();
        metrics::histogram!("lev_job_queue_wait_seconds").record(queue_wait.as_secs_f64());
        let mut first = Box::pin(execute(
            state.clone(),
            Arc::clone(&job),
            tenant.clone(),
            jobs.priority(),
        ));
        let first_result = futures::poll!(first.as_mut());
        let state = state.clone();
        let jobs = Arc::clone(&jobs);
        tokio::spawn(async move {
            let mut executed = match first_result {
                std::task::Poll::Ready(executed) => executed,
                std::task::Poll::Pending => first.await,
            };
            let mut claimed = claimed;
            // Engines failed it (5xx): run it again, up to max_attempts.
            while executed.status >= 500 && claimed.attempts < jobs.config.max_attempts.max(1) {
                tokio::time::sleep(Duration::from_millis(500) * claimed.attempts).await;
                wait_for_headroom(&state, &jobs, model.as_deref()).await;
                match state.control.store().claim_job(job.id).await {
                    Ok(Some(again)) => claimed = again,
                    _ => break,
                }
                executed = execute(
                    state.clone(),
                    Arc::clone(&job),
                    tenant.clone(),
                    jobs.priority(),
                )
                .await;
            }
            let status = if (200..300).contains(&executed.status) {
                JobStatus::Succeeded
            } else {
                JobStatus::Failed
            };
            let error = (status == JobStatus::Failed).then(|| error_message(&executed.body));
            finish(
                &state,
                &claimed,
                JobOutcome {
                    status,
                    request_id: executed.request_id,
                    response_status: Some(executed.status),
                    response: Some(executed.body),
                    error,
                },
            )
            .await;
            delivery.ack();
            drop(slot);
        });
    }
}

/// A running job's slot; released on every exit path.
struct Slot(Arc<Jobs>);

impl Drop for Slot {
    fn drop(&mut self) {
        self.0.release_slot();
    }
}

/// Claim a job, retrying the store briefly. `None` means it must not run.
async fn claim(state: &AppState, id: Uuid) -> Option<JobRecord> {
    for attempt in 0..3u32 {
        match state.control.store().claim_job(id).await {
            Ok(claimed) => return claimed,
            Err(error) => {
                tracing::warn!(%error, job = %id, attempt, "claiming a job failed");
                tokio::time::sleep(Duration::from_millis(200u64 << attempt)).await;
            }
        }
    }
    None
}

async fn wait_for_headroom(state: &AppState, jobs: &Jobs, model: Option<&str>) {
    let Some(model) = model else {
        // Unknown model: the job fails fast without touching an engine.
        return;
    };
    let started = Instant::now();
    let mut waited = false;
    loop {
        let bound = jobs.max_queue_delay();
        // No worker up counts as no room: jobs wait out an outage queued.
        if state
            .router
            .min_queue_delay(model)
            .is_some_and(|delay| delay <= bound)
        {
            break;
        }
        if !waited {
            waited = true;
            jobs.waiting_for_headroom.fetch_add(1, Ordering::Relaxed);
        }
        tokio::time::sleep(HEADROOM_POLL).await;
    }
    if waited {
        jobs.waiting_for_headroom.fetch_sub(1, Ordering::Relaxed);
        metrics::histogram!("lev_job_headroom_wait_seconds")
            .record(started.elapsed().as_secs_f64());
    }
}

async fn finish(state: &AppState, claimed: &JobRecord, outcome: JobOutcome) {
    metrics::counter!("lev_jobs_finished_total", "status" => outcome.status.as_str()).increment(1);
    if let Err(error) = state.control.store().finish_job(claimed.id, &outcome).await {
        tracing::error!(%error, job = %claimed.id, "storing a job outcome failed");
    }
}

fn failed(error: String) -> JobOutcome {
    JobOutcome {
        status: JobStatus::Failed,
        request_id: None,
        response_status: None,
        response: None,
        error: Some(error),
    }
}

fn error_message(body: &Value) -> String {
    body.pointer("/error/message")
        .and_then(Value::as_str)
        .map_or_else(|| body.to_string(), str::to_string)
}

/// A protocol response, as a job result.
struct Executed {
    status: u16,
    body: Value,
    request_id: Option<String>,
}

async fn execute(state: AppState, job: Arc<JobMessage>, tenant: Tenant, priority: i32) -> Executed {
    match job.endpoint {
        Endpoint::ChatCompletions => {
            execute_as::<ChatCompletionRequest>(state, &job.body, tenant, priority).await
        }
        Endpoint::Responses => {
            execute_as::<ResponsesRequest>(state, &job.body, tenant, priority).await
        }
        Endpoint::Messages => {
            execute_as::<MessagesRequest>(state, &job.body, tenant, priority).await
        }
    }
}

async fn execute_as<T>(state: AppState, body: &Value, tenant: Tenant, priority: i32) -> Executed
where
    T: ProtocolRequest + ProtocolInfo + DeserializeOwned,
    ProtocolChunk<T>: serde::Serialize + Send,
{
    let mut body = body.clone();
    if let Some(object) = body.as_object_mut() {
        object.insert("stream".to_string(), Value::Bool(false));
        object.remove("stream_options");
    }
    let bytes = serde_json::to_vec(&body).unwrap_or_default();
    let received_at = Instant::now();
    let response =
        match pipeline::prepare::<T>(&state, &HeaderMap::new(), &bytes, tenant, received_at) {
            Ok(mut prepared) => {
                prepared.generate.priority = priority;
                pipeline::respond::<T>(state, prepared).await
            }
            Err(error) => T::error_response(&error),
        };
    into_executed(response).await
}

async fn into_executed(response: Response) -> Executed {
    let status = response.status().as_u16();
    let request_id = response
        .headers()
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .map(str::to_string);
    let body = match axum::body::to_bytes(response.into_body(), MAX_RESPONSE_BYTES).await {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned())),
        Err(error) => {
            json!({ "error": { "message": format!("reading the response failed: {error}") } })
        }
    };
    Executed {
        status,
        body,
        request_id,
    }
}

/// The runner's state for the admin API.
pub async fn stats(state: &AppState) -> Value {
    let Some(jobs) = state.jobs.as_ref() else {
        return json!({ "enabled": false });
    };
    let counts = state.control.store().job_counts().await.ok();
    let delays: serde_json::Map<String, Value> = state
        .models
        .all()
        .iter()
        .map(|model| {
            (
                model.name.clone(),
                json!(
                    state
                        .router
                        .min_queue_delay(&model.name)
                        .map(|delay| delay.as_secs_f64() * 1e3)
                ),
            )
        })
        .collect();
    json!({
        "enabled": true,
        "queue": jobs.queue.stats().await,
        "runner": {
            "enabled": jobs.config.run,
            "in_flight": jobs.in_flight(),
            "max_in_flight": jobs.max_in_flight(),
            "max_queue_delay_ms": jobs.max_queue_delay().as_secs_f64() * 1e3,
            "priority": jobs.priority(),
            "waiting_for_headroom": jobs.waiting_for_headroom.load(Ordering::Relaxed) > 0,
            "min_queue_delay_ms": delays,
        },
        "jobs": counts,
    })
}
