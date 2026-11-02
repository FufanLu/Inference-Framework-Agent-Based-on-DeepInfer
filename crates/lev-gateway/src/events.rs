//! Live events for dashboards: routing decisions, request completions and
//! worker status, fanned out to every `/admin/live` subscriber.

use lev_router::{CandidateScore, Plan, Policy};
use serde::Serialize;
use tokio::sync::broadcast;

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LiveEvent {
    Route {
        request_id: String,
        protocol: &'static str,
        model: String,
        policy: Policy,
        plan: Plan,
        prompt_tokens: usize,
        candidates: Vec<CandidateScore>,
        affinity: bool,
        attempt: u32,
        at_ms: u64,
    },
    RequestDone {
        request_id: String,
        model: String,
        tenant: String,
        worker_id: Option<String>,
        outcome: &'static str,
        prompt_tokens: u32,
        cached_tokens: u32,
        completion_tokens: u32,
        ttft_ms: Option<f64>,
        e2e_ms: f64,
        at_ms: u64,
    },
    Worker {
        worker_id: String,
        endpoint: String,
        status: &'static str,
        detail: String,
        at_ms: u64,
    },
}

/// Fan-out of [`LiveEvent`]s. Slow subscribers miss events rather than
/// slowing the request path.
#[derive(Debug, Clone)]
pub struct EventBus {
    sender: broadcast::Sender<LiveEvent>,
}

impl EventBus {
    pub fn new(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity);
        Self { sender }
    }

    pub fn publish(&self, event: LiveEvent) {
        // No subscribers is fine.
        let _ = self.sender.send(event);
    }

    pub fn subscribe(&self) -> broadcast::Receiver<LiveEvent> {
        self.sender.subscribe()
    }
}

/// Milliseconds since the Unix epoch.
pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as u64)
        .unwrap_or(0)
}
