//! The audit log: what every request asked for, where it ran and what it
//! produced. It keeps the exact worker inputs (token IDs, sampling parameters
//! including the seed, constraint) so that a request can be replayed and its
//! output compared token by token.

use std::collections::VecDeque;
use std::sync::Arc;

use lev_proto::pb;
use parking_lot::Mutex;
use serde::Serialize;

/// Inputs that reproduce a request on a worker.
#[derive(Debug, Clone)]
pub struct ReplayInputs {
    pub model: String,
    pub generate: pb::GenerateRequest,
}

#[derive(Debug, Clone, Serialize)]
pub struct AuditEntry {
    pub request_id: String,
    pub response_id: String,
    pub protocol: &'static str,
    pub model: String,
    pub tenant: String,
    pub received_at_ms: u64,
    pub policy: Option<String>,
    pub plan: Option<serde_json::Value>,
    pub worker_id: Option<String>,
    pub prefill_worker_id: Option<String>,
    pub attempts: u32,
    pub prompt_tokens: u32,
    pub cached_tokens: u32,
    pub completion_tokens: u32,
    pub seed: u64,
    pub ttft_ms: Option<f64>,
    pub tpot_ms: Option<f64>,
    pub e2e_ms: f64,
    pub outcome: &'static str,
    pub finish_reason: Option<String>,
    pub error: Option<String>,
    pub spec_draft_tokens: u64,
    pub spec_accepted_tokens: u64,
    /// Where the time went: gateway stages, the worker's queue and prefill,
    /// decode, and the response tail.
    pub timeline: Vec<crate::pipeline::Stage>,
    #[serde(skip)]
    pub replay: Option<ReplayInputs>,
    #[serde(skip)]
    pub output_token_ids: Vec<u32>,
}

/// An in-memory ring buffer of recent requests.
#[derive(Debug)]
pub struct AuditLog {
    capacity: usize,
    entries: Mutex<VecDeque<Arc<AuditEntry>>>,
}

impl AuditLog {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            entries: Mutex::new(VecDeque::new()),
        }
    }

    pub fn push(&self, entry: AuditEntry) {
        let mut entries = self.entries.lock();
        if entries.len() == self.capacity {
            entries.pop_front();
        }
        entries.push_back(Arc::new(entry));
    }

    /// The most recent entries, newest first.
    pub fn recent(&self, limit: usize) -> Vec<Arc<AuditEntry>> {
        self.entries
            .lock()
            .iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }

    pub fn get(&self, request_id: &str) -> Option<Arc<AuditEntry>> {
        self.entries
            .lock()
            .iter()
            .rev()
            .find(|entry| entry.request_id == request_id || entry.response_id == request_id)
            .cloned()
    }
}
