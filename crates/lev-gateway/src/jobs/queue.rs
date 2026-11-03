//! The job queue: in-process for development, Kafka for deployments.

use std::fmt;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, mpsc};
use uuid::Uuid;

/// The protocol endpoint a job calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Endpoint {
    #[serde(rename = "/v1/chat/completions")]
    ChatCompletions,
    #[serde(rename = "/v1/responses")]
    Responses,
    #[serde(rename = "/v1/messages")]
    Messages,
}

impl Endpoint {
    pub fn as_str(self) -> &'static str {
        match self {
            Endpoint::ChatCompletions => "/v1/chat/completions",
            Endpoint::Responses => "/v1/responses",
            Endpoint::Messages => "/v1/messages",
        }
    }

    pub fn parse(path: &str) -> Option<Self> {
        match path {
            "/v1/chat/completions" => Some(Endpoint::ChatCompletions),
            "/v1/responses" => Some(Endpoint::Responses),
            "/v1/messages" => Some(Endpoint::Messages),
            _ => None,
        }
    }
}

/// One job as it travels through the queue: everything a runner on any
/// gateway needs to run it. Keyed by tenant in Kafka, so one tenant's jobs
/// stay in order on one partition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobMessage {
    pub id: Uuid,
    pub tenant_id: String,
    pub endpoint: Endpoint,
    /// The protocol request body, as submitted.
    pub body: serde_json::Value,
    pub submitted_at_ms: u64,
}

/// A job handed to a runner. Acknowledge it once its outcome is stored;
/// a delivery dropped without an ack is redelivered (Kafka) after a restart.
pub struct Delivery {
    pub job: JobMessage,
    ack: Option<Box<dyn FnOnce() + Send>>,
}

impl Delivery {
    pub fn new(job: JobMessage, ack: Option<Box<dyn FnOnce() + Send>>) -> Self {
        Self { job, ack }
    }

    pub fn ack(mut self) {
        if let Some(ack) = self.ack.take() {
            ack();
        }
    }
}

impl fmt::Debug for Delivery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Delivery")
            .field("job", &self.job.id)
            .finish()
    }
}

/// The queue rejected a submission because it is full.
#[derive(Debug, thiserror::Error)]
#[error("the job queue is full ({capacity} jobs)")]
pub struct QueueFull {
    pub capacity: usize,
}

#[async_trait]
pub trait JobQueue: Send + Sync + fmt::Debug {
    fn backend(&self) -> &'static str;
    /// Enqueue every job or fail.
    async fn enqueue(&self, jobs: &[JobMessage]) -> anyhow::Result<()>;
    /// The next job for this runner; waits while the queue is empty.
    async fn next(&self) -> anyhow::Result<Delivery>;
    /// Backlog and backend details for the admin API.
    async fn stats(&self) -> serde_json::Value;
}

/// An in-process queue: one gateway, lost on restart.
#[derive(Debug)]
pub struct MemoryQueue {
    sender: mpsc::Sender<JobMessage>,
    receiver: Mutex<mpsc::Receiver<JobMessage>>,
    capacity: usize,
}

impl MemoryQueue {
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        let (sender, receiver) = mpsc::channel(capacity);
        Self {
            sender,
            receiver: Mutex::new(receiver),
            capacity,
        }
    }
}

#[async_trait]
impl JobQueue for MemoryQueue {
    fn backend(&self) -> &'static str {
        "memory"
    }

    async fn enqueue(&self, jobs: &[JobMessage]) -> anyhow::Result<()> {
        if jobs.is_empty() {
            return Ok(());
        }
        // All or nothing: reserve every slot first.
        let permits = self
            .sender
            .try_reserve_many(jobs.len())
            .map_err(|_| QueueFull {
                capacity: self.capacity,
            })?;
        for (permit, job) in permits.zip(jobs) {
            permit.send(job.clone());
        }
        Ok(())
    }

    async fn next(&self) -> anyhow::Result<Delivery> {
        let job = self
            .receiver
            .lock()
            .await
            .recv()
            .await
            .ok_or_else(|| anyhow::anyhow!("the job queue is closed"))?;
        Ok(Delivery::new(job, None))
    }

    async fn stats(&self) -> serde_json::Value {
        serde_json::json!({
            "backend": "memory",
            "capacity": self.capacity,
            "backlog": self.capacity - self.sender.capacity(),
        })
    }
}
