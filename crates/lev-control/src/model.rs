//! Control plane data types.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Per-tenant limits; `None` means unlimited.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Limits {
    /// Requests per minute.
    pub rpm: Option<u64>,
    /// Tokens (prompt + completion) per minute.
    pub tpm: Option<u64>,
    /// Tokens per calendar month (UTC).
    pub monthly_token_quota: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tenant {
    pub id: String,
    pub name: String,
    #[serde(flatten)]
    pub limits: Limits,
    #[serde(default)]
    pub disabled: bool,
}

/// An API key as listed: never the secret itself.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ApiKeyInfo {
    pub id: Uuid,
    pub tenant_id: String,
    pub name: String,
    /// Displayable start of the key, e.g. `lev-3fA9xQ…`.
    pub prefix: String,
    pub created_at: DateTime<Utc>,
    pub revoked_at: Option<DateTime<Utc>>,
}

/// A newly created key: the only time the secret is returned.
#[derive(Debug, Clone, Serialize)]
pub struct CreatedKey {
    #[serde(flatten)]
    pub info: ApiKeyInfo,
    pub key: String,
}

/// Usage of one finished request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct UsageRecord {
    pub tenant_id: String,
    pub request_id: String,
    pub model: String,
    pub prompt_tokens: u64,
    pub cached_tokens: u64,
    pub completion_tokens: u64,
    pub at: DateTime<Utc>,
}

impl UsageRecord {
    /// Tokens charged against TPM and the monthly quota.
    pub fn billable_tokens(&self) -> u64 {
        self.prompt_tokens + self.completion_tokens
    }
}

/// Aggregated usage.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct UsageTotals {
    pub requests: u64,
    pub prompt_tokens: u64,
    pub cached_tokens: u64,
    pub completion_tokens: u64,
}

impl UsageTotals {
    pub fn add(&mut self, record: &UsageRecord) {
        self.requests += 1;
        self.prompt_tokens += record.prompt_tokens;
        self.cached_tokens += record.cached_tokens;
        self.completion_tokens += record.completion_tokens;
    }

    pub fn billable_tokens(&self) -> u64 {
        self.prompt_tokens + self.completion_tokens
    }
}

/// Usage of one tenant and model on one UTC day.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DailyUsage {
    pub day: chrono::NaiveDate,
    pub tenant_id: String,
    pub model: String,
    #[serde(flatten)]
    pub totals: UsageTotals,
}

/// Where an asynchronous job is in its life.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    /// Accepted and in the queue.
    Queued,
    /// Claimed by a runner.
    Running,
    Succeeded,
    Failed,
}

impl JobStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            JobStatus::Queued => "queued",
            JobStatus::Running => "running",
            JobStatus::Succeeded => "succeeded",
            JobStatus::Failed => "failed",
        }
    }

    pub fn is_terminal(self) -> bool {
        matches!(self, JobStatus::Succeeded | JobStatus::Failed)
    }
}

impl std::str::FromStr for JobStatus {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "queued" => Ok(JobStatus::Queued),
            "running" => Ok(JobStatus::Running),
            "succeeded" => Ok(JobStatus::Succeeded),
            "failed" => Ok(JobStatus::Failed),
            other => Err(format!("unknown job status `{other}`")),
        }
    }
}

/// An asynchronous inference job: its status and, once finished, its result.
/// The request body itself travels in the queue, not in the store.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct JobRecord {
    pub id: Uuid,
    pub tenant_id: String,
    /// The caller's own identifier, echoed back.
    pub custom_id: Option<String>,
    /// `/v1/chat/completions`, `/v1/responses` or `/v1/messages`.
    pub endpoint: String,
    pub status: JobStatus,
    /// Times a runner has claimed the job (more than one after a redelivery
    /// or a retry).
    pub attempts: u32,
    /// The gateway request that ran the job (audit log key).
    pub request_id: Option<String>,
    /// HTTP status of the protocol response.
    pub response_status: Option<u16>,
    /// The protocol response body (or error body).
    pub response: Option<serde_json::Value>,
    pub error: Option<String>,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub finished_at: Option<DateTime<Utc>>,
}

impl JobRecord {
    pub fn queued(id: Uuid, tenant_id: &str, custom_id: Option<String>, endpoint: &str) -> Self {
        Self {
            id,
            tenant_id: tenant_id.to_string(),
            custom_id,
            endpoint: endpoint.to_string(),
            status: JobStatus::Queued,
            attempts: 0,
            request_id: None,
            response_status: None,
            response: None,
            error: None,
            created_at: Utc::now(),
            started_at: None,
            finished_at: None,
        }
    }
}

/// The outcome a runner records for a job.
#[derive(Debug, Clone, PartialEq)]
pub struct JobOutcome {
    pub status: JobStatus,
    pub request_id: Option<String>,
    pub response_status: Option<u16>,
    pub response: Option<serde_json::Value>,
    pub error: Option<String>,
}

/// Jobs per status.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct JobCounts {
    pub queued: u64,
    pub running: u64,
    pub succeeded: u64,
    pub failed: u64,
}
