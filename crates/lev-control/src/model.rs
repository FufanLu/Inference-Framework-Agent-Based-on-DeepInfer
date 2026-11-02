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
