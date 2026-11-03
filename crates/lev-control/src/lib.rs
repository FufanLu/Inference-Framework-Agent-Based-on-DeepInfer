//! Leviathan control plane: who may call the API, how much, and what they used.
//!
//! - [`store`]: tenants, hashed API keys and the usage ledger, in memory or in
//!   Postgres ([`pg`]).
//! - [`limiter`]: RPM / TPM token buckets, in-process or in Redis so that every
//!   gateway replica enforces one budget.
//! - [`ControlPlane`]: authentication with a short-lived key cache, admission
//!   (RPM, TPM, monthly quota) and usage settlement, batched into the ledger
//!   off the request path.

pub mod keys;
pub mod limiter;
pub mod model;
#[cfg(feature = "postgres")]
pub mod pg;
pub mod store;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::{DateTime, Datelike, TimeZone, Utc};
use parking_lot::Mutex;
use tokio::sync::mpsc;
use uuid::Uuid;

pub use limiter::{Limiter, LocalLimiter};
pub use model::{
    ApiKeyInfo, CreatedKey, DailyUsage, JobCounts, JobOutcome, JobRecord, JobStatus, Limits,
    Tenant, UsageRecord, UsageTotals,
};
pub use store::{MemoryStore, Store};

const KEY_CACHE_TTL: Duration = Duration::from_secs(30);
const MONTH_REFRESH: Duration = Duration::from_secs(30);
const FLUSH_INTERVAL: Duration = Duration::from_millis(500);
const FLUSH_BATCH: usize = 512;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AuthError {
    #[error("missing API key")]
    Missing,
    #[error("invalid API key")]
    Invalid,
    #[error("tenant `{0}` is disabled")]
    Disabled(String),
    #[error("control plane unavailable: {0}")]
    Unavailable(String),
}

/// Why a request was not admitted.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Throttle {
    #[error("rate limit of {limit} requests per minute exceeded")]
    Requests { limit: u64, retry_after: Duration },
    #[error("rate limit of {limit} tokens per minute exceeded")]
    Tokens { limit: u64, retry_after: Duration },
    #[error("monthly token quota of {quota} exhausted ({used} used)")]
    Quota { quota: u64, used: u64 },
}

impl Throttle {
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            Throttle::Requests { retry_after, .. } | Throttle::Tokens { retry_after, .. } => {
                Some(*retry_after)
            }
            Throttle::Quota { .. } => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct MonthUsage {
    month: (i32, u32),
    tokens: u64,
    refreshed: Instant,
}

fn month_start(now: DateTime<Utc>) -> DateTime<Utc> {
    Utc.with_ymd_and_hms(now.year(), now.month(), 1, 0, 0, 0)
        .single()
        .unwrap_or(now)
}

/// Authentication, admission and metering for the gateway.
#[derive(Debug)]
pub struct ControlPlane {
    store: Arc<dyn Store>,
    limiter: Arc<dyn Limiter>,
    /// Tenant used for unauthenticated requests; `None` requires a key.
    anonymous: Option<Tenant>,
    key_cache: Mutex<HashMap<keys::KeyHash, (Tenant, Instant)>>,
    month: Mutex<HashMap<String, MonthUsage>>,
    usage: mpsc::Sender<UsageRecord>,
}

impl ControlPlane {
    /// Start the control plane and its ledger flusher.
    pub fn new(
        store: Arc<dyn Store>,
        limiter: Arc<dyn Limiter>,
        anonymous: Option<Tenant>,
    ) -> Arc<Self> {
        let (usage, receiver) = mpsc::channel(16_384);
        tokio::spawn(flush_usage(Arc::clone(&store), receiver));
        Arc::new(Self {
            store,
            limiter,
            anonymous,
            key_cache: Mutex::new(HashMap::new()),
            month: Mutex::new(HashMap::new()),
            usage,
        })
    }

    pub fn store(&self) -> &Arc<dyn Store> {
        &self.store
    }

    pub fn requires_key(&self) -> bool {
        self.anonymous.is_none()
    }

    /// Make sure a configured static key and its tenant exist.
    pub async fn ensure_key(&self, tenant: &Tenant, key: &str, name: &str) -> anyhow::Result<()> {
        self.store.upsert_tenant(tenant).await?;
        self.store
            .insert_key(
                &tenant.id,
                name,
                &keys::display_prefix(key),
                &keys::hash(key),
            )
            .await?;
        Ok(())
    }

    /// The tenant of a presented API key.
    pub async fn authenticate(&self, presented: Option<&str>) -> Result<Tenant, AuthError> {
        let Some(key) = presented.map(str::trim).filter(|key| !key.is_empty()) else {
            return self.anonymous.clone().ok_or(AuthError::Missing);
        };
        let hash = keys::hash(key);
        if let Some((tenant, at)) = self.key_cache.lock().get(&hash)
            && at.elapsed() < KEY_CACHE_TTL
        {
            return check_enabled(tenant.clone());
        }
        match self.store.lookup_key(&hash).await {
            Ok(Some((_, tenant))) => {
                self.key_cache
                    .lock()
                    .insert(hash, (tenant.clone(), Instant::now()));
                check_enabled(tenant)
            }
            // An unknown key falls back to anonymous access when it is enabled,
            // so that SDKs that always send a placeholder key keep working.
            Ok(None) => self.anonymous.clone().ok_or(AuthError::Invalid),
            Err(error) => Err(AuthError::Unavailable(error.to_string())),
        }
    }

    /// Admit one request of `tenant` or say why not.
    pub async fn admit(&self, tenant: &Tenant) -> Result<(), Throttle> {
        if let Some(quota) = tenant.limits.monthly_token_quota {
            let used = self.month_tokens(&tenant.id).await;
            if used >= quota {
                return Err(Throttle::Quota { quota, used });
            }
        }
        if let Some(tpm) = tenant.limits.tpm {
            let balance = self
                .limiter
                .available(&format!("{}:tpm", tenant.id), tpm)
                .await;
            if balance <= 0.0 {
                return Err(Throttle::Tokens {
                    limit: tpm,
                    retry_after: limiter::wait_for(balance, 1.0, tpm),
                });
            }
        }
        if let Some(rpm) = tenant.limits.rpm
            && let Err(retry_after) = self
                .limiter
                .acquire(&format!("{}:rpm", tenant.id), 1.0, rpm)
                .await
        {
            return Err(Throttle::Requests {
                limit: rpm,
                retry_after,
            });
        }
        Ok(())
    }

    /// Admit an asynchronous job at submission. Only the monthly quota
    /// applies: RPM and TPM limit how fast requests hit the engines, and the
    /// job queue already decides that. Usage is still charged when the job
    /// runs.
    pub async fn admit_job(&self, tenant: &Tenant) -> Result<(), Throttle> {
        if let Some(quota) = tenant.limits.monthly_token_quota {
            let used = self.month_tokens(&tenant.id).await;
            if used >= quota {
                return Err(Throttle::Quota { quota, used });
            }
        }
        Ok(())
    }

    /// A tenant by ID, including the anonymous tenant: job runners act on
    /// behalf of the tenant that submitted the job.
    pub async fn tenant(&self, id: &str) -> anyhow::Result<Option<Tenant>> {
        if let Some(anonymous) = self.anonymous.as_ref().filter(|t| t.id == id) {
            return Ok(Some(anonymous.clone()));
        }
        self.store.get_tenant(id).await
    }

    /// Charge a finished request: TPM bucket, monthly counter, ledger.
    pub async fn settle(&self, tenant: &Tenant, record: UsageRecord) {
        let tokens = record.billable_tokens();
        if let Some(tpm) = tenant.limits.tpm {
            self.limiter
                .debit(&format!("{}:tpm", tenant.id), tokens as f64, tpm)
                .await;
        }
        if let Some(entry) = self.month.lock().get_mut(&tenant.id) {
            entry.tokens += tokens;
        }
        if self.usage.try_send(record).is_err() {
            tracing::warn!("usage ledger queue is full; dropping a usage record");
        }
    }

    async fn month_tokens(&self, tenant_id: &str) -> u64 {
        let now = Utc::now();
        let month = (now.year(), now.month());
        if let Some(entry) = self.month.lock().get(tenant_id)
            && entry.month == month
            && entry.refreshed.elapsed() < MONTH_REFRESH
        {
            return entry.tokens;
        }
        let tokens = match self.store.usage_since(tenant_id, month_start(now)).await {
            Ok(totals) => totals.billable_tokens(),
            Err(error) => {
                tracing::warn!(%error, tenant_id, "could not read monthly usage");
                return self.month.lock().get(tenant_id).map_or(0, |e| e.tokens);
            }
        };
        self.month.lock().insert(
            tenant_id.to_string(),
            MonthUsage {
                month,
                tokens,
                refreshed: Instant::now(),
            },
        );
        tokens
    }

    // ------------------------------------------------------------ admin

    pub async fn upsert_tenant(&self, tenant: &Tenant) -> anyhow::Result<()> {
        self.store.upsert_tenant(tenant).await?;
        // Cached keys carry the old limits.
        self.key_cache
            .lock()
            .retain(|_, (cached, _)| cached.id != tenant.id);
        Ok(())
    }

    pub async fn create_key(&self, tenant_id: &str, name: &str) -> anyhow::Result<CreatedKey> {
        let key = keys::generate();
        let info = self
            .store
            .insert_key(
                tenant_id,
                name,
                &keys::display_prefix(&key),
                &keys::hash(&key),
            )
            .await?;
        Ok(CreatedKey { info, key })
    }

    pub async fn revoke_key(&self, id: Uuid) -> anyhow::Result<bool> {
        let revoked = self.store.revoke_key(id).await?;
        // The cache is keyed by hash, which the listing does not expose: drop it.
        self.key_cache.lock().clear();
        Ok(revoked)
    }

    /// Every tenant with its usage this month.
    pub async fn tenants_with_usage(&self) -> anyhow::Result<Vec<(Tenant, UsageTotals)>> {
        let since = month_start(Utc::now());
        let mut out = Vec::new();
        for tenant in self.store.list_tenants().await? {
            let usage = self.store.usage_since(&tenant.id, since).await?;
            out.push((tenant, usage));
        }
        Ok(out)
    }
}

fn check_enabled(tenant: Tenant) -> Result<Tenant, AuthError> {
    if tenant.disabled {
        Err(AuthError::Disabled(tenant.id))
    } else {
        Ok(tenant)
    }
}

async fn flush_usage(store: Arc<dyn Store>, mut receiver: mpsc::Receiver<UsageRecord>) {
    let mut batch = Vec::with_capacity(FLUSH_BATCH);
    loop {
        let first = receiver.recv().await;
        let Some(first) = first else { break };
        batch.push(first);
        let deadline = tokio::time::Instant::now() + FLUSH_INTERVAL;
        while batch.len() < FLUSH_BATCH {
            match tokio::time::timeout_at(deadline, receiver.recv()).await {
                Ok(Some(record)) => batch.push(record),
                Ok(None) | Err(_) => break,
            }
        }
        if let Err(error) = store.record_usage(&batch).await {
            tracing::error!(%error, records = batch.len(), "writing the usage ledger failed");
        }
        batch.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tenant(id: &str, limits: Limits) -> Tenant {
        Tenant {
            id: id.into(),
            name: id.into(),
            limits,
            disabled: false,
        }
    }

    fn plane(anonymous: Option<Tenant>) -> Arc<ControlPlane> {
        ControlPlane::new(
            Arc::new(MemoryStore::new()),
            Arc::new(LocalLimiter::new()),
            anonymous,
        )
    }

    fn usage(tenant: &str, tokens: u64) -> UsageRecord {
        UsageRecord {
            tenant_id: tenant.into(),
            request_id: "r".into(),
            model: "m".into(),
            prompt_tokens: tokens,
            cached_tokens: 0,
            completion_tokens: 0,
            at: Utc::now(),
        }
    }

    #[tokio::test]
    async fn keys_authenticate_their_tenant() {
        let plane = plane(None);
        let acme = tenant("acme", Limits::default());
        plane.upsert_tenant(&acme).await.unwrap();
        let created = plane.create_key("acme", "ci").await.unwrap();
        assert_eq!(plane.authenticate(Some(&created.key)).await.unwrap(), acme);
        assert_eq!(
            plane.authenticate(Some("lev-nope")).await,
            Err(AuthError::Invalid)
        );
        assert_eq!(plane.authenticate(None).await, Err(AuthError::Missing));
        assert!(plane.revoke_key(created.info.id).await.unwrap());
        assert_eq!(
            plane.authenticate(Some(&created.key)).await,
            Err(AuthError::Invalid)
        );
    }

    #[tokio::test]
    async fn anonymous_access_when_enabled() {
        let anon = tenant("anonymous", Limits::default());
        let plane = plane(Some(anon.clone()));
        assert_eq!(plane.authenticate(None).await.unwrap(), anon);
        assert_eq!(
            plane.authenticate(Some("sk-placeholder")).await.unwrap(),
            anon
        );
    }

    #[tokio::test]
    async fn rpm_limit_throttles_with_retry_after() {
        let plane = plane(None);
        let t = tenant(
            "t",
            Limits {
                rpm: Some(2),
                ..Limits::default()
            },
        );
        assert!(plane.admit(&t).await.is_ok());
        assert!(plane.admit(&t).await.is_ok());
        let Err(Throttle::Requests { retry_after, .. }) = plane.admit(&t).await else {
            panic!("expected an RPM throttle");
        };
        assert!(retry_after > Duration::from_secs(20));
    }

    #[tokio::test]
    async fn tpm_is_charged_after_the_request() {
        let plane = plane(None);
        let t = tenant(
            "t",
            Limits {
                tpm: Some(1_000),
                ..Limits::default()
            },
        );
        assert!(plane.admit(&t).await.is_ok());
        plane.settle(&t, usage("t", 5_000)).await;
        assert!(matches!(
            plane.admit(&t).await,
            Err(Throttle::Tokens { .. })
        ));
    }

    #[tokio::test]
    async fn monthly_quota_counts_the_ledger() {
        let plane = plane(None);
        let t = tenant(
            "t",
            Limits {
                monthly_token_quota: Some(100),
                ..Limits::default()
            },
        );
        plane.upsert_tenant(&t).await.unwrap();
        plane.store().record_usage(&[usage("t", 60)]).await.unwrap();
        assert!(plane.admit(&t).await.is_ok());
        plane.settle(&t, usage("t", 50)).await;
        assert!(matches!(
            plane.admit(&t).await,
            Err(Throttle::Quota {
                quota: 100,
                used: 110
            })
        ));
    }

    #[tokio::test]
    async fn settled_usage_reaches_the_ledger() {
        let plane = plane(None);
        let t = tenant("t", Limits::default());
        plane.upsert_tenant(&t).await.unwrap();
        plane.settle(&t, usage("t", 7)).await;
        for _ in 0..50 {
            let totals = plane
                .store()
                .usage_since("t", month_start(Utc::now()))
                .await
                .unwrap();
            if totals.requests == 1 {
                assert_eq!(totals.prompt_tokens, 7);
                return;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        panic!("usage was not flushed");
    }

    #[tokio::test]
    async fn disabled_tenants_are_rejected() {
        let plane = plane(None);
        let mut t = tenant("t", Limits::default());
        t.disabled = true;
        plane.upsert_tenant(&t).await.unwrap();
        let created = plane.create_key("t", "k").await.unwrap();
        assert_eq!(
            plane.authenticate(Some(&created.key)).await,
            Err(AuthError::Disabled("t".into()))
        );
    }
}
