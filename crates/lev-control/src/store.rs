//! Persistent control-plane state: tenants, keys and the usage ledger.

use std::collections::HashMap;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use uuid::Uuid;

use crate::keys::KeyHash;
use crate::model::{ApiKeyInfo, DailyUsage, Tenant, UsageRecord, UsageTotals};

#[async_trait]
pub trait Store: Send + Sync + std::fmt::Debug {
    /// The live (not revoked) key with this hash and its tenant.
    async fn lookup_key(&self, hash: &KeyHash) -> anyhow::Result<Option<(ApiKeyInfo, Tenant)>>;
    async fn upsert_tenant(&self, tenant: &Tenant) -> anyhow::Result<()>;
    async fn get_tenant(&self, id: &str) -> anyhow::Result<Option<Tenant>>;
    async fn list_tenants(&self) -> anyhow::Result<Vec<Tenant>>;
    /// Store a key; inserting an existing hash again is a no-op that returns
    /// the existing key.
    async fn insert_key(
        &self,
        tenant_id: &str,
        name: &str,
        prefix: &str,
        hash: &KeyHash,
    ) -> anyhow::Result<ApiKeyInfo>;
    async fn list_keys(&self, tenant_id: Option<&str>) -> anyhow::Result<Vec<ApiKeyInfo>>;
    async fn revoke_key(&self, id: Uuid) -> anyhow::Result<bool>;
    async fn record_usage(&self, records: &[UsageRecord]) -> anyhow::Result<()>;
    async fn usage_since(
        &self,
        tenant_id: &str,
        since: DateTime<Utc>,
    ) -> anyhow::Result<UsageTotals>;
    async fn daily_usage(
        &self,
        tenant_id: Option<&str>,
        since: DateTime<Utc>,
    ) -> anyhow::Result<Vec<DailyUsage>>;
}

#[derive(Debug, Default)]
struct MemoryState {
    tenants: HashMap<String, Tenant>,
    keys: Vec<(KeyHash, ApiKeyInfo)>,
    ledger: Vec<UsageRecord>,
}

/// A store that lives in memory: development and tests.
#[derive(Debug, Default)]
pub struct MemoryStore {
    state: Mutex<MemoryState>,
}

impl MemoryStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl Store for MemoryStore {
    async fn lookup_key(&self, hash: &KeyHash) -> anyhow::Result<Option<(ApiKeyInfo, Tenant)>> {
        let state = self.state.lock();
        let Some((_, info)) = state
            .keys
            .iter()
            .find(|(h, info)| h == hash && info.revoked_at.is_none())
        else {
            return Ok(None);
        };
        Ok(state
            .tenants
            .get(&info.tenant_id)
            .map(|tenant| (info.clone(), tenant.clone())))
    }

    async fn upsert_tenant(&self, tenant: &Tenant) -> anyhow::Result<()> {
        self.state
            .lock()
            .tenants
            .insert(tenant.id.clone(), tenant.clone());
        Ok(())
    }

    async fn get_tenant(&self, id: &str) -> anyhow::Result<Option<Tenant>> {
        Ok(self.state.lock().tenants.get(id).cloned())
    }

    async fn list_tenants(&self) -> anyhow::Result<Vec<Tenant>> {
        let mut tenants: Vec<_> = self.state.lock().tenants.values().cloned().collect();
        tenants.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(tenants)
    }

    async fn insert_key(
        &self,
        tenant_id: &str,
        name: &str,
        prefix: &str,
        hash: &KeyHash,
    ) -> anyhow::Result<ApiKeyInfo> {
        let mut state = self.state.lock();
        anyhow::ensure!(
            state.tenants.contains_key(tenant_id),
            "unknown tenant `{tenant_id}`"
        );
        if let Some((_, info)) = state.keys.iter().find(|(h, _)| h == hash) {
            return Ok(info.clone());
        }
        let info = ApiKeyInfo {
            id: Uuid::new_v4(),
            tenant_id: tenant_id.to_string(),
            name: name.to_string(),
            prefix: prefix.to_string(),
            created_at: Utc::now(),
            revoked_at: None,
        };
        state.keys.push((*hash, info.clone()));
        Ok(info)
    }

    async fn list_keys(&self, tenant_id: Option<&str>) -> anyhow::Result<Vec<ApiKeyInfo>> {
        Ok(self
            .state
            .lock()
            .keys
            .iter()
            .map(|(_, info)| info)
            .filter(|info| tenant_id.is_none_or(|t| info.tenant_id == t))
            .cloned()
            .collect())
    }

    async fn revoke_key(&self, id: Uuid) -> anyhow::Result<bool> {
        let mut state = self.state.lock();
        match state
            .keys
            .iter_mut()
            .find(|(_, info)| info.id == id && info.revoked_at.is_none())
        {
            Some((_, info)) => {
                info.revoked_at = Some(Utc::now());
                Ok(true)
            }
            None => Ok(false),
        }
    }

    async fn record_usage(&self, records: &[UsageRecord]) -> anyhow::Result<()> {
        self.state.lock().ledger.extend_from_slice(records);
        Ok(())
    }

    async fn usage_since(
        &self,
        tenant_id: &str,
        since: DateTime<Utc>,
    ) -> anyhow::Result<UsageTotals> {
        let mut totals = UsageTotals::default();
        for record in self
            .state
            .lock()
            .ledger
            .iter()
            .filter(|r| r.tenant_id == tenant_id && r.at >= since)
        {
            totals.add(record);
        }
        Ok(totals)
    }

    async fn daily_usage(
        &self,
        tenant_id: Option<&str>,
        since: DateTime<Utc>,
    ) -> anyhow::Result<Vec<DailyUsage>> {
        let mut by_key: HashMap<(chrono::NaiveDate, String, String), UsageTotals> = HashMap::new();
        for record in self
            .state
            .lock()
            .ledger
            .iter()
            .filter(|r| r.at >= since && tenant_id.is_none_or(|t| r.tenant_id == t))
        {
            by_key
                .entry((
                    record.at.date_naive(),
                    record.tenant_id.clone(),
                    record.model.clone(),
                ))
                .or_default()
                .add(record);
        }
        let mut rows: Vec<_> = by_key
            .into_iter()
            .map(|((day, tenant_id, model), totals)| DailyUsage {
                day,
                tenant_id,
                model,
                totals,
            })
            .collect();
        rows.sort_by(|a, b| (a.day, &a.tenant_id, &a.model).cmp(&(b.day, &b.tenant_id, &b.model)));
        Ok(rows)
    }
}
