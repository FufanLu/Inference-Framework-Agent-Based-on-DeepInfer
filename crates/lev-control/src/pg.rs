//! Postgres store.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use sqlx::postgres::{PgPool, PgPoolOptions};
use sqlx::{QueryBuilder, Row};
use uuid::Uuid;

use crate::keys::KeyHash;
use crate::model::{ApiKeyInfo, DailyUsage, Limits, Tenant, UsageRecord, UsageTotals};
use crate::store::Store;

const SCHEMA: &str = include_str!("../migrations/0001_init.sql");

#[derive(Debug, Clone)]
pub struct PgStore {
    pool: PgPool,
}

fn to_i64(value: Option<u64>) -> Option<i64> {
    value.map(|v| v.min(i64::MAX as u64) as i64)
}

fn to_u64(value: Option<i64>) -> Option<u64> {
    value.map(|v| v.max(0) as u64)
}

fn tenant_from_row(row: &sqlx::postgres::PgRow, prefix: &str) -> Result<Tenant, sqlx::Error> {
    let column = |name: &str| format!("{prefix}{name}");
    Ok(Tenant {
        id: row.try_get(column("id").as_str())?,
        name: row.try_get(column("name").as_str())?,
        limits: Limits {
            rpm: to_u64(row.try_get(column("rpm").as_str())?),
            tpm: to_u64(row.try_get(column("tpm").as_str())?),
            monthly_token_quota: to_u64(row.try_get(column("monthly_token_quota").as_str())?),
        },
        disabled: row.try_get(column("disabled").as_str())?,
    })
}

fn key_from_row(row: &sqlx::postgres::PgRow) -> Result<ApiKeyInfo, sqlx::Error> {
    Ok(ApiKeyInfo {
        id: row.try_get("key_id")?,
        tenant_id: row.try_get("key_tenant_id")?,
        name: row.try_get("key_name")?,
        prefix: row.try_get("key_prefix")?,
        created_at: row.try_get("key_created_at")?,
        revoked_at: row.try_get("key_revoked_at")?,
    })
}

/// Key columns, aliased for [`key_from_row`]. A macro so that queries stay
/// `&'static str` (sqlx refuses dynamically built SQL).
macro_rules! key_columns {
    () => {
        "k.id AS key_id, k.tenant_id AS key_tenant_id, k.name AS key_name, \
         k.prefix AS key_prefix, k.created_at AS key_created_at, k.revoked_at AS key_revoked_at"
    };
}

impl PgStore {
    /// Connect and apply the (idempotent) schema.
    pub async fn connect(url: &str) -> anyhow::Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(16)
            .connect(url)
            .await?;
        sqlx::raw_sql(SCHEMA).execute(&pool).await?;
        Ok(Self { pool })
    }
}

#[async_trait]
impl Store for PgStore {
    async fn lookup_key(&self, hash: &KeyHash) -> anyhow::Result<Option<(ApiKeyInfo, Tenant)>> {
        let row = sqlx::query(concat!(
            "SELECT ",
            key_columns!(),
            ", t.id AS t_id, t.name AS t_name, t.rpm AS t_rpm, t.tpm AS t_tpm, \
             t.monthly_token_quota AS t_monthly_token_quota, t.disabled AS t_disabled \
             FROM api_keys k JOIN tenants t ON t.id = k.tenant_id \
             WHERE k.sha256 = $1 AND k.revoked_at IS NULL"
        ))
        .bind(hash.as_slice())
        .fetch_optional(&self.pool)
        .await?;
        row.map(|row| Ok((key_from_row(&row)?, tenant_from_row(&row, "t_")?)))
            .transpose()
    }

    async fn upsert_tenant(&self, tenant: &Tenant) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT INTO tenants (id, name, rpm, tpm, monthly_token_quota, disabled) \
             VALUES ($1, $2, $3, $4, $5, $6) \
             ON CONFLICT (id) DO UPDATE SET name = $2, rpm = $3, tpm = $4, \
             monthly_token_quota = $5, disabled = $6",
        )
        .bind(&tenant.id)
        .bind(&tenant.name)
        .bind(to_i64(tenant.limits.rpm))
        .bind(to_i64(tenant.limits.tpm))
        .bind(to_i64(tenant.limits.monthly_token_quota))
        .bind(tenant.disabled)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn get_tenant(&self, id: &str) -> anyhow::Result<Option<Tenant>> {
        let row = sqlx::query("SELECT * FROM tenants WHERE id = $1")
            .bind(id)
            .fetch_optional(&self.pool)
            .await?;
        Ok(row.map(|row| tenant_from_row(&row, "")).transpose()?)
    }

    async fn list_tenants(&self) -> anyhow::Result<Vec<Tenant>> {
        let rows = sqlx::query("SELECT * FROM tenants ORDER BY id")
            .fetch_all(&self.pool)
            .await?;
        Ok(rows
            .iter()
            .map(|row| tenant_from_row(row, ""))
            .collect::<Result<_, _>>()?)
    }

    async fn insert_key(
        &self,
        tenant_id: &str,
        name: &str,
        prefix: &str,
        hash: &KeyHash,
    ) -> anyhow::Result<ApiKeyInfo> {
        sqlx::query(
            "INSERT INTO api_keys (id, tenant_id, name, prefix, sha256) VALUES ($1, $2, $3, $4, $5) \
             ON CONFLICT (sha256) DO NOTHING",
        )
        .bind(Uuid::new_v4())
        .bind(tenant_id)
        .bind(name)
        .bind(prefix)
        .bind(hash.as_slice())
        .execute(&self.pool)
        .await?;
        let row = sqlx::query(concat!(
            "SELECT ",
            key_columns!(),
            " FROM api_keys k WHERE k.sha256 = $1"
        ))
        .bind(hash.as_slice())
        .fetch_one(&self.pool)
        .await?;
        Ok(key_from_row(&row)?)
    }

    async fn list_keys(&self, tenant_id: Option<&str>) -> anyhow::Result<Vec<ApiKeyInfo>> {
        let rows = sqlx::query(concat!(
            "SELECT ",
            key_columns!(),
            " FROM api_keys k WHERE $1::TEXT IS NULL OR k.tenant_id = $1 ORDER BY k.created_at"
        ))
        .bind(tenant_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows.iter().map(key_from_row).collect::<Result<_, _>>()?)
    }

    async fn revoke_key(&self, id: Uuid) -> anyhow::Result<bool> {
        let result = sqlx::query(
            "UPDATE api_keys SET revoked_at = now() WHERE id = $1 AND revoked_at IS NULL",
        )
        .bind(id)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected() > 0)
    }

    async fn record_usage(&self, records: &[UsageRecord]) -> anyhow::Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        let mut builder = QueryBuilder::new(
            "INSERT INTO usage_ledger (tenant_id, request_id, model, prompt_tokens, \
             cached_tokens, completion_tokens, created_at) ",
        );
        builder.push_values(records, |mut row, record| {
            row.push_bind(&record.tenant_id)
                .push_bind(&record.request_id)
                .push_bind(&record.model)
                .push_bind(record.prompt_tokens as i64)
                .push_bind(record.cached_tokens as i64)
                .push_bind(record.completion_tokens as i64)
                .push_bind(record.at);
        });
        builder.build().execute(&self.pool).await?;
        Ok(())
    }

    async fn usage_since(
        &self,
        tenant_id: &str,
        since: DateTime<Utc>,
    ) -> anyhow::Result<UsageTotals> {
        let row = sqlx::query(
            "SELECT COUNT(*) AS requests, COALESCE(SUM(prompt_tokens), 0)::BIGINT AS prompt, \
             COALESCE(SUM(cached_tokens), 0)::BIGINT AS cached, \
             COALESCE(SUM(completion_tokens), 0)::BIGINT AS completion \
             FROM usage_ledger WHERE tenant_id = $1 AND created_at >= $2",
        )
        .bind(tenant_id)
        .bind(since)
        .fetch_one(&self.pool)
        .await?;
        Ok(UsageTotals {
            requests: row.try_get::<i64, _>("requests")? as u64,
            prompt_tokens: row.try_get::<i64, _>("prompt")? as u64,
            cached_tokens: row.try_get::<i64, _>("cached")? as u64,
            completion_tokens: row.try_get::<i64, _>("completion")? as u64,
        })
    }

    async fn daily_usage(
        &self,
        tenant_id: Option<&str>,
        since: DateTime<Utc>,
    ) -> anyhow::Result<Vec<DailyUsage>> {
        let rows = sqlx::query(
            "SELECT (created_at AT TIME ZONE 'UTC')::DATE AS day, tenant_id, model, COUNT(*) AS requests, \
             SUM(prompt_tokens)::BIGINT AS prompt, SUM(cached_tokens)::BIGINT AS cached, \
             SUM(completion_tokens)::BIGINT AS completion FROM usage_ledger \
             WHERE created_at >= $1 AND ($2::TEXT IS NULL OR tenant_id = $2) \
             GROUP BY 1, 2, 3 ORDER BY 1, 2, 3",
        )
        .bind(since)
        .bind(tenant_id)
        .fetch_all(&self.pool)
        .await?;
        rows.iter()
            .map(|row| {
                Ok(DailyUsage {
                    day: row.try_get("day")?,
                    tenant_id: row.try_get("tenant_id")?,
                    model: row.try_get("model")?,
                    totals: UsageTotals {
                        requests: row.try_get::<i64, _>("requests")? as u64,
                        prompt_tokens: row.try_get::<i64, _>("prompt")? as u64,
                        cached_tokens: row.try_get::<i64, _>("cached")? as u64,
                        completion_tokens: row.try_get::<i64, _>("completion")? as u64,
                    },
                })
            })
            .collect()
    }
}
