//! Postgres store.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use sqlx::postgres::{PgPool, PgPoolOptions};
use sqlx::{QueryBuilder, Row};
use uuid::Uuid;

use crate::keys::KeyHash;
use crate::model::{
    ApiKeyInfo, DailyUsage, JobCounts, JobOutcome, JobRecord, JobStatus, Limits, Tenant,
    UsageRecord, UsageTotals,
};
use crate::store::Store;

const SCHEMA: &str = include_str!("../migrations/0001_init.sql");
/// Advisory lock key held while the schema is applied ("LEVSCHEM").
const SCHEMA_LOCK: i64 = 0x4C45_5653_4348_454D;

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

fn job_from_row(row: &sqlx::postgres::PgRow) -> anyhow::Result<JobRecord> {
    let status: String = row.try_get("status")?;
    let response: Option<String> = row.try_get("response")?;
    Ok(JobRecord {
        id: row.try_get("id")?,
        tenant_id: row.try_get("tenant_id")?,
        custom_id: row.try_get("custom_id")?,
        endpoint: row.try_get("endpoint")?,
        status: status.parse::<JobStatus>().map_err(anyhow::Error::msg)?,
        attempts: row.try_get::<i32, _>("attempts")?.max(0) as u32,
        request_id: row.try_get("request_id")?,
        response_status: row
            .try_get::<Option<i32>, _>("response_status")?
            .map(|status| status.clamp(0, i32::from(u16::MAX)) as u16),
        response: response.and_then(|text| serde_json::from_str(&text).ok()),
        error: row.try_get("error")?,
        created_at: row.try_get("created_at")?,
        started_at: row.try_get("started_at")?,
        finished_at: row.try_get("finished_at")?,
    })
}

/// Rows per multi-row INSERT (Postgres allows 65535 bind parameters).
const JOB_INSERT_CHUNK: usize = 4096;

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
        // Concurrent `CREATE ... IF NOT EXISTS` from gateways starting at
        // the same time can still collide in the catalog: serialize them.
        let mut tx = pool.begin().await?;
        sqlx::query("SELECT pg_advisory_xact_lock($1)")
            .bind(SCHEMA_LOCK)
            .execute(&mut *tx)
            .await?;
        sqlx::raw_sql(SCHEMA).execute(&mut *tx).await?;
        tx.commit().await?;
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

    async fn insert_jobs(&self, jobs: &[JobRecord]) -> anyhow::Result<()> {
        for chunk in jobs.chunks(JOB_INSERT_CHUNK) {
            let mut builder = QueryBuilder::new(
                "INSERT INTO jobs (id, tenant_id, custom_id, endpoint, status, created_at) ",
            );
            builder.push_values(chunk, |mut row, job| {
                row.push_bind(job.id)
                    .push_bind(&job.tenant_id)
                    .push_bind(&job.custom_id)
                    .push_bind(&job.endpoint)
                    .push_bind(job.status.as_str())
                    .push_bind(job.created_at);
            });
            builder.push(" ON CONFLICT (id) DO NOTHING");
            builder.build().execute(&self.pool).await?;
        }
        Ok(())
    }

    async fn claim_job(&self, id: Uuid) -> anyhow::Result<Option<JobRecord>> {
        let row = sqlx::query(
            "UPDATE jobs SET status = 'running', attempts = attempts + 1, started_at = now() \
             WHERE id = $1 AND status IN ('queued', 'running') RETURNING *",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;
        row.as_ref().map(job_from_row).transpose()
    }

    async fn finish_job(&self, id: Uuid, outcome: &JobOutcome) -> anyhow::Result<()> {
        let response = outcome
            .response
            .as_ref()
            .map(serde_json::to_string)
            .transpose()?;
        sqlx::query(
            "UPDATE jobs SET status = $2, request_id = $3, response_status = $4, response = $5, \
             error = $6, finished_at = now() WHERE id = $1",
        )
        .bind(id)
        .bind(outcome.status.as_str())
        .bind(&outcome.request_id)
        .bind(outcome.response_status.map(i32::from))
        .bind(response)
        .bind(&outcome.error)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn get_job(&self, id: Uuid) -> anyhow::Result<Option<JobRecord>> {
        let row = sqlx::query("SELECT * FROM jobs WHERE id = $1")
            .bind(id)
            .fetch_optional(&self.pool)
            .await?;
        row.as_ref().map(job_from_row).transpose()
    }

    async fn job_counts(&self) -> anyhow::Result<JobCounts> {
        let rows = sqlx::query("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status")
            .fetch_all(&self.pool)
            .await?;
        let mut counts = JobCounts::default();
        for row in &rows {
            let status: String = row.try_get("status")?;
            let n = row.try_get::<i64, _>("n")?.max(0) as u64;
            match status.parse::<JobStatus>() {
                Ok(JobStatus::Queued) => counts.queued = n,
                Ok(JobStatus::Running) => counts.running = n,
                Ok(JobStatus::Succeeded) => counts.succeeded = n,
                Ok(JobStatus::Failed) => counts.failed = n,
                Err(_) => {}
            }
        }
        Ok(counts)
    }
}
