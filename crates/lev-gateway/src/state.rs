//! Shared gateway state.

use std::ops::Deref;
use std::sync::Arc;
use std::time::Instant;

use lev_control::{ControlPlane, Limiter, LocalLimiter, MemoryStore, Store, Tenant};
use lev_router::Router;
use metrics_exporter_prometheus::PrometheusHandle;

use crate::audit::AuditLog;
use crate::config::Config;
use crate::events::EventBus;
use crate::model::ModelRegistry;
use crate::workers::WorkerPool;

#[derive(Debug)]
pub struct StateInner {
    pub config: Config,
    pub models: Arc<ModelRegistry>,
    pub router: Arc<Router>,
    pub pool: WorkerPool,
    pub audit: AuditLog,
    pub events: EventBus,
    pub metrics: Option<PrometheusHandle>,
    /// Tenants, keys, limits and the usage ledger.
    pub control: Arc<ControlPlane>,
    pub started_at: Instant,
    /// Distinguishes this gateway process in `system_fingerprint`.
    pub fingerprint: String,
}

#[derive(Debug, Clone)]
pub struct AppState(Arc<StateInner>);

impl Deref for AppState {
    type Target = StateInner;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AppState {
    /// Build the state, connect the control plane and start supervising the
    /// configured workers.
    pub async fn new(config: Config, metrics: Option<PrometheusHandle>) -> anyhow::Result<Self> {
        let control = build_control(&config).await?;
        let models = Arc::new(ModelRegistry::from_configs(&config.models)?);
        let router = Arc::new(Router::new(config.router.to_router_config()?));
        let events = EventBus::new(4096);
        let pool = WorkerPool::new(Arc::clone(&router), Arc::clone(&models), events.clone());
        for worker in &config.workers {
            pool.add_endpoint(&worker.endpoint)?;
        }
        let fingerprint = format!("fp_lev_{}", &uuid::Uuid::new_v4().simple().to_string()[..8]);
        Ok(Self(Arc::new(StateInner {
            audit: AuditLog::new(config.audit.capacity),
            config,
            models,
            router,
            pool,
            events,
            metrics,
            control,
            started_at: Instant::now(),
            fingerprint,
        })))
    }
}

async fn build_control(config: &Config) -> anyhow::Result<Arc<ControlPlane>> {
    let store: Arc<dyn Store> = match config.control.store.as_str() {
        "memory" | "" => Arc::new(MemoryStore::new()),
        url if url.starts_with("postgres") => {
            Arc::new(lev_control::pg::PgStore::connect(url).await?)
        }
        other => {
            anyhow::bail!("unknown control store `{other}`: use `memory` or a postgres:// URL")
        }
    };
    let limiter: Arc<dyn Limiter> = match &config.control.redis {
        Some(url) => Arc::new(lev_control::limiter::RedisLimiter::connect(url).await?),
        None => Arc::new(LocalLimiter::new()),
    };
    let allow_anonymous = config
        .auth
        .allow_anonymous
        .unwrap_or(config.auth.api_keys.is_empty());
    let anonymous = allow_anonymous.then(|| Tenant {
        id: "anonymous".to_string(),
        name: "Anonymous".to_string(),
        limits: config.auth.anonymous_limits.into(),
        disabled: false,
    });
    let control = ControlPlane::new(store, limiter, anonymous);
    for key in &config.auth.api_keys {
        let tenant = Tenant {
            id: key.tenant.clone(),
            name: key.tenant.clone(),
            limits: key.limits.into(),
            disabled: false,
        };
        control
            .ensure_key(
                &tenant,
                &key.key,
                key.name.as_deref().unwrap_or("configured"),
            )
            .await?;
    }
    tracing::info!(
        store = %config.control.store,
        shared_limits = config.control.redis.is_some(),
        allow_anonymous,
        static_keys = config.auth.api_keys.len(),
        "control plane ready"
    );
    Ok(control)
}
