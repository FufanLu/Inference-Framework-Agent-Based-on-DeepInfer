//! Shared gateway state.

use std::collections::HashMap;
use std::ops::Deref;
use std::sync::Arc;
use std::time::Instant;

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
    /// API key -> tenant.
    pub api_keys: HashMap<String, String>,
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
    /// Build the state and start supervising the configured workers.
    pub fn new(config: Config, metrics: Option<PrometheusHandle>) -> anyhow::Result<Self> {
        let models = Arc::new(ModelRegistry::from_configs(&config.models)?);
        let router = Arc::new(Router::new(config.router.to_router_config()?));
        let events = EventBus::new(4096);
        let pool = WorkerPool::new(Arc::clone(&router), Arc::clone(&models), events.clone());
        for worker in &config.workers {
            pool.add_endpoint(&worker.endpoint)?;
        }
        let api_keys = config
            .auth
            .api_keys
            .iter()
            .map(|key| (key.key.clone(), key.tenant.clone()))
            .collect();
        let fingerprint = format!("fp_lev_{}", &uuid::Uuid::new_v4().simple().to_string()[..8]);
        Ok(Self(Arc::new(StateInner {
            audit: AuditLog::new(config.audit.capacity),
            config,
            models,
            router,
            pool,
            events,
            metrics,
            api_keys,
            started_at: Instant::now(),
            fingerprint,
        })))
    }
}
