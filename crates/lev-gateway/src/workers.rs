//! Worker connections: one supervisor task per endpoint keeps the router's
//! view of that worker current (metadata, health, KV events, load).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use lev_proto::pb;
use lev_proto::pb::worker_client::WorkerClient;
use lev_router::{KvChange, Medium, Role, Router, WorkerLoad, WorkerMeta};
use parking_lot::RwLock;
use tonic::transport::{Channel, Endpoint};

use crate::events::{EventBus, LiveEvent, now_ms};
use crate::model::ModelRegistry;

/// A connected worker.
#[derive(Debug, Clone)]
pub struct WorkerEntry {
    pub id: String,
    pub endpoint: String,
    pub info: pb::WorkerInfo,
    pub client: WorkerClient<Channel>,
}

#[derive(Debug)]
struct PoolInner {
    router: Arc<Router>,
    models: Arc<ModelRegistry>,
    events: EventBus,
    workers: RwLock<HashMap<String, WorkerEntry>>,
    endpoints: RwLock<Vec<String>>,
}

/// All workers the gateway knows about.
#[derive(Debug, Clone)]
pub struct WorkerPool {
    inner: Arc<PoolInner>,
}

impl WorkerPool {
    pub fn new(router: Arc<Router>, models: Arc<ModelRegistry>, events: EventBus) -> Self {
        Self {
            inner: Arc::new(PoolInner {
                router,
                models,
                events,
                workers: RwLock::new(HashMap::new()),
                endpoints: RwLock::new(Vec::new()),
            }),
        }
    }

    /// Start supervising a worker endpoint. Idempotent.
    pub fn add_endpoint(&self, endpoint: &str) -> anyhow::Result<()> {
        let endpoint = endpoint.trim_end_matches('/').to_string();
        {
            let mut endpoints = self.inner.endpoints.write();
            if endpoints.contains(&endpoint) {
                return Ok(());
            }
            endpoints.push(endpoint.clone());
        }
        let channel = Endpoint::from_shared(endpoint.clone())?
            .connect_timeout(Duration::from_secs(2))
            .tcp_nodelay(true)
            .http2_keep_alive_interval(Duration::from_secs(10))
            .keep_alive_timeout(Duration::from_secs(5))
            .keep_alive_while_idle(true)
            .connect_lazy();
        let pool = self.clone();
        tokio::spawn(async move { pool.supervise(endpoint, WorkerClient::new(channel)).await });
        Ok(())
    }

    pub fn endpoints(&self) -> Vec<String> {
        self.inner.endpoints.read().clone()
    }

    pub fn get(&self, worker_id: &str) -> Option<WorkerEntry> {
        self.inner.workers.read().get(worker_id).cloned()
    }

    pub fn all(&self) -> Vec<WorkerEntry> {
        let mut workers: Vec<_> = self.inner.workers.read().values().cloned().collect();
        workers.sort_by(|a, b| a.id.cmp(&b.id));
        workers
    }

    /// A request to this worker failed at the transport level: stop routing
    /// to it until its supervisor reconnects.
    pub fn report_failure(&self, worker_id: &str, detail: &str) {
        self.inner.router.set_healthy(worker_id, false);
        if let Some(entry) = self.get(worker_id) {
            self.publish_status(&entry.id, &entry.endpoint, "unhealthy", detail);
        }
    }

    fn publish_status(&self, worker_id: &str, endpoint: &str, status: &'static str, detail: &str) {
        self.inner.events.publish(LiveEvent::Worker {
            worker_id: worker_id.to_string(),
            endpoint: endpoint.to_string(),
            status,
            detail: detail.to_string(),
            at_ms: now_ms(),
        });
    }

    async fn supervise(self, endpoint: String, mut client: WorkerClient<Channel>) {
        let mut backoff = Duration::from_millis(250);
        let mut current_id: Option<String> = None;
        loop {
            match self
                .connect_once(&endpoint, &mut client, &mut current_id)
                .await
            {
                Ok(()) => backoff = Duration::from_millis(250),
                Err(error) => {
                    tracing::debug!(%endpoint, %error, "worker connection attempt failed");
                }
            }
            if let Some(id) = &current_id {
                self.inner.router.set_healthy(id, false);
            }
            tokio::time::sleep(backoff).await;
            backoff = (backoff * 2).min(Duration::from_secs(5));
        }
    }

    /// Register the worker and follow its event stream until it ends.
    async fn connect_once(
        &self,
        endpoint: &str,
        client: &mut WorkerClient<Channel>,
        current_id: &mut Option<String>,
    ) -> anyhow::Result<()> {
        let info = client.get_info(pb::GetInfoRequest {}).await?.into_inner();
        let id = if info.worker_id.is_empty() {
            endpoint.to_string()
        } else {
            info.worker_id.clone()
        };
        if let Some(previous) = current_id.as_ref()
            && *previous != id
        {
            // The endpoint now serves a different worker.
            self.inner.router.remove_worker(previous);
            self.inner.workers.write().remove(previous);
        }
        *current_id = Some(id.clone());

        let model = self
            .inner
            .models
            .canonical(&info.model)
            .map(str::to_string)
            .unwrap_or_else(|| {
                tracing::warn!(%endpoint, model = %info.model, "worker serves a model the gateway does not");
                info.model.clone()
            });
        let role = match pb::Role::try_from(info.role).unwrap_or(pb::Role::Unified) {
            pb::Role::Prefill => Role::Prefill,
            pb::Role::Decode => Role::Decode,
            pb::Role::Unified | pb::Role::Unspecified => Role::Unified,
        };
        self.inner.router.upsert_worker(WorkerMeta {
            id: id.clone(),
            endpoint: endpoint.to_string(),
            model,
            role,
            block_size: info.block_size.max(1) as usize,
            total_blocks: info.total_blocks,
            kv_events: info.supports_kv_events,
        })?;
        self.inner.workers.write().insert(
            id.clone(),
            WorkerEntry {
                id: id.clone(),
                endpoint: endpoint.to_string(),
                info: info.clone(),
                client: client.clone(),
            },
        );

        let mut stream = client
            .subscribe_events(pb::SubscribeEventsRequest {
                snapshot: true,
                load_report_interval_ms: 250,
            })
            .await?
            .into_inner();
        self.inner.router.set_healthy(&id, true);
        self.publish_status(&id, endpoint, "healthy", &info.backend);
        tracing::info!(worker = %id, %endpoint, backend = %info.backend, role = ?role, "worker connected");

        let mut next_seq: Option<u64> = None;
        loop {
            let Some(event) = stream.message().await? else {
                anyhow::bail!("event stream closed");
            };
            match event.event {
                Some(pb::worker_event::Event::Kv(batch)) => {
                    if let Some(expected) = next_seq
                        && batch.seq != expected
                    {
                        // Lost events: resubscribe for a fresh snapshot.
                        anyhow::bail!("kv event gap: expected seq {expected}, got {}", batch.seq);
                    }
                    next_seq = Some(batch.seq + 1);
                    for kv in batch.events {
                        if let Some(change) = kv_change(kv) {
                            self.inner.router.apply_kv(&id, change);
                        }
                    }
                }
                Some(pb::worker_event::Event::Load(report)) => {
                    self.inner.router.update_load(
                        &id,
                        WorkerLoad {
                            running: report.running,
                            waiting: report.waiting,
                            kv_usage: report.kv_usage,
                            free_blocks: report.free_blocks,
                            prefill_tokens_per_s: report.prefill_tokens_per_s,
                            decode_tokens_per_s: report.decode_tokens_per_s,
                            spec_accept_rate: report.spec_accept_rate,
                        },
                    );
                }
                None => {}
            }
        }
    }
}

fn medium(value: i32) -> Medium {
    match pb::Medium::try_from(value).unwrap_or(pb::Medium::Gpu) {
        pb::Medium::Cpu => Medium::Cpu,
        pb::Medium::Gpu | pb::Medium::Unspecified => Medium::Gpu,
    }
}

fn kv_change(event: pb::KvEvent) -> Option<KvChange> {
    match event.event? {
        pb::kv_event::Event::Stored(stored) => Some(KvChange::Stored {
            hashes: stored.block_hashes,
            medium: medium(stored.medium),
        }),
        pb::kv_event::Event::Removed(removed) => Some(KvChange::Removed {
            hashes: removed.block_hashes,
            medium: medium(removed.medium),
        }),
        pb::kv_event::Event::Cleared(_) => Some(KvChange::Cleared),
    }
}
