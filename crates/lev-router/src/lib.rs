//! Leviathan request router.
//!
//! DeepInfer's open-source release leaves load balancing across data-parallel
//! replicas and prefill/decode disaggregation to the serving layer. This crate
//! is that layer's decision core. It is pure logic (no I/O, no async runtime):
//! the gateway feeds it worker metadata, KV cache events and load reports, and
//! asks it where to send each request.
//!
//! - [`index`] tracks which prompt prefixes each worker caches, exactly from
//!   KV events or approximately from the router's own routing history.
//! - [`Policy`] chooses among workers. [`Policy::CacheAware`] treats routing
//!   as minimising the estimated time to first token: the prefill work a
//!   worker still has to do for this prompt (prompt minus cached prefix) plus
//!   the work already queued on it.
//! - With prefill and decode pools available, a request gets a prefill worker
//!   (chosen cache-aware) and a decode worker (chosen by load).
//! - A [`Ticket`] tracks the in-flight work of every routed request, so that
//!   load is visible to the router before the next worker report arrives.

pub mod hashing;
pub mod index;

use std::collections::HashMap;
use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

pub use index::{MAX_SLOTS, Medium, PrefixMatch, ShadowConfig};
use index::{PrefixIndex, SlotMask};

/// The part a worker plays in prefill/decode disaggregation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    Unified,
    Prefill,
    Decode,
}

/// How the router chooses among eligible workers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Policy {
    RoundRobin,
    LeastLoad,
    /// Power of two choices: the less loaded of two random workers.
    #[serde(rename = "p2c", alias = "power_of_two")]
    PowerOfTwo,
    /// Minimise the estimated time to first token.
    CacheAware,
}

impl Policy {
    pub const ALL: [Policy; 4] = [
        Policy::RoundRobin,
        Policy::LeastLoad,
        Policy::PowerOfTwo,
        Policy::CacheAware,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Policy::RoundRobin => "round_robin",
            Policy::LeastLoad => "least_load",
            Policy::PowerOfTwo => "p2c",
            Policy::CacheAware => "cache_aware",
        }
    }
}

impl FromStr for Policy {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "round_robin" | "rr" => Ok(Policy::RoundRobin),
            "least_load" => Ok(Policy::LeastLoad),
            "p2c" | "power_of_two" => Ok(Policy::PowerOfTwo),
            "cache_aware" => Ok(Policy::CacheAware),
            other => Err(format!(
                "unknown policy `{other}`: expected round_robin, least_load, p2c or cache_aware"
            )),
        }
    }
}

/// Where prefix information comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexMode {
    /// KV events where the worker publishes them, the shadow index otherwise.
    Exact,
    /// The shadow index everywhere; KV events are ignored.
    Approximate,
    /// No prefix tracking.
    Off,
}

impl FromStr for IndexMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "exact" => Ok(IndexMode::Exact),
            "approximate" | "approx" => Ok(IndexMode::Approximate),
            "off" => Ok(IndexMode::Off),
            other => Err(format!(
                "unknown index mode `{other}`: expected exact, approximate or off"
            )),
        }
    }
}

/// Router tuning.
#[derive(Debug, Clone)]
pub struct RouterConfig {
    pub policy: Policy,
    pub index_mode: IndexMode,
    /// Split requests across prefill and decode pools when both exist.
    pub disaggregation: bool,
    /// Prefer a stable worker per session key when it costs little.
    pub session_affinity: bool,
    pub shadow: ShadowConfig,
    /// Prefill cost of one uncached prompt token on a worker (its capacity,
    /// calibrated per model and hardware). Workers' observed prefill
    /// throughput is deliberately not used: a replica serving mostly cache
    /// hits computes few prefill tokens and would look slow.
    pub prefill_us_per_token: f64,
    /// Estimated TTFT added by each sequence already running on a worker.
    pub decode_interference_us: f64,
    /// A CPU-tier hit costs this fraction of recomputing the block.
    pub cpu_hit_cost_ratio: f64,
    /// Workers above this KV usage are penalised.
    pub max_kv_usage: f32,
    /// The affinity worker is kept while its cost is within this fraction of
    /// the best worker's.
    pub affinity_slack: f64,
    /// Seed of the router's random choices (power of two).
    pub seed: u64,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            policy: Policy::CacheAware,
            index_mode: IndexMode::Exact,
            disaggregation: true,
            session_affinity: false,
            shadow: ShadowConfig::default(),
            prefill_us_per_token: 30.0,
            decode_interference_us: 2_000.0,
            cpu_hit_cost_ratio: 0.2,
            max_kv_usage: 0.95,
            affinity_slack: 0.25,
            seed: 0x5EED,
        }
    }
}

/// What the router knows about a worker from its `GetInfo`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerMeta {
    pub id: String,
    pub endpoint: String,
    /// Canonical served-model name.
    pub model: String,
    pub role: Role,
    pub block_size: usize,
    pub total_blocks: u64,
    /// The worker publishes KV cache events.
    pub kv_events: bool,
}

/// The latest load report of a worker.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct WorkerLoad {
    pub running: u32,
    pub waiting: u32,
    pub kv_usage: f32,
    pub free_blocks: u64,
    pub prefill_tokens_per_s: f32,
    pub decode_tokens_per_s: f32,
    pub spec_accept_rate: f32,
}

/// A change to a worker's KV cache, from its event stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvChange {
    Stored { hashes: Vec<u64>, medium: Medium },
    Removed { hashes: Vec<u64>, medium: Medium },
    Cleared,
}

#[derive(Debug)]
struct WorkerState {
    meta: WorkerMeta,
    generation: u64,
    healthy: bool,
    load: WorkerLoad,
    load_at: Option<Instant>,
    inflight: u32,
    pending_prefill_tokens: u64,
    dispatched: u64,
}

impl WorkerState {
    /// Sequences on the worker: its own report, or the router's count of
    /// requests it sent there when that is higher (fresher).
    fn sequences(&self) -> u32 {
        (self.load.running + self.load.waiting).max(self.inflight)
    }

    fn index_source(&self, mode: IndexMode) -> IndexSource {
        match mode {
            IndexMode::Off => IndexSource::Off,
            IndexMode::Exact if self.meta.kv_events => IndexSource::Events,
            IndexMode::Exact | IndexMode::Approximate => IndexSource::Shadow,
        }
    }
}

/// Where the router's prefix view of a worker comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexSource {
    Events,
    Shadow,
    Off,
}

#[derive(Debug)]
struct Inner {
    config: RouterConfig,
    slots: Vec<Option<WorkerState>>,
    by_id: HashMap<String, usize>,
    index: PrefixIndex,
    next_generation: u64,
    rng: u64,
}

impl Inner {
    fn worker(&self, slot: usize) -> &WorkerState {
        self.slots[slot].as_ref().expect("slot is occupied")
    }

    fn worker_mut(&mut self, slot: usize) -> &mut WorkerState {
        self.slots[slot].as_mut().expect("slot is occupied")
    }

    fn next_random(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

/// One request to route.
#[derive(Debug, Clone, Copy)]
pub struct RouteRequest<'a> {
    pub model: &'a str,
    pub prompt: &'a [u32],
    pub session_key: Option<&'a str>,
    /// Workers to skip, e.g. after a failed attempt.
    pub exclude: &'a [String],
}

/// A worker chosen for one stage of a request.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Target {
    pub worker_id: String,
    pub endpoint: String,
    /// Prompt tokens the router expects the worker to have cached.
    pub cached_tokens: usize,
}

/// Which workers run a request.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Plan {
    Unified { worker: Target },
    Disaggregated { prefill: Target, decode: Target },
}

impl Plan {
    /// The worker that runs prefill.
    pub fn prefill_target(&self) -> &Target {
        match self {
            Plan::Unified { worker } => worker,
            Plan::Disaggregated { prefill, .. } => prefill,
        }
    }
}

/// How one candidate scored, for observability.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CandidateScore {
    pub worker_id: String,
    pub role: Role,
    pub cached_tokens: usize,
    pub sequences: u32,
    pub kv_usage: f32,
    /// Estimated time to first token in microseconds (cache-aware cost).
    pub est_ttft_us: f64,
    pub chosen: bool,
}

/// The router's answer for one request.
#[derive(Debug)]
pub struct RouteDecision {
    pub plan: Plan,
    pub policy: Policy,
    pub prompt_tokens: usize,
    /// Scores of the candidates for the prefill (or unified) stage.
    pub candidates: Vec<CandidateScore>,
    /// The session affinity worker was chosen.
    pub affinity: bool,
    pub ticket: Ticket,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouteError {
    /// No healthy worker serves the model.
    NoWorkers { model: String },
}

impl std::fmt::Display for RouteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RouteError::NoWorkers { model } => {
                write!(f, "no healthy worker is serving model `{model}`")
            }
        }
    }
}

impl std::error::Error for RouteError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryError {
    TooManyWorkers,
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "the router tracks at most {MAX_SLOTS} workers")
    }
}

impl std::error::Error for RegistryError {}

#[derive(Debug)]
struct TicketEntry {
    slot: usize,
    generation: u64,
    prefill_tokens: u64,
    ready: bool,
}

/// In-flight accounting of one routed request. Dropping it releases the
/// request's load from its workers.
#[derive(Debug)]
pub struct Ticket {
    inner: Arc<Mutex<Inner>>,
    entries: Vec<TicketEntry>,
}

impl Ticket {
    /// Prefill finished: its queued prefill tokens are no longer pending.
    pub fn prefill_done(&mut self) {
        let mut inner = self.inner.lock();
        for entry in &mut self.entries {
            if entry.ready {
                continue;
            }
            entry.ready = true;
            if let Some(Some(worker)) = inner.slots.get_mut(entry.slot)
                && worker.generation == entry.generation
            {
                worker.pending_prefill_tokens = worker
                    .pending_prefill_tokens
                    .saturating_sub(entry.prefill_tokens);
            }
        }
    }
}

impl Drop for Ticket {
    fn drop(&mut self) {
        let mut inner = self.inner.lock();
        for entry in &self.entries {
            if let Some(Some(worker)) = inner.slots.get_mut(entry.slot)
                && worker.generation == entry.generation
            {
                worker.inflight = worker.inflight.saturating_sub(1);
                if !entry.ready {
                    worker.pending_prefill_tokens = worker
                        .pending_prefill_tokens
                        .saturating_sub(entry.prefill_tokens);
                }
            }
        }
    }
}

/// A worker as the router sees it.
#[derive(Debug, Clone, Serialize)]
pub struct WorkerView {
    pub id: String,
    pub endpoint: String,
    pub model: String,
    pub role: Role,
    pub healthy: bool,
    pub block_size: usize,
    pub total_blocks: u64,
    pub index_source: IndexSource,
    pub load: WorkerLoad,
    /// Milliseconds since the last load report.
    pub load_age_ms: Option<u64>,
    pub inflight: u32,
    pub pending_prefill_tokens: u64,
    pub dispatched: u64,
    pub gpu_blocks: usize,
    pub cpu_blocks: usize,
}

/// The router's state, for dashboards.
#[derive(Debug, Clone, Serialize)]
pub struct RouterSnapshot {
    pub policy: Policy,
    pub index_mode: IndexMode,
    pub disaggregation: bool,
    pub session_affinity: bool,
    pub indexed_blocks: usize,
    pub workers: Vec<WorkerView>,
}

/// The router. Cheap to share: clone the `Arc`.
#[derive(Debug)]
pub struct Router {
    inner: Arc<Mutex<Inner>>,
    round_robin: AtomicU64,
}

impl Router {
    pub fn new(config: RouterConfig) -> Self {
        let rng = config.seed.max(1);
        Self {
            inner: Arc::new(Mutex::new(Inner {
                config,
                slots: Vec::new(),
                by_id: HashMap::new(),
                index: PrefixIndex::new(),
                next_generation: 1,
                rng,
            })),
            round_robin: AtomicU64::new(0),
        }
    }

    /// Register a worker or update its metadata. A changed block size or role
    /// invalidates what the router knows about its cache.
    pub fn upsert_worker(&self, meta: WorkerMeta) -> Result<(), RegistryError> {
        let mut inner = self.inner.lock();
        if let Some(&slot) = inner.by_id.get(&meta.id) {
            let invalidate = {
                let worker = inner.worker(slot);
                worker.meta.block_size != meta.block_size || worker.meta.role != meta.role
            };
            if invalidate {
                inner.index.clear_slot(slot);
            }
            inner.worker_mut(slot).meta = meta;
            return Ok(());
        }
        let slot = match inner.slots.iter().position(Option::is_none) {
            Some(slot) => slot,
            None if inner.slots.len() < MAX_SLOTS => {
                inner.slots.push(None);
                inner.slots.len() - 1
            }
            None => return Err(RegistryError::TooManyWorkers),
        };
        let generation = inner.next_generation;
        inner.next_generation += 1;
        inner.by_id.insert(meta.id.clone(), slot);
        inner.slots[slot] = Some(WorkerState {
            meta,
            generation,
            healthy: false,
            load: WorkerLoad::default(),
            load_at: None,
            inflight: 0,
            pending_prefill_tokens: 0,
            dispatched: 0,
        });
        Ok(())
    }

    /// Forget a worker entirely.
    pub fn remove_worker(&self, id: &str) {
        let mut inner = self.inner.lock();
        if let Some(slot) = inner.by_id.remove(id) {
            inner.index.clear_slot(slot);
            inner.slots[slot] = None;
        }
    }

    /// Mark a worker healthy or not. Losing health drops the router's view of
    /// its cache: the worker may have restarted, and it resends a snapshot
    /// when its event stream reconnects.
    pub fn set_healthy(&self, id: &str, healthy: bool) {
        let mut inner = self.inner.lock();
        let Some(&slot) = inner.by_id.get(id) else {
            return;
        };
        if !healthy {
            inner.index.clear_slot(slot);
        }
        inner.worker_mut(slot).healthy = healthy;
    }

    pub fn update_load(&self, id: &str, load: WorkerLoad) {
        let mut inner = self.inner.lock();
        if let Some(&slot) = inner.by_id.get(id) {
            let worker = inner.worker_mut(slot);
            worker.load = load;
            worker.load_at = Some(Instant::now());
        }
    }

    /// Apply a KV cache change reported by the worker. Ignored unless the
    /// router uses events for this worker.
    pub fn apply_kv(&self, id: &str, change: KvChange) {
        let mut inner = self.inner.lock();
        let Some(&slot) = inner.by_id.get(id) else {
            return;
        };
        let mode = inner.config.index_mode;
        if inner.worker(slot).index_source(mode) != IndexSource::Events {
            return;
        }
        match change {
            KvChange::Stored { hashes, medium } => inner.index.store(slot, &hashes, medium),
            KvChange::Removed { hashes, medium } => inner.index.remove(slot, &hashes, medium),
            KvChange::Cleared => inner.index.clear_slot(slot),
        }
    }

    pub fn set_policy(&self, policy: Policy) {
        self.inner.lock().config.policy = policy;
    }

    pub fn set_index_mode(&self, mode: IndexMode) {
        let mut inner = self.inner.lock();
        if inner.config.index_mode != mode {
            inner.config.index_mode = mode;
            // The other source's view is not valid for the new mode.
            let slots: Vec<usize> = inner.by_id.values().copied().collect();
            for slot in slots {
                inner.index.clear_slot(slot);
            }
        }
    }

    pub fn set_disaggregation(&self, enabled: bool) {
        self.inner.lock().config.disaggregation = enabled;
    }

    pub fn config(&self) -> RouterConfig {
        self.inner.lock().config.clone()
    }

    /// Choose the workers for a request and account for its load.
    pub fn route(&self, request: RouteRequest<'_>) -> Result<RouteDecision, RouteError> {
        let mut guard = self.inner.lock();
        let inner = &mut *guard;
        let now = Instant::now();
        if inner.config.index_mode != IndexMode::Off {
            inner.index.shadow_expire(now);
        }

        let mut pools: HashMap<Role, Vec<usize>> = HashMap::new();
        for (slot, worker) in inner.slots.iter().enumerate() {
            let Some(worker) = worker else { continue };
            if worker.healthy
                && worker.meta.model == request.model
                && !request.exclude.contains(&worker.meta.id)
            {
                pools.entry(worker.meta.role).or_default().push(slot);
            }
        }
        let prefill_pool = pools.remove(&Role::Prefill).unwrap_or_default();
        let decode_pool = pools.remove(&Role::Decode).unwrap_or_default();
        let unified_pool = pools.remove(&Role::Unified).unwrap_or_default();

        let disaggregate =
            inner.config.disaggregation && !prefill_pool.is_empty() && !decode_pool.is_empty();
        let (primary_pool, decode_pool) = if disaggregate {
            (prefill_pool, Some(decode_pool))
        } else if !unified_pool.is_empty() {
            (unified_pool, None)
        } else {
            // Only one disaggregated pool is up: its workers run requests
            // end to end.
            let mut any: Vec<usize> = prefill_pool.into_iter().chain(decode_pool).collect();
            any.sort_unstable();
            (any, None)
        };
        if primary_pool.is_empty() {
            return Err(RouteError::NoWorkers {
                model: request.model.to_string(),
            });
        }

        let policy = inner.config.policy;
        let scored = score_candidates(inner, &primary_pool, &request);
        let (choice, affinity) = choose(inner, &self.round_robin, policy, &scored, &request);
        let chosen = &scored[choice];

        let mut entries = Vec::with_capacity(2);
        let primary_target = commit(inner, chosen, now, &mut entries);

        let plan = match decode_pool {
            Some(decode_pool) => {
                let decode_scored = score_candidates(inner, &decode_pool, &request);
                let decode_choice = least_loaded(&decode_scored);
                let decode = &decode_scored[decode_choice];
                let slot = decode.slot;
                let worker = inner.worker_mut(slot);
                worker.inflight += 1;
                worker.dispatched += 1;
                entries.push(TicketEntry {
                    slot,
                    generation: worker.generation,
                    prefill_tokens: 0,
                    ready: true,
                });
                Plan::Disaggregated {
                    prefill: primary_target,
                    decode: Target {
                        worker_id: worker.meta.id.clone(),
                        endpoint: worker.meta.endpoint.clone(),
                        cached_tokens: decode.cached_tokens,
                    },
                }
            }
            None => Plan::Unified {
                worker: primary_target,
            },
        };

        let candidates = scored
            .iter()
            .enumerate()
            .map(|(i, s)| CandidateScore {
                worker_id: inner.worker(s.slot).meta.id.clone(),
                role: inner.worker(s.slot).meta.role,
                cached_tokens: s.cached_tokens,
                sequences: s.sequences,
                kv_usage: s.kv_usage,
                est_ttft_us: s.cost,
                chosen: i == choice,
            })
            .collect();

        Ok(RouteDecision {
            plan,
            policy,
            prompt_tokens: request.prompt.len(),
            candidates,
            affinity,
            ticket: Ticket {
                inner: Arc::clone(&self.inner),
                entries,
            },
        })
    }

    /// The router's view of every worker.
    pub fn snapshot(&self) -> RouterSnapshot {
        let inner = self.inner.lock();
        let now = Instant::now();
        let mode = inner.config.index_mode;
        let mut workers: Vec<WorkerView> = inner
            .slots
            .iter()
            .enumerate()
            .filter_map(|(slot, worker)| worker.as_ref().map(|worker| (slot, worker)))
            .map(|(slot, worker)| WorkerView {
                id: worker.meta.id.clone(),
                endpoint: worker.meta.endpoint.clone(),
                model: worker.meta.model.clone(),
                role: worker.meta.role,
                healthy: worker.healthy,
                block_size: worker.meta.block_size,
                total_blocks: worker.meta.total_blocks,
                index_source: worker.index_source(mode),
                load: worker.load,
                load_age_ms: worker
                    .load_at
                    .map(|at| now.duration_since(at).as_millis() as u64),
                inflight: worker.inflight,
                pending_prefill_tokens: worker.pending_prefill_tokens,
                dispatched: worker.dispatched,
                gpu_blocks: inner.index.blocks(slot, Medium::Gpu),
                cpu_blocks: inner.index.blocks(slot, Medium::Cpu),
            })
            .collect();
        workers.sort_by(|a, b| a.id.cmp(&b.id));
        RouterSnapshot {
            policy: inner.config.policy,
            index_mode: mode,
            disaggregation: inner.config.disaggregation,
            session_affinity: inner.config.session_affinity,
            indexed_blocks: inner.index.len(),
            workers,
        }
    }
}

#[derive(Debug, Clone)]
struct Scored {
    slot: usize,
    hashes: Arc<Vec<u64>>,
    cached_tokens: usize,
    /// Prompt tokens this worker would still have to prefill.
    new_tokens: u64,
    sequences: u32,
    kv_usage: f32,
    cost: f64,
}

fn score_candidates(inner: &Inner, pool: &[usize], request: &RouteRequest<'_>) -> Vec<Scored> {
    let config = &inner.config;
    let prompt_len = request.prompt.len();
    let mut hashes_by_block_size: HashMap<usize, Arc<Vec<u64>>> = HashMap::new();
    let mut matches_by_block_size: HashMap<usize, HashMap<usize, PrefixMatch>> = HashMap::new();
    if config.index_mode != IndexMode::Off {
        let mut masks: HashMap<usize, SlotMask> = HashMap::new();
        for &slot in pool {
            *masks.entry(inner.worker(slot).meta.block_size).or_default() |= 1u128 << slot;
        }
        for (block_size, mask) in masks {
            let hashes = Arc::new(hashing::block_hashes(request.prompt, block_size));
            let matches = inner
                .index
                .match_prefix(&hashes, mask)
                .into_iter()
                .collect();
            hashes_by_block_size.insert(block_size, hashes);
            matches_by_block_size.insert(block_size, matches);
        }
    }

    pool.iter()
        .map(|&slot| {
            let worker = inner.worker(slot);
            let block_size = worker.meta.block_size;
            let matched = matches_by_block_size
                .get(&block_size)
                .and_then(|matches| matches.get(&slot))
                .copied()
                .unwrap_or_default();
            let hashes = hashes_by_block_size
                .get(&block_size)
                .cloned()
                .unwrap_or_default();
            let gpu_tokens = (matched.gpu_blocks * block_size).min(prompt_len);
            let cpu_tokens = (matched.cpu_blocks * block_size).min(prompt_len - gpu_tokens);
            let cached_tokens = gpu_tokens + cpu_tokens;
            let uncached = (prompt_len - cached_tokens) as f64;
            let effective_new = uncached + cpu_tokens as f64 * config.cpu_hit_cost_ratio;
            let sequences = worker.sequences();
            let mut cost = (worker.pending_prefill_tokens as f64 + effective_new)
                * config.prefill_us_per_token
                + config.decode_interference_us * f64::from(sequences);
            if worker.load.kv_usage > config.max_kv_usage {
                // Nearly full: admitting more work forces evictions or queueing.
                cost *= 4.0;
            }
            Scored {
                slot,
                hashes,
                cached_tokens,
                new_tokens: (prompt_len - cached_tokens) as u64,
                sequences,
                kv_usage: worker.load.kv_usage,
                cost,
            }
        })
        .collect()
}

fn least_loaded(scored: &[Scored]) -> usize {
    scored
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            a.sequences
                .cmp(&b.sequences)
                .then(a.kv_usage.total_cmp(&b.kv_usage))
                .then(a.slot.cmp(&b.slot))
        })
        .map(|(i, _)| i)
        .expect("scored is not empty")
}

fn cheapest(scored: &[Scored]) -> usize {
    scored
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            a.cost
                .total_cmp(&b.cost)
                .then(a.sequences.cmp(&b.sequences))
                .then(a.slot.cmp(&b.slot))
        })
        .map(|(i, _)| i)
        .expect("scored is not empty")
}

/// Rendezvous hashing: the candidate with the highest hash of
/// (session key, worker id).
fn affinity_choice(inner: &Inner, scored: &[Scored], key: &str) -> usize {
    scored
        .iter()
        .enumerate()
        .max_by_key(|(_, s)| {
            let id = &inner.worker(s.slot).meta.id;
            xxhash_rust::xxh3::xxh3_64_with_seed(
                format!("{key}\u{0}{id}").as_bytes(),
                hashing::ROOT_SEED,
            )
        })
        .map(|(i, _)| i)
        .expect("scored is not empty")
}

fn choose(
    inner: &mut Inner,
    round_robin: &AtomicU64,
    policy: Policy,
    scored: &[Scored],
    request: &RouteRequest<'_>,
) -> (usize, bool) {
    let base = match policy {
        Policy::RoundRobin => (round_robin.fetch_add(1, Ordering::Relaxed) as usize) % scored.len(),
        Policy::LeastLoad => least_loaded(scored),
        Policy::PowerOfTwo => {
            if scored.len() == 1 {
                0
            } else {
                let n = scored.len() as u64;
                let a = (inner.next_random() % n) as usize;
                let mut b = (inner.next_random() % (n - 1)) as usize;
                if b >= a {
                    b += 1;
                }
                let pair = [scored[a].clone(), scored[b].clone()];
                if least_loaded(&pair) == 0 { a } else { b }
            }
        }
        Policy::CacheAware => cheapest(scored),
    };
    let Some(key) = request
        .session_key
        .filter(|_| inner.config.session_affinity && scored.len() > 1)
    else {
        return (base, false);
    };
    let preferred = affinity_choice(inner, scored, key);
    if preferred == base {
        return (base, true);
    }
    let acceptable = match policy {
        Policy::CacheAware => {
            scored[preferred].cost <= scored[base].cost * (1.0 + inner.config.affinity_slack)
        }
        _ => scored[preferred].sequences <= scored[base].sequences + 2,
    };
    if acceptable {
        (preferred, true)
    } else {
        (base, false)
    }
}

/// Account for a request on its prefill (or unified) worker.
fn commit(
    inner: &mut Inner,
    chosen: &Scored,
    now: Instant,
    entries: &mut Vec<TicketEntry>,
) -> Target {
    let mode = inner.config.index_mode;
    let shadow = inner.config.shadow;
    let worker = inner.worker_mut(chosen.slot);
    worker.inflight += 1;
    worker.dispatched += 1;
    worker.pending_prefill_tokens += chosen.new_tokens;
    entries.push(TicketEntry {
        slot: chosen.slot,
        generation: worker.generation,
        prefill_tokens: chosen.new_tokens,
        ready: false,
    });
    let target = Target {
        worker_id: worker.meta.id.clone(),
        endpoint: worker.meta.endpoint.clone(),
        cached_tokens: chosen.cached_tokens,
    };
    let capacity = worker.meta.total_blocks as usize;
    if worker.index_source(mode) == IndexSource::Shadow {
        inner
            .index
            .shadow_touch(chosen.slot, &chosen.hashes, capacity, shadow, now);
    }
    target
}

#[cfg(test)]
mod tests;
