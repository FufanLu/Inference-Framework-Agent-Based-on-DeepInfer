//! Print the router's reasoning for a few hand-built situations.
//!
//! `cargo run -p lev-router --example explain` produces the numbers used in
//! docs/illustrated/04-cache-aware-routing.md.

use lev_router::{
    KvChange, Medium, Policy, Role, RouteRequest, Router, RouterConfig, WorkerLoad, WorkerMeta,
    hashing,
};

const MODEL: &str = "DeepSeek-V4.1-Flash";
const BLOCK: usize = 64;

fn worker(id: &str) -> WorkerMeta {
    WorkerMeta {
        id: id.into(),
        endpoint: format!("http://{id}:50051"),
        model: MODEL.into(),
        role: Role::Unified,
        block_size: BLOCK,
        total_blocks: 300_000,
        kv_events: true,
    }
}

fn main() {
    // A 4096-token agent system prompt shared by every request, plus a
    // 512-token user turn.
    let system: Vec<u32> = (0..4096).collect();
    let mut prompt = system.clone();
    prompt.extend(100_000..100_512);

    for (title, a_running, b_running) in [
        ("1. b caches the system prompt; both idle", 0, 0),
        ("2. b caches it but runs 24 sequences", 0, 24),
        ("3. b caches it and runs 80 sequences", 0, 80),
    ] {
        let router = Router::new(RouterConfig {
            policy: Policy::CacheAware,
            ..RouterConfig::default()
        });
        for id in ["worker-a", "worker-b"] {
            router.upsert_worker(worker(id)).unwrap();
            router.set_healthy(id, true);
        }
        router.apply_kv(
            "worker-b",
            KvChange::Stored {
                hashes: hashing::block_hashes(&system, BLOCK),
                medium: Medium::Gpu,
            },
        );
        router.update_load(
            "worker-a",
            WorkerLoad {
                running: a_running,
                prefill_tokens_per_s: 40_000.0,
                ..WorkerLoad::default()
            },
        );
        router.update_load(
            "worker-b",
            WorkerLoad {
                running: b_running,
                prefill_tokens_per_s: 40_000.0,
                ..WorkerLoad::default()
            },
        );
        let decision = router
            .route(RouteRequest {
                model: MODEL,
                prompt: &prompt,
                session_key: None,
                exclude: &[],
            })
            .unwrap();
        println!("## {title}");
        println!(
            "{}",
            serde_json::to_string_pretty(&decision.candidates).unwrap()
        );
    }
}
