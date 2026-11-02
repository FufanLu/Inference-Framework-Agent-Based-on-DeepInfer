use super::*;

const MODEL: &str = "DeepSeek-V4.1-Flash";
const BLOCK: usize = 4;

fn meta(id: &str, role: Role) -> WorkerMeta {
    WorkerMeta {
        id: id.to_string(),
        endpoint: format!("http://{id}"),
        model: MODEL.to_string(),
        role,
        block_size: BLOCK,
        total_blocks: 1_000,
        kv_events: true,
    }
}

fn router_with(config: RouterConfig, workers: &[(&str, Role)]) -> Router {
    let router = Router::new(config);
    for (id, role) in workers {
        router.upsert_worker(meta(id, *role)).unwrap();
        router.set_healthy(id, true);
    }
    router
}

fn config(policy: Policy) -> RouterConfig {
    RouterConfig {
        policy,
        ..RouterConfig::default()
    }
}

fn request<'a>(prompt: &'a [u32]) -> RouteRequest<'a> {
    RouteRequest {
        model: MODEL,
        prompt,
        session_key: None,
        exclude: &[],
    }
}

fn unified_id(decision: &RouteDecision) -> &str {
    match &decision.plan {
        Plan::Unified { worker } => &worker.worker_id,
        Plan::Disaggregated { .. } => panic!("expected a unified plan"),
    }
}

fn prompt(len: u32, offset: u32) -> Vec<u32> {
    (offset..offset + len).collect()
}

#[test]
fn round_robin_cycles_through_workers() {
    let router = router_with(
        config(Policy::RoundRobin),
        &[("a", Role::Unified), ("b", Role::Unified)],
    );
    let p = prompt(8, 0);
    let ids: Vec<String> = (0..4)
        .map(|_| unified_id(&router.route(request(&p)).unwrap()).to_string())
        .collect();
    assert_eq!(ids, ["a", "b", "a", "b"]);
}

#[test]
fn least_load_counts_in_flight_requests() {
    let router = router_with(
        config(Policy::LeastLoad),
        &[("a", Role::Unified), ("b", Role::Unified)],
    );
    let p = prompt(8, 0);
    let first = router.route(request(&p)).unwrap();
    let second = router.route(request(&p)).unwrap();
    assert_ne!(unified_id(&first), unified_id(&second));
    drop(first);
    // The first worker is idle again.
    let third = router.route(request(&p)).unwrap();
    assert_ne!(unified_id(&third), unified_id(&second));
}

#[test]
fn cache_aware_follows_kv_events() {
    let router = router_with(
        config(Policy::CacheAware),
        &[("a", Role::Unified), ("b", Role::Unified)],
    );
    let p = prompt(64, 0);
    let hashes = hashing::block_hashes(&p, BLOCK);
    router.apply_kv(
        "b",
        KvChange::Stored {
            hashes: hashes.clone(),
            medium: Medium::Gpu,
        },
    );
    let decision = router.route(request(&p)).unwrap();
    assert_eq!(unified_id(&decision), "b");
    assert_eq!(decision.plan.prefill_target().cached_tokens, 64);
    let chosen: Vec<_> = decision.candidates.iter().filter(|c| c.chosen).collect();
    assert_eq!(chosen.len(), 1);
    assert_eq!(chosen[0].worker_id, "b");
}

#[test]
fn cache_aware_trades_cache_hits_against_load() {
    let router = router_with(
        RouterConfig {
            policy: Policy::CacheAware,
            decode_interference_us: 50_000.0,
            ..RouterConfig::default()
        },
        &[("a", Role::Unified), ("b", Role::Unified)],
    );
    let short = prompt(16, 0);
    router.apply_kv(
        "b",
        KvChange::Stored {
            hashes: hashing::block_hashes(&short, BLOCK),
            medium: Medium::Gpu,
        },
    );
    router.update_load(
        "b",
        WorkerLoad {
            running: 40,
            ..WorkerLoad::default()
        },
    );
    // Saving 16 tokens of prefill is not worth queueing behind 40 sequences.
    let decision = router.route(request(&short)).unwrap();
    assert_eq!(unified_id(&decision), "a");
}

#[test]
fn cpu_tier_hits_count_but_cost_more_than_gpu_hits() {
    let router = router_with(
        config(Policy::CacheAware),
        &[("a", Role::Unified), ("b", Role::Unified)],
    );
    let p = prompt(64, 0);
    let hashes = hashing::block_hashes(&p, BLOCK);
    router.apply_kv(
        "a",
        KvChange::Stored {
            hashes: hashes.clone(),
            medium: Medium::Cpu,
        },
    );
    router.apply_kv(
        "b",
        KvChange::Stored {
            hashes,
            medium: Medium::Gpu,
        },
    );
    let decision = router.route(request(&p)).unwrap();
    assert_eq!(unified_id(&decision), "b");
    let a = decision
        .candidates
        .iter()
        .find(|c| c.worker_id == "a")
        .unwrap();
    assert_eq!(a.cached_tokens, 64);
}

#[test]
fn approximate_mode_remembers_where_prompts_went() {
    let router = router_with(
        RouterConfig {
            policy: Policy::CacheAware,
            index_mode: IndexMode::Approximate,
            ..RouterConfig::default()
        },
        &[("a", Role::Unified), ("b", Role::Unified)],
    );
    let shared = prompt(256, 0);
    let first = router.route(request(&shared)).unwrap();
    let first_id = unified_id(&first).to_string();
    drop(first);
    let mut follow_up = shared.clone();
    follow_up.extend(prompt(32, 10_000));
    let second = router.route(request(&follow_up)).unwrap();
    assert_eq!(unified_id(&second), first_id);
    assert_eq!(second.plan.prefill_target().cached_tokens, 256);
}

#[test]
fn approximate_mode_ignores_kv_events() {
    let router = router_with(
        RouterConfig {
            policy: Policy::CacheAware,
            index_mode: IndexMode::Approximate,
            ..RouterConfig::default()
        },
        &[("a", Role::Unified)],
    );
    let p = prompt(16, 0);
    router.apply_kv(
        "a",
        KvChange::Stored {
            hashes: hashing::block_hashes(&p, BLOCK),
            medium: Medium::Gpu,
        },
    );
    assert_eq!(router.snapshot().workers[0].gpu_blocks, 0);
}

#[test]
fn disaggregated_plan_uses_both_pools() {
    let router = router_with(
        config(Policy::CacheAware),
        &[
            ("p0", Role::Prefill),
            ("p1", Role::Prefill),
            ("d0", Role::Decode),
            ("d1", Role::Decode),
        ],
    );
    let p = prompt(64, 0);
    router.apply_kv(
        "p1",
        KvChange::Stored {
            hashes: hashing::block_hashes(&p, BLOCK),
            medium: Medium::Gpu,
        },
    );
    let first = router.route(request(&p)).unwrap();
    let Plan::Disaggregated { prefill, decode } = &first.plan else {
        panic!("expected disaggregation");
    };
    assert_eq!(prefill.worker_id, "p1");
    let first_decode = decode.worker_id.clone();
    let second = router.route(request(&p)).unwrap();
    let Plan::Disaggregated { decode, .. } = &second.plan else {
        panic!("expected disaggregation");
    };
    // Decode is balanced by load.
    assert_ne!(decode.worker_id, first_decode);
}

#[test]
fn single_pool_runs_requests_end_to_end() {
    let router = router_with(config(Policy::LeastLoad), &[("p0", Role::Prefill)]);
    let p = prompt(8, 0);
    let decision = router.route(request(&p)).unwrap();
    assert_eq!(unified_id(&decision), "p0");
}

#[test]
fn disaggregation_can_be_disabled() {
    let router = router_with(
        RouterConfig {
            disaggregation: false,
            ..config(Policy::LeastLoad)
        },
        &[("p0", Role::Prefill), ("d0", Role::Decode)],
    );
    let p = prompt(8, 0);
    assert!(matches!(
        router.route(request(&p)).unwrap().plan,
        Plan::Unified { .. }
    ));
}

#[test]
fn tickets_release_load() {
    let router = router_with(config(Policy::CacheAware), &[("a", Role::Unified)]);
    let p = prompt(100, 0);
    let mut decision = router.route(request(&p)).unwrap();
    let view = &router.snapshot().workers[0];
    assert_eq!(view.inflight, 1);
    assert_eq!(view.pending_prefill_tokens, 100);
    decision.ticket.prefill_done();
    let view = &router.snapshot().workers[0];
    assert_eq!(view.inflight, 1);
    assert_eq!(view.pending_prefill_tokens, 0);
    drop(decision);
    let view = &router.snapshot().workers[0];
    assert_eq!(view.inflight, 0);
    assert_eq!(view.dispatched, 1);
}

#[test]
fn stale_tickets_do_not_touch_a_reused_slot() {
    let router = router_with(config(Policy::LeastLoad), &[("a", Role::Unified)]);
    let p = prompt(8, 0);
    let decision = router.route(request(&p)).unwrap();
    router.remove_worker("a");
    router.upsert_worker(meta("b", Role::Unified)).unwrap();
    router.set_healthy("b", true);
    let held = router.route(request(&p)).unwrap();
    drop(decision);
    assert_eq!(router.snapshot().workers[0].inflight, 1);
    drop(held);
    assert_eq!(router.snapshot().workers[0].inflight, 0);
}

#[test]
fn excluded_and_unhealthy_workers_are_skipped() {
    let router = router_with(
        config(Policy::RoundRobin),
        &[("a", Role::Unified), ("b", Role::Unified)],
    );
    let p = prompt(8, 0);
    let exclude = vec!["a".to_string()];
    for _ in 0..3 {
        let decision = router
            .route(RouteRequest {
                exclude: &exclude,
                ..request(&p)
            })
            .unwrap();
        assert_eq!(unified_id(&decision), "b");
    }
    router.set_healthy("b", false);
    assert!(matches!(
        router.route(RouteRequest {
            exclude: &exclude,
            ..request(&p)
        }),
        Err(RouteError::NoWorkers { .. })
    ));
}

#[test]
fn losing_health_drops_cache_view() {
    let router = router_with(config(Policy::CacheAware), &[("a", Role::Unified)]);
    let p = prompt(16, 0);
    router.apply_kv(
        "a",
        KvChange::Stored {
            hashes: hashing::block_hashes(&p, BLOCK),
            medium: Medium::Gpu,
        },
    );
    assert_eq!(router.snapshot().workers[0].gpu_blocks, 4);
    router.set_healthy("a", false);
    assert_eq!(router.snapshot().workers[0].gpu_blocks, 0);
}

#[test]
fn other_models_are_not_candidates() {
    let router = router_with(config(Policy::RoundRobin), &[("a", Role::Unified)]);
    let p = prompt(8, 0);
    let result = router.route(RouteRequest {
        model: "DeepSeek-V4-Pro",
        ..request(&p)
    });
    assert!(matches!(result, Err(RouteError::NoWorkers { .. })));
}

#[test]
fn session_affinity_is_stable() {
    let router = router_with(
        RouterConfig {
            session_affinity: true,
            ..config(Policy::RoundRobin)
        },
        &[
            ("a", Role::Unified),
            ("b", Role::Unified),
            ("c", Role::Unified),
        ],
    );
    let p = prompt(8, 0);
    let chosen: Vec<String> = (0..6)
        .map(|_| {
            let decision = router
                .route(RouteRequest {
                    session_key: Some("user-42"),
                    ..request(&p)
                })
                .unwrap();
            assert!(decision.affinity);
            unified_id(&decision).to_string()
        })
        .collect();
    assert!(chosen.windows(2).all(|pair| pair[0] == pair[1]));
}

#[test]
fn power_of_two_picks_the_less_loaded_of_its_pair() {
    let router = router_with(
        config(Policy::PowerOfTwo),
        &[("a", Role::Unified), ("b", Role::Unified)],
    );
    router.update_load(
        "a",
        WorkerLoad {
            running: 10,
            ..WorkerLoad::default()
        },
    );
    let p = prompt(8, 0);
    for _ in 0..8 {
        let decision = router.route(request(&p)).unwrap();
        assert_eq!(unified_id(&decision), "b");
    }
}

#[test]
fn policies_parse_from_their_names() {
    for policy in Policy::ALL {
        assert_eq!(policy.as_str().parse::<Policy>().unwrap(), policy);
    }
    assert!("fastest".parse::<Policy>().is_err());
}

mod properties {
    use super::super::index::{PrefixIndex, slots};
    use super::*;
    use proptest::prelude::*;
    use std::collections::HashSet;

    #[derive(Debug, Clone)]
    enum Op {
        Store(usize, Vec<u64>),
        Remove(usize, Vec<u64>),
        Clear(usize),
    }

    fn op() -> impl Strategy<Value = Op> {
        let hashes = proptest::collection::vec(0u64..24, 0..8);
        prop_oneof![
            (0usize..3, hashes.clone()).prop_map(|(s, h)| Op::Store(s, h)),
            (0usize..3, hashes).prop_map(|(s, h)| Op::Remove(s, h)),
            (0usize..3).prop_map(Op::Clear),
        ]
    }

    proptest! {
        /// The index agrees with a naive per-slot set model after any
        /// sequence of stores, removals and clears.
        #[test]
        fn index_matches_naive_model(
            ops in proptest::collection::vec(op(), 0..64),
            query in proptest::collection::vec(0u64..24, 0..12),
        ) {
            let mut index = PrefixIndex::new();
            let mut model: Vec<HashSet<u64>> = vec![HashSet::new(); 3];
            for op in ops {
                match op {
                    Op::Store(slot, hashes) => {
                        index.store(slot, &hashes, Medium::Gpu);
                        model[slot].extend(hashes);
                    }
                    Op::Remove(slot, hashes) => {
                        index.remove(slot, &hashes, Medium::Gpu);
                        for hash in hashes {
                            model[slot].remove(&hash);
                        }
                    }
                    Op::Clear(slot) => {
                        index.clear_slot(slot);
                        model[slot].clear();
                    }
                }
            }
            for (slot, matched) in index.match_prefix(&query, 0b111) {
                let expected = query.iter().take_while(|h| model[slot].contains(h)).count();
                prop_assert_eq!(matched.blocks(), expected);
                prop_assert_eq!(index.blocks(slot, Medium::Gpu), model[slot].len());
            }
            prop_assert_eq!(slots(0b111).count(), 3);
        }
    }
}
