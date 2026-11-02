//! Prometheus metrics.

use metrics_exporter_prometheus::{Matcher, PrometheusBuilder, PrometheusHandle};

const LATENCY_BUCKETS: &[f64] = &[
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0,
];
const TPOT_BUCKETS: &[f64] = &[
    0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.25,
];

/// Install the global Prometheus recorder.
pub fn install() -> anyhow::Result<PrometheusHandle> {
    let handle = PrometheusBuilder::new()
        .set_buckets_for_metric(Matcher::Full("lev_ttft_seconds".into()), LATENCY_BUCKETS)?
        .set_buckets_for_metric(Matcher::Full("lev_e2e_seconds".into()), LATENCY_BUCKETS)?
        .set_buckets_for_metric(Matcher::Full("lev_tpot_seconds".into()), TPOT_BUCKETS)?
        .install_recorder()?;
    describe();
    Ok(handle)
}

fn describe() {
    use metrics::{describe_counter, describe_gauge, describe_histogram};
    describe_gauge!(
        "lev_worker_running",
        "Sequences running on a worker (its last report)."
    );
    describe_gauge!("lev_worker_waiting", "Sequences queued on a worker.");
    describe_gauge!(
        "lev_worker_kv_usage",
        "Fraction of a worker's KV blocks in use."
    );
    describe_gauge!(
        "lev_worker_decode_tokens_per_second",
        "Decode throughput of a worker."
    );
    describe_gauge!(
        "lev_worker_spec_accept_rate",
        "Speculative draft acceptance of a worker."
    );
    describe_counter!(
        "lev_requests_total",
        "Requests by protocol, model and outcome."
    );
    describe_counter!(
        "lev_route_decisions_total",
        "Routing decisions by policy and plan."
    );
    describe_counter!(
        "lev_dispatch_retries_total",
        "Dispatch attempts retried on another worker."
    );
    describe_counter!(
        "lev_prompt_tokens_total",
        "Prompt tokens by model and tenant."
    );
    describe_counter!(
        "lev_cached_tokens_total",
        "Prompt tokens served from the prefix cache."
    );
    describe_counter!(
        "lev_completion_tokens_total",
        "Output tokens by model and tenant."
    );
    describe_counter!(
        "lev_spec_draft_tokens_total",
        "Speculative draft tokens proposed."
    );
    describe_counter!(
        "lev_spec_accepted_tokens_total",
        "Speculative draft tokens accepted."
    );
    describe_histogram!(
        "lev_ttft_seconds",
        "Time to first token, measured at the gateway."
    );
    describe_histogram!("lev_tpot_seconds", "Time per output token after the first.");
    describe_histogram!("lev_e2e_seconds", "End-to-end request latency.");
}
