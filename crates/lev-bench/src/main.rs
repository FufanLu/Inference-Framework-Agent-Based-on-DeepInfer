//! Leviathan benchmark harness.
//!
//! Runs one workload against the gateway under each routing policy, switching
//! the policy through the admin API between runs, and reports client-side
//! TTFT, TPOT, end-to-end latency, throughput and prefix-cache hit rate.
//!
//! ```sh
//! lev-bench --gateway http://127.0.0.1:8080 --workload agent-shared-prefix \
//!     --policies round_robin,least_load,p2c,cache_aware --requests 400 --concurrency 32
//! ```

mod workload;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;
use clap::Parser;
use futures::StreamExt;
use serde::Serialize;
use serde_json::{Value, json};
use tokio::io::AsyncWriteExt;
use tokio::sync::Semaphore;

use workload::{Kind, Planned, Spec};

#[derive(Debug, Parser)]
#[command(version, about)]
struct Args {
    #[arg(long, default_value = "http://127.0.0.1:8080")]
    gateway: String,
    #[arg(long, value_enum, default_value = "agent-shared-prefix")]
    workload: Kind,
    /// Comma-separated policies to compare.
    #[arg(long, default_value = "round_robin,least_load,p2c,cache_aware")]
    policies: String,
    /// Prefix index mode for every run: exact, approximate or off.
    #[arg(long)]
    index_mode: Option<String>,
    #[arg(long, default_value_t = 200)]
    requests: usize,
    /// Requests in flight at once (closed loop).
    #[arg(long, default_value_t = 16)]
    concurrency: usize,
    /// Open loop: Poisson arrivals at this rate (requests/s), still capped by
    /// --concurrency.
    #[arg(long)]
    rate: Option<f64>,
    #[arg(long, default_value = "deepseek-flash")]
    model: String,
    #[arg(long, default_value_t = 64)]
    max_tokens: u32,
    #[arg(long, default_value_t = 8)]
    agents: usize,
    #[arg(long, default_value_t = 1200)]
    system_words: usize,
    #[arg(long, default_value_t = 40)]
    question_words: usize,
    #[arg(long, default_value_t = 16)]
    sessions: usize,
    #[arg(long)]
    thinking: bool,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Bearer token for the admin API.
    #[arg(long, env = "LEV_ADMIN_TOKEN")]
    admin_token: Option<String>,
    /// API key for the inference API.
    #[arg(long, env = "LEV_API_KEY")]
    api_key: Option<String>,
    /// Directory for per-request records and the summary.
    #[arg(long, default_value = "bench/results/latest")]
    out: PathBuf,
    /// Label stored with the results (e.g. hardware).
    #[arg(long, default_value = "local-sim")]
    tag: String,
}

#[derive(Debug, Clone, Serialize)]
struct Record {
    policy: String,
    index: usize,
    group: usize,
    ok: bool,
    error: Option<String>,
    status: u16,
    ttft_ms: Option<f64>,
    tpot_ms: Option<f64>,
    e2e_ms: f64,
    prompt_tokens: u64,
    cached_tokens: u64,
    completion_tokens: u64,
}

#[derive(Debug, Clone, Serialize)]
struct Summary {
    tag: String,
    workload: String,
    policy: String,
    index_mode: Option<String>,
    requests: usize,
    errors: usize,
    concurrency: usize,
    rate: Option<f64>,
    wall_s: f64,
    req_per_s: f64,
    output_tok_per_s: f64,
    ttft_p50_ms: f64,
    ttft_p90_ms: f64,
    ttft_p99_ms: f64,
    tpot_p50_ms: f64,
    e2e_p50_ms: f64,
    e2e_p99_ms: f64,
    prompt_tokens: u64,
    cached_tokens: u64,
    cache_hit_rate: f64,
}

fn percentile(values: &mut [f64], q: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(f64::total_cmp);
    let rank = ((values.len() - 1) as f64 * q).round() as usize;
    values[rank]
}

/// Usage fields of a final chunk, in either DeepSeek or OpenAI spelling.
fn usage_tokens(usage: &Value) -> (u64, u64, u64) {
    let prompt = usage["prompt_tokens"].as_u64().unwrap_or(0);
    let cached = usage["prompt_cache_hit_tokens"]
        .as_u64()
        .or_else(|| usage["prompt_tokens_details"]["cached_tokens"].as_u64())
        .unwrap_or(0);
    let completion = usage["completion_tokens"].as_u64().unwrap_or(0);
    (prompt, cached, completion)
}

async fn run_one(
    client: &reqwest::Client,
    args: &Args,
    policy: &str,
    index: usize,
    planned: Planned,
) -> Record {
    let started = Instant::now();
    let mut record = Record {
        policy: policy.to_string(),
        index,
        group: planned.group,
        ok: false,
        error: None,
        status: 0,
        ttft_ms: None,
        tpot_ms: None,
        e2e_ms: 0.0,
        prompt_tokens: 0,
        cached_tokens: 0,
        completion_tokens: 0,
    };
    let mut request = client
        .post(format!("{}/v1/chat/completions", args.gateway))
        .json(&planned.body);
    if let Some(session) = &planned.session {
        request = request.header("x-session-id", session);
    }
    if let Some(key) = &args.api_key {
        request = request.bearer_auth(key);
    }
    let response = match request.send().await {
        Ok(response) => response,
        Err(error) => {
            record.error = Some(error.to_string());
            record.e2e_ms = started.elapsed().as_secs_f64() * 1e3;
            return record;
        }
    };
    record.status = response.status().as_u16();
    if !response.status().is_success() {
        record.error = Some(response.text().await.unwrap_or_default());
        record.e2e_ms = started.elapsed().as_secs_f64() * 1e3;
        return record;
    }
    let mut first_token: Option<Instant> = None;
    let mut last_token: Option<Instant> = None;
    let mut chunks_with_content = 0u64;
    let mut buffer = String::new();
    let mut body = response.bytes_stream();
    let mut saw_done = false;
    while let Some(bytes) = body.next().await {
        let bytes = match bytes {
            Ok(bytes) => bytes,
            Err(error) => {
                record.error = Some(error.to_string());
                break;
            }
        };
        buffer.push_str(&String::from_utf8_lossy(&bytes));
        while let Some(end) = buffer.find("\n\n") {
            let frame: String = buffer.drain(..end + 2).collect();
            for line in frame.lines() {
                let Some(data) = line.strip_prefix("data: ") else {
                    continue;
                };
                if data == "[DONE]" {
                    saw_done = true;
                    continue;
                }
                let Ok(chunk) = serde_json::from_str::<Value>(data) else {
                    continue;
                };
                if chunk.get("error").is_some() {
                    record.error = Some(chunk["error"].to_string());
                }
                let delta = &chunk["choices"][0]["delta"];
                let has_output = delta["content"].as_str().is_some_and(|s| !s.is_empty())
                    || delta["reasoning_content"]
                        .as_str()
                        .is_some_and(|s| !s.is_empty())
                    || delta.get("tool_calls").is_some();
                if has_output {
                    let now = Instant::now();
                    first_token.get_or_insert(now);
                    last_token = Some(now);
                    chunks_with_content += 1;
                }
                if chunk["usage"].is_object() {
                    let (prompt, cached, completion) = usage_tokens(&chunk["usage"]);
                    record.prompt_tokens = prompt;
                    record.cached_tokens = cached;
                    record.completion_tokens = completion;
                }
            }
        }
    }
    record.e2e_ms = started.elapsed().as_secs_f64() * 1e3;
    record.ttft_ms = first_token.map(|at| at.duration_since(started).as_secs_f64() * 1e3);
    let decoded = record.completion_tokens.max(chunks_with_content);
    if let (Some(first), Some(last)) = (first_token, last_token)
        && decoded > 1
    {
        record.tpot_ms =
            Some(last.duration_since(first).as_secs_f64() * 1e3 / (decoded - 1) as f64);
    }
    record.ok = record.error.is_none() && saw_done;
    if !saw_done && record.error.is_none() {
        record.error = Some("stream ended without [DONE]".to_string());
    }
    record
}

async fn set_router(client: &reqwest::Client, args: &Args, policy: &str) -> anyhow::Result<()> {
    let mut body = json!({ "policy": policy });
    if let Some(mode) = &args.index_mode {
        body["index_mode"] = json!(mode);
    }
    let mut request = client
        .put(format!("{}/admin/router", args.gateway))
        .json(&body);
    if let Some(token) = &args.admin_token {
        request = request.bearer_auth(token);
    }
    let response = request.send().await.context("PUT /admin/router")?;
    anyhow::ensure!(
        response.status().is_success(),
        "PUT /admin/router failed: {}",
        response.text().await.unwrap_or_default()
    );
    Ok(())
}

fn exponential(rng: &mut rand::rngs::StdRng, rate: f64) -> Duration {
    use rand::Rng;
    let u: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE);
    Duration::from_secs_f64(-u.ln() / rate)
}

async fn run_policy(
    client: &reqwest::Client,
    args: Arc<Args>,
    policy: &str,
) -> anyhow::Result<(Summary, Vec<Record>)> {
    set_router(client, &args, policy).await?;
    let spec = Spec {
        kind: args.workload,
        requests: args.requests,
        model: args.model.clone(),
        max_tokens: args.max_tokens,
        agents: args.agents,
        system_words: args.system_words,
        question_words: args.question_words,
        sessions: args.sessions,
        thinking: args.thinking,
        seed: args.seed,
    };
    // Same request sequence for every policy, different prefixes per run.
    let salt = format!("{policy}-{}", args.seed);
    let planned = workload::plan(&spec, &salt);
    let semaphore = Arc::new(Semaphore::new(args.concurrency.max(1)));
    let mut arrivals = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(args.seed ^ 0xA11);
    let started = Instant::now();
    let mut tasks = Vec::with_capacity(planned.len());
    for (index, request) in planned.into_iter().enumerate() {
        if let Some(rate) = args.rate {
            tokio::time::sleep(exponential(&mut arrivals, rate)).await;
        }
        let permit = Arc::clone(&semaphore).acquire_owned().await?;
        let client = client.clone();
        let args = Arc::clone(&args);
        let policy = policy.to_string();
        tasks.push(tokio::spawn(async move {
            let record = run_one(&client, &args, &policy, index, request).await;
            drop(permit);
            record
        }));
    }
    let mut records = Vec::with_capacity(tasks.len());
    for task in tasks {
        records.push(task.await?);
    }
    let wall = started.elapsed().as_secs_f64();
    let ok: Vec<&Record> = records.iter().filter(|r| r.ok).collect();
    let mut ttft: Vec<f64> = ok.iter().filter_map(|r| r.ttft_ms).collect();
    let mut tpot: Vec<f64> = ok.iter().filter_map(|r| r.tpot_ms).collect();
    let mut e2e: Vec<f64> = ok.iter().map(|r| r.e2e_ms).collect();
    let prompt_tokens: u64 = ok.iter().map(|r| r.prompt_tokens).sum();
    let cached_tokens: u64 = ok.iter().map(|r| r.cached_tokens).sum();
    let output_tokens: u64 = ok.iter().map(|r| r.completion_tokens).sum();
    let summary = Summary {
        tag: args.tag.clone(),
        workload: format!("{:?}", args.workload),
        policy: policy.to_string(),
        index_mode: args.index_mode.clone(),
        requests: records.len(),
        errors: records.len() - ok.len(),
        concurrency: args.concurrency,
        rate: args.rate,
        wall_s: wall,
        req_per_s: ok.len() as f64 / wall,
        output_tok_per_s: output_tokens as f64 / wall,
        ttft_p50_ms: percentile(&mut ttft, 0.5),
        ttft_p90_ms: percentile(&mut ttft, 0.9),
        ttft_p99_ms: percentile(&mut ttft, 0.99),
        tpot_p50_ms: percentile(&mut tpot, 0.5),
        e2e_p50_ms: percentile(&mut e2e, 0.5),
        e2e_p99_ms: percentile(&mut e2e, 0.99),
        prompt_tokens,
        cached_tokens,
        cache_hit_rate: if prompt_tokens > 0 {
            cached_tokens as f64 / prompt_tokens as f64
        } else {
            0.0
        },
    };
    Ok((summary, records))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Arc::new(Args::parse());
    let client = reqwest::Client::builder()
        .pool_max_idle_per_host(args.concurrency * 2)
        .timeout(Duration::from_secs(600))
        .build()?;
    tokio::fs::create_dir_all(&args.out).await?;
    let records_path = args.out.join("requests.jsonl");
    let summary_path = args.out.join("summary.jsonl");
    let mut records_file = tokio::fs::File::create(&records_path).await?;
    let mut summary_file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&summary_path)
        .await?;

    let policies: Vec<String> = args
        .policies
        .split(',')
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
        .collect();
    let mut summaries = Vec::new();
    for policy in &policies {
        eprintln!(
            "running {:?} with policy {policy} ({} requests, concurrency {})",
            args.workload, args.requests, args.concurrency
        );
        let (summary, records) = run_policy(&client, Arc::clone(&args), policy).await?;
        for record in &records {
            records_file
                .write_all(format!("{}\n", serde_json::to_string(record)?).as_bytes())
                .await?;
        }
        summary_file
            .write_all(format!("{}\n", serde_json::to_string(&summary)?).as_bytes())
            .await?;
        summaries.push(summary);
    }

    println!(
        "\n| policy | ok/total | TTFT p50 | TTFT p90 | TTFT p99 | TPOT p50 | E2E p50 | out tok/s | cache hit |"
    );
    println!("| --- | --- | --- | --- | --- | --- | --- | --- | --- |");
    for s in &summaries {
        println!(
            "| {} | {}/{} | {:.1} ms | {:.1} ms | {:.1} ms | {:.2} ms | {:.1} ms | {:.0} | {:.1}% |",
            s.policy,
            s.requests - s.errors,
            s.requests,
            s.ttft_p50_ms,
            s.ttft_p90_ms,
            s.ttft_p99_ms,
            s.tpot_p50_ms,
            s.e2e_p50_ms,
            s.output_tok_per_s,
            s.cache_hit_rate * 100.0
        );
    }
    eprintln!(
        "\nwrote {} and {}",
        records_path.display(),
        summary_path.display()
    );
    Ok(())
}
