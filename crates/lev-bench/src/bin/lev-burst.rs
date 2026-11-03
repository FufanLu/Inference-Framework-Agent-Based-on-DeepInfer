//! A burst against steady interactive traffic.
//!
//! Interactive clients run a closed loop of streaming chat requests for the
//! whole run. At `--burst-at` seconds a burst of `--burst-requests` arrives
//! at once, either straight to `/v1/chat/completions` (`--mode direct`) or as
//! asynchronous jobs (`--mode jobs`); `--mode none` is the baseline. The
//! question is what the burst does to the interactive requests' TTFT, and how
//! long the burst takes to finish.
//!
//! ```sh
//! lev-burst --mode jobs --burst-requests 2000 --duration 60 --out bench/results/burst/jobs
//! ```

#[allow(dead_code)]
#[path = "../workload.rs"]
mod workload;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;
use clap::Parser;
use futures::StreamExt;
use parking_lot::Mutex;
use serde::Serialize;
use serde_json::{Value, json};

use workload::{Kind, Planned, Spec};

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
enum Mode {
    None,
    Direct,
    Jobs,
}

#[derive(Debug, Parser)]
#[command(version, about)]
struct Args {
    #[arg(long, default_value = "http://127.0.0.1:8080")]
    gateway: String,
    #[arg(long, value_enum, default_value = "jobs")]
    mode: Mode,
    /// Length of the run (seconds of interactive traffic).
    #[arg(long, default_value_t = 60.0)]
    duration: f64,
    /// Interactive clients (closed loop).
    #[arg(long, default_value_t = 8)]
    interactive: usize,
    #[arg(long, default_value_t = 64)]
    interactive_max_tokens: u32,
    /// When the burst arrives (seconds after the start).
    #[arg(long, default_value_t = 10.0)]
    burst_at: f64,
    #[arg(long, default_value_t = 2000)]
    burst_requests: usize,
    #[arg(long, default_value_t = 256)]
    burst_max_tokens: u32,
    #[arg(long, default_value_t = 400)]
    burst_words: usize,
    /// Jobs per submission in `--mode jobs`.
    #[arg(long, default_value_t = 500)]
    submit_batch: usize,
    /// Stop waiting for the burst this long after the run ends.
    #[arg(long, default_value_t = 600.0)]
    drain_timeout: f64,
    /// Runner limits to set before the run (`--mode jobs`).
    #[arg(long)]
    max_in_flight: Option<usize>,
    #[arg(long)]
    max_queue_delay_ms: Option<f64>,
    /// Engine priority of job requests (interactive requests use 0).
    #[arg(long)]
    job_priority: Option<i32>,
    #[arg(long, default_value = "deepseek-flash")]
    model: String,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, env = "LEV_ADMIN_TOKEN")]
    admin_token: Option<String>,
    #[arg(long, env = "LEV_API_KEY")]
    api_key: Option<String>,
    #[arg(long, default_value = "bench/results/burst/latest")]
    out: PathBuf,
    #[arg(long, default_value = "local-sim")]
    tag: String,
}

/// One finished request, client side.
#[derive(Debug, Clone, Serialize)]
struct Sample {
    kind: &'static str,
    /// Seconds after the start of the run.
    started_s: f64,
    finished_s: f64,
    ok: bool,
    status: u16,
    ttft_ms: Option<f64>,
    error: Option<String>,
}

fn percentile(values: &[f64], q: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let rank = (q * (sorted.len() - 1) as f64).round() as usize;
    Some(sorted[rank.min(sorted.len() - 1)])
}

fn round(value: Option<f64>) -> Value {
    value.map_or(Value::Null, |v| json!((v * 10.0).round() / 10.0))
}

struct Bench {
    args: Args,
    client: reqwest::Client,
    start: Instant,
}

impl Bench {
    fn at(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    fn auth(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.args.api_key {
            Some(key) => request.bearer_auth(key),
            None => request,
        }
    }

    fn admin(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.args.admin_token {
            Some(token) => request.bearer_auth(token),
            None => request,
        }
    }

    /// One streaming chat request; TTFT is the first frame with output.
    async fn stream(&self, kind: &'static str, body: &Value) -> Sample {
        let started_s = self.at();
        let started = Instant::now();
        let mut sample = Sample {
            kind,
            started_s,
            finished_s: 0.0,
            ok: false,
            status: 0,
            ttft_ms: None,
            error: None,
        };
        let request = self.auth(
            self.client
                .post(format!("{}/v1/chat/completions", self.args.gateway))
                .json(body),
        );
        match request.send().await {
            Err(error) => sample.error = Some(error.to_string()),
            Ok(response) => {
                sample.status = response.status().as_u16();
                if !response.status().is_success() {
                    sample.error = Some(response.text().await.unwrap_or_default());
                } else {
                    let mut stream = response.bytes_stream();
                    let mut buffer = String::new();
                    let mut done = false;
                    while let Some(bytes) = stream.next().await {
                        let Ok(bytes) = bytes else {
                            sample.error = Some("stream broke".into());
                            break;
                        };
                        buffer.push_str(&String::from_utf8_lossy(&bytes));
                        while let Some(end) = buffer.find("\n\n") {
                            let frame: String = buffer.drain(..end + 2).collect();
                            for data in frame.lines().filter_map(|l| l.strip_prefix("data: ")) {
                                if data == "[DONE]" {
                                    done = true;
                                    continue;
                                }
                                let Ok(chunk) = serde_json::from_str::<Value>(data) else {
                                    continue;
                                };
                                if chunk.get("error").is_some() {
                                    sample.error = Some(chunk["error"].to_string());
                                }
                                let delta = &chunk["choices"][0]["delta"];
                                let output = ["content", "reasoning_content"]
                                    .iter()
                                    .any(|k| delta[*k].as_str().is_some_and(|s| !s.is_empty()))
                                    || delta.get("tool_calls").is_some();
                                if output && sample.ttft_ms.is_none() {
                                    sample.ttft_ms = Some(started.elapsed().as_secs_f64() * 1e3);
                                }
                            }
                        }
                    }
                    sample.ok = done && sample.error.is_none();
                }
            }
        }
        sample.finished_s = self.at();
        sample
    }
}

fn spec(args: &Args, kind: Kind, requests: usize, max_tokens: u32, seed: u64) -> Spec {
    Spec {
        kind,
        requests,
        model: args.model.clone(),
        max_tokens,
        agents: 8,
        system_words: 1200,
        question_words: if kind == Kind::Unique {
            args.burst_words
        } else {
            40
        },
        sessions: 16,
        thinking: false,
        seed,
    }
}

async fn interactive_client(
    bench: Arc<Bench>,
    plan: Vec<Planned>,
    samples: Arc<Mutex<Vec<Sample>>>,
) {
    let deadline = bench.args.duration;
    for planned in plan.iter().cycle() {
        if bench.at() >= deadline {
            break;
        }
        let sample = bench.stream("interactive", &planned.body).await;
        samples.lock().push(sample);
    }
}

async fn burst_direct(bench: Arc<Bench>, plan: Vec<Planned>, samples: Arc<Mutex<Vec<Sample>>>) {
    let tasks = plan.into_iter().map(|planned| {
        let bench = Arc::clone(&bench);
        let samples = Arc::clone(&samples);
        tokio::spawn(async move {
            let sample = bench.stream("burst", &planned.body).await;
            samples.lock().push(sample);
        })
    });
    futures::future::join_all(tasks).await;
}

/// Submit the burst as jobs, then poll each until it finishes. Returns the
/// submission latencies (ms).
async fn burst_jobs(
    bench: Arc<Bench>,
    plan: Vec<Planned>,
    samples: Arc<Mutex<Vec<Sample>>>,
) -> anyhow::Result<Vec<f64>> {
    let mut ids = Vec::with_capacity(plan.len());
    let mut submit_ms = Vec::new();
    let submitted_s = bench.at();
    for chunk in plan.chunks(bench.args.submit_batch.max(1)) {
        let jobs: Vec<Value> = chunk
            .iter()
            .map(|planned| json!({"endpoint": "/v1/chat/completions", "body": planned.body}))
            .collect();
        let started = Instant::now();
        let response = bench
            .auth(
                bench
                    .client
                    .post(format!("{}/v1/leviathan/jobs", bench.args.gateway))
                    .json(&json!({ "jobs": jobs })),
            )
            .send()
            .await?;
        let status = response.status();
        let body: Value = response.json().await?;
        anyhow::ensure!(
            status.as_u16() == 202,
            "submission failed ({status}): {body}"
        );
        submit_ms.push(started.elapsed().as_secs_f64() * 1e3);
        ids.extend(
            body["data"]
                .as_array()
                .context("a job list")?
                .iter()
                .filter_map(|job| job["id"].as_str().map(str::to_string)),
        );
    }
    // Poll until every job finished.
    let deadline = bench.args.duration + bench.args.drain_timeout;
    let mut pending = ids;
    while !pending.is_empty() && bench.at() < deadline {
        tokio::time::sleep(Duration::from_millis(250)).await;
        let checks = pending.iter().map(|id| {
            let bench = Arc::clone(&bench);
            let id = id.clone();
            async move {
                let job: Value = bench
                    .auth(
                        bench
                            .client
                            .get(format!("{}/v1/leviathan/jobs/{id}", bench.args.gateway)),
                    )
                    .send()
                    .await
                    .ok()?
                    .json()
                    .await
                    .ok()?;
                Some((id, job))
            }
        });
        let results = futures::stream::iter(checks)
            .buffer_unordered(64)
            .collect::<Vec<_>>()
            .await;
        let mut still = Vec::new();
        let now = bench.at();
        for (id, job) in results.into_iter().flatten() {
            match job["status"].as_str() {
                Some("succeeded" | "failed") => {
                    let ok = job["status"] == "succeeded";
                    samples.lock().push(Sample {
                        kind: "burst",
                        started_s: submitted_s,
                        // Polled: accurate to the poll interval.
                        finished_s: now,
                        ok,
                        status: job["response"]["status_code"].as_u64().unwrap_or(0) as u16,
                        ttft_ms: job["queue_wait_ms"].as_f64(),
                        error: job["error"].as_str().map(str::to_string),
                    });
                }
                _ => still.push(id),
            }
        }
        pending = still;
    }
    Ok(submit_ms)
}

/// Interactive TTFT per second of the run, for the timeline chart.
fn timeline(samples: &[Sample], duration: f64) -> Vec<Value> {
    let seconds = duration.ceil() as usize;
    (0..seconds)
        .map(|second| {
            let ttfts: Vec<f64> = samples
                .iter()
                .filter(|s| s.kind == "interactive" && s.started_s as usize == second)
                .filter_map(|s| s.ttft_ms)
                .collect();
            json!({
                "t": second,
                "n": ttfts.len(),
                "ttft_p50_ms": round(percentile(&ttfts, 0.5)),
                "ttft_p99_ms": round(percentile(&ttfts, 0.99)),
            })
        })
        .collect()
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let client = reqwest::Client::builder()
        .pool_max_idle_per_host(4096)
        .timeout(Duration::from_secs(1800))
        .build()?;
    // Distinct prefixes per run: no run reuses another's cache.
    let salt = format!("burst-{}", std::process::id());
    let mut interactive_spec = spec(
        &args,
        Kind::AgentSharedPrefix,
        4096,
        args.interactive_max_tokens,
        args.seed,
    );
    interactive_spec.agents = 8;
    let interactive_plan = workload::plan(&interactive_spec, &salt);
    let burst_plan = workload::plan(
        &spec(
            &args,
            Kind::Unique,
            args.burst_requests,
            args.burst_max_tokens,
            args.seed + 1,
        ),
        &salt,
    );
    let bench = Arc::new(Bench {
        args,
        client,
        start: Instant::now(),
    });
    let args = &bench.args;

    if args.mode == Mode::Jobs
        && (args.max_in_flight.is_some() || args.max_queue_delay_ms.is_some())
    {
        let mut update = serde_json::Map::new();
        if let Some(max) = args.max_in_flight {
            update.insert("max_in_flight".into(), json!(max));
        }
        if let Some(ms) = args.max_queue_delay_ms {
            update.insert("max_queue_delay_ms".into(), json!(ms));
        }
        if let Some(priority) = args.job_priority {
            update.insert("priority".into(), json!(priority));
        }
        let response = bench
            .admin(
                bench
                    .client
                    .put(format!("{}/admin/jobs", args.gateway))
                    .json(&update),
            )
            .send()
            .await?;
        anyhow::ensure!(
            response.status().is_success(),
            "PUT /admin/jobs: {}",
            response.text().await?
        );
    }
    let runner: Value = bench
        .admin(bench.client.get(format!("{}/admin/jobs", args.gateway)))
        .send()
        .await?
        .json()
        .await
        .unwrap_or(Value::Null);

    let samples = Arc::new(Mutex::new(Vec::new()));
    let clients: Vec<_> = (0..args.interactive)
        .map(|client| {
            // Each client walks its own slice of the plan.
            let plan: Vec<Planned> = interactive_plan
                .iter()
                .skip(client)
                .step_by(args.interactive.max(1))
                .cloned()
                .collect();
            tokio::spawn(interactive_client(
                Arc::clone(&bench),
                plan,
                Arc::clone(&samples),
            ))
        })
        .collect();

    tokio::time::sleep(Duration::from_secs_f64(args.burst_at)).await;
    let burst_started_s = bench.at();
    let submit_ms = match args.mode {
        Mode::None => Vec::new(),
        Mode::Direct => {
            burst_direct(Arc::clone(&bench), burst_plan, Arc::clone(&samples)).await;
            Vec::new()
        }
        Mode::Jobs => burst_jobs(Arc::clone(&bench), burst_plan, Arc::clone(&samples)).await?,
    };
    futures::future::join_all(clients).await;

    let samples = samples.lock().clone();
    let burst: Vec<&Sample> = samples.iter().filter(|s| s.kind == "burst").collect();
    let burst_ok = burst.iter().filter(|s| s.ok).count();
    let burst_end_s = burst
        .iter()
        .map(|s| s.finished_s)
        .fold(burst_started_s, f64::max);
    let mut errors: std::collections::BTreeMap<String, usize> = Default::default();
    for sample in burst.iter().filter(|s| !s.ok) {
        *errors.entry(sample.status.to_string()).or_default() += 1;
    }
    let interactive_ttft = |from: f64, to: f64| -> Vec<f64> {
        samples
            .iter()
            .filter(|s| s.kind == "interactive" && s.started_s >= from && s.started_s < to)
            .filter_map(|s| s.ttft_ms)
            .collect()
    };
    let window_end = if args.mode == Mode::None {
        args.duration
    } else {
        burst_end_s.min(args.duration)
    };
    let before = interactive_ttft(0.0, burst_started_s);
    let during = interactive_ttft(burst_started_s, window_end.max(burst_started_s + 1.0));
    let interactive_all: Vec<&Sample> =
        samples.iter().filter(|s| s.kind == "interactive").collect();
    let interactive_failed = interactive_all.iter().filter(|s| !s.ok).count();
    let queue_waits: Vec<f64> = burst.iter().filter_map(|s| s.ttft_ms).collect();
    let summary = json!({
        "tag": args.tag,
        "mode": args.mode,
        "duration_s": args.duration,
        "interactive_clients": args.interactive,
        "burst": {
            "requests": args.burst_requests,
            "max_tokens": args.burst_max_tokens,
            "at_s": round(Some(burst_started_s)),
            "ok": burst_ok,
            "errors_by_status": errors,
            "makespan_s": round((!burst.is_empty()).then_some(burst_end_s - burst_started_s)),
            "submit_ms_p50": round(percentile(&submit_ms, 0.5)),
            "submit_ms_max": round(submit_ms.iter().copied().reduce(f64::max)),
            // jobs: time from submission to start (queue wait); direct: TTFT.
            "first_token_or_start_ms_p50": round(percentile(&queue_waits, 0.5)),
            "first_token_or_start_ms_p99": round(percentile(&queue_waits, 0.99)),
        },
        "interactive": {
            "requests": interactive_all.len(),
            "failed": interactive_failed,
            "before_burst": {
                "n": before.len(),
                "ttft_p50_ms": round(percentile(&before, 0.5)),
                "ttft_p99_ms": round(percentile(&before, 0.99)),
            },
            "during_burst": {
                "window_s": [round(Some(burst_started_s)), round(Some(window_end))],
                "n": during.len(),
                "ttft_p50_ms": round(percentile(&during, 0.5)),
                "ttft_p90_ms": round(percentile(&during, 0.9)),
                "ttft_p99_ms": round(percentile(&during, 0.99)),
                "ttft_max_ms": round(during.iter().copied().reduce(f64::max)),
            },
        },
        "runner": runner.get("runner").cloned().unwrap_or(Value::Null),
        "queue": runner.pointer("/queue/backend").cloned().unwrap_or(Value::Null),
        "timeline": timeline(&samples, args.duration),
    });
    std::fs::create_dir_all(&args.out)?;
    std::fs::write(
        args.out.join("summary.json"),
        serde_json::to_string_pretty(&summary)?,
    )?;
    let mut lines = String::new();
    for sample in &samples {
        lines.push_str(&serde_json::to_string(sample)?);
        lines.push('\n');
    }
    std::fs::write(args.out.join("samples.jsonl"), lines)?;
    let mut brief = summary.clone();
    if let Some(object) = brief.as_object_mut() {
        object.remove("timeline");
    }
    println!("{}", serde_json::to_string_pretty(&brief)?);
    Ok(())
}
