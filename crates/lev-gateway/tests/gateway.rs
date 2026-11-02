//! End-to-end gateway tests against in-process mock workers.
//!
//! The mock workers speak the real gRPC protocol and stream token IDs from the
//! real DeepSeek V4.1 tokenizer, so these tests exercise recipe's conversion,
//! rendering, tokenization and output parsing exactly as production does.

use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use futures::Stream;
use http_body_util::BodyExt;
use lev_gateway::config::{Config, WorkerConfig};
use lev_gateway::state::AppState;
use lev_proto::pb;
use lev_proto::pb::worker_server::{Worker, WorkerServer};
use serde_json::{Value, json};
use tokenizers::Tokenizer;
use tokio::sync::broadcast;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::{BroadcastStream, TcpListenerStream};
use tonic::{Status, async_trait};
use tower::ServiceExt;

const MODEL: &str = "DeepSeek-V4.1-Flash";
const BLOCK: usize = 16;

fn tokenizer_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/tokenizers/v41/tokenizer.json")
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Behavior {
    Serve,
    /// Every Generate call fails with UNAVAILABLE.
    Down,
}

#[derive(Clone)]
struct MockWorker {
    id: String,
    role: pb::Role,
    behavior: Behavior,
    tokenizer: Arc<Tokenizer>,
    events: broadcast::Sender<pb::WorkerEvent>,
    served: Arc<AtomicUsize>,
    seq: Arc<AtomicUsize>,
}

impl MockWorker {
    /// The synthetic model output for a request.
    fn output(&self, request: &pb::GenerateRequest) -> String {
        let thinking = request.hints.get("thinking").map(String::as_str) == Some("true");
        if let Some(constraint) = &request.constraint {
            if constraint.kind == pb::output_constraint::Kind::ToolCall as i32
                && request.hints.get("last_role").map(String::as_str) != Some("tool")
            {
                let tools: Value = serde_json::from_str(&constraint.schema).unwrap();
                let name = tools[0]["function"]["name"].as_str().unwrap().to_string();
                let call = format!(
                    "\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"{name}\">\n<｜DSML｜ parameter name=\"city\" string=\"true\">Boston</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>"
                );
                return if thinking {
                    format!("I should call {name}.</think>{call}")
                } else {
                    call
                };
            }
            if constraint.kind == pb::output_constraint::Kind::Json as i32 {
                return r#"{"answer": 42}"#.to_string();
            }
        }
        let seed = request.sampling.as_ref().map_or(0, |s| s.seed);
        let answer = format!("Hello from {} (seed {seed}).", self.id);
        if thinking {
            format!("The user greets me.</think>{answer}")
        } else {
            answer
        }
    }

    fn publish_blocks(&self, prompt: &[u32]) {
        let hashes = lev_router::hashing::block_hashes(prompt, BLOCK);
        let seq = self.seq.fetch_add(1, Ordering::SeqCst) as u64;
        let _ = self.events.send(pb::WorkerEvent {
            event: Some(pb::worker_event::Event::Kv(pb::KvEventBatch {
                seq,
                events: vec![pb::KvEvent {
                    event: Some(pb::kv_event::Event::Stored(pb::BlockStored {
                        block_hashes: hashes,
                        parent_hash: None,
                        medium: pb::Medium::Gpu as i32,
                    })),
                }],
            })),
        });
    }
}

type BoxStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send>>;

fn message(event: pb::generate_response::Event) -> Result<pb::GenerateResponse, Status> {
    Ok(pb::GenerateResponse { event: Some(event) })
}

#[async_trait]
impl Worker for MockWorker {
    type GenerateStream = BoxStream<pb::GenerateResponse>;
    type SubscribeEventsStream = BoxStream<pb::WorkerEvent>;

    async fn generate(
        &self,
        request: tonic::Request<pb::GenerateRequest>,
    ) -> Result<tonic::Response<Self::GenerateStream>, Status> {
        if self.behavior == Behavior::Down {
            return Err(Status::unavailable("mock worker is down"));
        }
        let request = request.into_inner();
        if request.prompt_token_ids.is_empty() {
            return Err(Status::invalid_argument("empty prompt"));
        }
        self.served.fetch_add(1, Ordering::SeqCst);
        self.publish_blocks(&request.prompt_token_ids);
        let prompt_tokens = request.prompt_token_ids.len() as u32;
        use pb::generate_response::Event;
        if request.stage == pb::Role::Prefill as i32 {
            let items = vec![
                message(Event::Ready(pb::Ready {
                    prompt_tokens,
                    cached_tokens: 0,
                    queue_time_us: 0,
                    prefill_time_us: 100,
                    kv_transfer: Some(pb::KvTransfer {
                        source_worker_id: self.id.clone(),
                        handle: "mock-handle".into(),
                        num_tokens: prompt_tokens,
                    }),
                })),
                message(Event::Finished(pb::Finished {
                    reason: pb::FinishReason::Stop as i32,
                    ..Default::default()
                })),
            ];
            return Ok(tonic::Response::new(Box::pin(tokio_stream::iter(items))));
        }
        let transferred = request.kv_transfer.as_ref().map_or(0, |kv| kv.num_tokens);
        let text = self.output(&request);
        let ids = self
            .tokenizer
            .encode(text.as_str(), false)
            .unwrap()
            .get_ids()
            .to_vec();
        let max_tokens = request.sampling.as_ref().map_or(u32::MAX, |s| s.max_tokens) as usize;
        let truncated = ids.len() > max_tokens;
        let ids: Vec<u32> = ids.into_iter().take(max_tokens).collect();
        let mut items = vec![message(Event::Ready(pb::Ready {
            prompt_tokens,
            cached_tokens: transferred,
            queue_time_us: 0,
            prefill_time_us: 100,
            kv_transfer: None,
        }))];
        for chunk in ids.chunks(3) {
            items.push(message(Event::Tokens(pb::TokenBatch {
                token_ids: chunk.to_vec(),
                logprobs: vec![],
                draft_tokens: 2,
                accepted_tokens: 1,
            })));
        }
        items.push(message(Event::Finished(pb::Finished {
            reason: if truncated {
                pb::FinishReason::Length
            } else {
                pb::FinishReason::Stop
            } as i32,
            completion_tokens: ids.len() as u32,
            ..Default::default()
        })));
        Ok(tonic::Response::new(Box::pin(tokio_stream::iter(items))))
    }

    async fn abort(
        &self,
        _request: tonic::Request<pb::AbortRequest>,
    ) -> Result<tonic::Response<pb::AbortResponse>, Status> {
        Ok(tonic::Response::new(pb::AbortResponse { found: false }))
    }

    async fn subscribe_events(
        &self,
        _request: tonic::Request<pb::SubscribeEventsRequest>,
    ) -> Result<tonic::Response<Self::SubscribeEventsStream>, Status> {
        // A fresh subscription restarts the sequence.
        self.seq.store(0, Ordering::SeqCst);
        let stream =
            BroadcastStream::new(self.events.subscribe()).filter_map(|event| event.ok().map(Ok));
        Ok(tonic::Response::new(Box::pin(stream)))
    }

    async fn get_info(
        &self,
        _request: tonic::Request<pb::GetInfoRequest>,
    ) -> Result<tonic::Response<pb::WorkerInfo>, Status> {
        Ok(tonic::Response::new(pb::WorkerInfo {
            worker_id: self.id.clone(),
            backend: "mock".into(),
            model: "deepseek-ai/DeepSeek-V4.1-Flash-1017".into(),
            role: self.role as i32,
            block_size: BLOCK as u32,
            context_limit: 131_072,
            total_blocks: 10_000,
            supports_kv_events: true,
            supports_constraints: true,
            supports_logprobs: false,
            deterministic: true,
            labels: Default::default(),
        }))
    }

    async fn health(
        &self,
        _request: tonic::Request<pb::HealthRequest>,
    ) -> Result<tonic::Response<pb::HealthResponse>, Status> {
        Ok(tonic::Response::new(pb::HealthResponse {
            ready: true,
            status: "ok".into(),
        }))
    }
}

struct Cluster {
    app: axum::Router,
    state: AppState,
    served: Vec<Arc<AtomicUsize>>,
}

async fn start_worker(id: &str, role: pb::Role, behavior: Behavior) -> (String, Arc<AtomicUsize>) {
    let tokenizer = Arc::new(Tokenizer::from_file(tokenizer_path()).unwrap());
    let (events, _) = broadcast::channel(1024);
    let served = Arc::new(AtomicUsize::new(0));
    let worker = MockWorker {
        id: id.to_string(),
        role,
        behavior,
        tokenizer,
        events,
        served: Arc::clone(&served),
        seq: Arc::new(AtomicUsize::new(0)),
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(WorkerServer::new(worker))
            .serve_with_incoming(TcpListenerStream::new(listener))
            .await
            .unwrap();
    });
    (format!("http://{addr}"), served)
}

async fn cluster(
    workers: &[(&str, pb::Role, Behavior)],
    configure: impl FnOnce(&mut Config),
) -> Cluster {
    let mut endpoints = Vec::new();
    let mut served = Vec::new();
    for (id, role, behavior) in workers {
        let (endpoint, counter) = start_worker(id, *role, *behavior).await;
        endpoints.push(endpoint);
        served.push(counter);
    }
    let mut config = Config::default();
    config.models[0].tokenizer = tokenizer_path();
    config.workers = endpoints
        .into_iter()
        .map(|endpoint| WorkerConfig { endpoint })
        .collect();
    configure(&mut config);
    let state = AppState::new(config, None).await.unwrap();
    // Wait until every worker is registered and healthy.
    for _ in 0..200 {
        let snapshot = state.router.snapshot();
        if snapshot.workers.len() == workers.len() && snapshot.workers.iter().all(|w| w.healthy) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let app = lev_gateway::api::router(state.clone());
    Cluster { app, state, served }
}

async fn post(app: &axum::Router, path: &str, body: Value) -> (StatusCode, String) {
    let response = app
        .clone()
        .oneshot(
            Request::post(path)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

async fn get(app: &axum::Router, path: &str) -> (StatusCode, Value) {
    let response = app
        .clone()
        .oneshot(Request::get(path).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

fn one_worker() -> [(&'static str, pb::Role, Behavior); 1] {
    [("w0", pb::Role::Unified, Behavior::Serve)]
}

#[tokio::test]
async fn chat_completion_round_trip() {
    let cluster = cluster(&one_worker(), |_| {}).await;
    let (status, body) = post(
        &cluster.app,
        "/v1/chat/completions",
        json!({
            "model": "deepseek-flash",
            "messages": [{"role": "user", "content": "Hi"}],
            "thinking": {"type": "disabled"},
            "seed": 7,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let body: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["model"], MODEL);
    assert_eq!(
        body["choices"][0]["message"]["content"],
        "Hello from w0 (seed 7)."
    );
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert!(body["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
    assert!(body["id"].as_str().unwrap().starts_with("chatcmpl-"));
}

#[tokio::test]
async fn thinking_mode_separates_reasoning() {
    let cluster = cluster(&one_worker(), |_| {}).await;
    let (status, body) = post(
        &cluster.app,
        "/v1/chat/completions",
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hi"}],
            "thinking": {"type": "enabled"},
            "seed": 1,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let body: Value = serde_json::from_str(&body).unwrap();
    let message = &body["choices"][0]["message"];
    assert_eq!(message["reasoning_content"], "The user greets me.");
    assert_eq!(message["content"], "Hello from w0 (seed 1).");
}

#[tokio::test]
async fn streaming_ends_with_done() {
    let cluster = cluster(&one_worker(), |_| {}).await;
    let (status, body) = post(
        &cluster.app,
        "/v1/chat/completions",
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hi"}],
            "thinking": {"type": "disabled"},
            "stream": true,
            "stream_options": {"include_usage": true},
            "seed": 3,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let frames: Vec<&str> = body
        .split("\n\n")
        .filter(|frame| !frame.is_empty())
        .collect();
    assert_eq!(frames.last().copied(), Some("data: [DONE]"));
    let content: String = frames
        .iter()
        .filter_map(|frame| frame.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .filter_map(|data| serde_json::from_str::<Value>(data).ok())
        .filter_map(|chunk| {
            chunk["choices"][0]["delta"]["content"]
                .as_str()
                .map(str::to_string)
        })
        .collect();
    assert_eq!(content, "Hello from w0 (seed 3).");
}

#[tokio::test]
async fn tool_calls_are_parsed_from_dsml() {
    let cluster = cluster(&one_worker(), |_| {}).await;
    let (status, body) = post(
        &cluster.app,
        "/v1/chat/completions",
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "Weather in Boston?"}],
            "thinking": {"type": "disabled"},
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let body: Value = serde_json::from_str(&body).unwrap();
    let choice = &body["choices"][0];
    assert_eq!(choice["finish_reason"], "tool_calls");
    let call = &choice["message"]["tool_calls"][0];
    assert_eq!(call["function"]["name"], "get_weather");
    let arguments: Value =
        serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(arguments, json!({"city": "Boston"}));
}

#[tokio::test]
async fn anthropic_messages_round_trip() {
    let cluster = cluster(&one_worker(), |_| {}).await;
    let (status, body) = post(
        &cluster.app,
        "/v1/messages",
        json!({
            "model": MODEL,
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "Hi"}],
            "thinking": {"type": "disabled"},
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let body: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(body["type"], "message");
    assert_eq!(body["role"], "assistant");
    assert!(body["id"].as_str().unwrap().starts_with("msg_"));
    let text = body["content"][0]["text"].as_str().unwrap();
    assert!(text.starts_with("Hello from w0"), "{text}");
    assert_eq!(body["stop_reason"], "end_turn");
}

#[tokio::test]
async fn responses_api_round_trip() {
    let cluster = cluster(&one_worker(), |_| {}).await;
    let (status, body) = post(
        &cluster.app,
        "/v1/responses",
        json!({
            "model": MODEL,
            "input": "Hi",
            "reasoning": {"effort": "low"},
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let body: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(body["object"], "response");
    assert!(body["id"].as_str().unwrap().starts_with("resp_"));
}

#[tokio::test]
async fn max_tokens_truncates_with_length() {
    let cluster = cluster(&one_worker(), |_| {}).await;
    let (status, body) = post(
        &cluster.app,
        "/v1/chat/completions",
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hi"}],
            "thinking": {"type": "disabled"},
            "max_tokens": 2,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let body: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(body["choices"][0]["finish_reason"], "length");
    assert_eq!(body["usage"]["completion_tokens"], 2);
}

#[tokio::test]
async fn unknown_model_is_404_in_protocol_format() {
    let cluster = cluster(&one_worker(), |_| {}).await;
    let (status, body) = post(
        &cluster.app,
        "/v1/messages",
        json!({"model": "gpt-4o", "max_tokens": 8, "messages": [{"role": "user", "content": "x"}]}),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let body: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(body["type"], "error");
    assert_eq!(body["error"]["type"], "not_found_error");
}

#[tokio::test]
async fn context_overflow_is_rejected_before_dispatch() {
    let cluster = cluster(&one_worker(), |config| config.models[0].context_limit = 64).await;
    let (status, body) = post(
        &cluster.app,
        "/v1/chat/completions",
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "word ".repeat(200)}],
            "max_tokens": 16,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let body: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(body["error"]["code"], "context_length_exceeded");
    assert_eq!(cluster.served[0].load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn failover_skips_a_dead_worker() {
    let cluster = cluster(
        &[
            ("down", pb::Role::Unified, Behavior::Down),
            ("up", pb::Role::Unified, Behavior::Serve),
        ],
        |config| config.router.policy = "round_robin".into(),
    )
    .await;
    for _ in 0..4 {
        let (status, body) = post(
            &cluster.app,
            "/v1/chat/completions",
            json!({
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "thinking": {"type": "disabled"},
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(body.contains("Hello from up"), "{body}");
    }
    assert_eq!(cluster.served[1].load(Ordering::SeqCst), 4);
}

#[tokio::test]
async fn no_workers_is_503() {
    let cluster = cluster(&[], |_| {}).await;
    let (status, body) = post(
        &cluster.app,
        "/v1/chat/completions",
        json!({"model": MODEL, "messages": [{"role": "user", "content": "Hi"}]}),
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
}

#[tokio::test]
async fn disaggregated_requests_hand_kv_to_decode() {
    let cluster = cluster(
        &[
            ("prefill-0", pb::Role::Prefill, Behavior::Serve),
            ("decode-0", pb::Role::Decode, Behavior::Serve),
        ],
        |_| {},
    )
    .await;
    let (status, body) = post(
        &cluster.app,
        "/v1/chat/completions",
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hi"}],
            "thinking": {"type": "disabled"},
            "seed": 5,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(body.contains("Hello from decode-0"), "{body}");
    assert_eq!(cluster.served[0].load(Ordering::SeqCst), 1);
    assert_eq!(cluster.served[1].load(Ordering::SeqCst), 1);
    let (_, requests) = get(&cluster.app, "/admin/requests").await;
    let entry = &requests["requests"][0];
    assert_eq!(entry["prefill_worker_id"], "prefill-0");
    assert_eq!(entry["worker_id"], "decode-0");
    assert_eq!(entry["plan"]["kind"], "disaggregated");
}

#[tokio::test]
async fn cache_aware_routing_follows_kv_events() {
    let cluster = cluster(
        &[
            ("a", pb::Role::Unified, Behavior::Serve),
            ("b", pb::Role::Unified, Behavior::Serve),
        ],
        |config| config.router.policy = "cache_aware".into(),
    )
    .await;
    let system = "You are a meticulous assistant. ".repeat(40);
    let body = |question: &str| {
        json!({
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
            "thinking": {"type": "disabled"},
        })
    };
    let (_, first) = post(&cluster.app, "/v1/chat/completions", body("first")).await;
    let first_worker = if first.contains("Hello from a") {
        "a"
    } else {
        "b"
    };
    // Give the KV event a moment to reach the router.
    tokio::time::sleep(Duration::from_millis(100)).await;
    for question in ["second", "third", "fourth"] {
        let (_, response) = post(&cluster.app, "/v1/chat/completions", body(question)).await;
        assert!(
            response.contains(&format!("Hello from {first_worker}")),
            "expected {first_worker}: {response}"
        );
    }
    let (_, requests) = get(&cluster.app, "/admin/requests?limit=1").await;
    assert!(requests["requests"][0]["cached_tokens"].as_u64().is_some());
    let snapshot = cluster.state.router.snapshot();
    let cached: usize = snapshot.workers.iter().map(|w| w.gpu_blocks).sum();
    assert!(cached > 0);
}

#[tokio::test]
async fn prompt_inspector_shows_the_rendered_prompt() {
    let cluster = cluster(&one_worker(), |_| {}).await;
    let (status, body) = post(
        &cluster.app,
        "/v1/leviathan/render",
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hello"}],
            "thinking": {"type": "disabled"},
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let body: Value = serde_json::from_str(&body).unwrap();
    let prompt = body["prompt"].as_str().unwrap();
    assert!(prompt.starts_with("<｜begin▁of▁sentence｜>"), "{prompt}");
    assert!(prompt.ends_with("<｜Assistant｜></think>"), "{prompt}");
    let tokens = body["tokens"].as_array().unwrap();
    assert_eq!(tokens[0]["id"], 0);
    assert_eq!(tokens[0]["special"], true);
    assert_eq!(tokens.last().unwrap()["text"], "</think>");
}

#[tokio::test]
async fn replay_reproduces_the_output() {
    let cluster = cluster(&one_worker(), |_| {}).await;
    let response = cluster
        .app
        .clone()
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": MODEL,
                        "messages": [{"role": "user", "content": "Hi"}],
                        "thinking": {"type": "disabled"},
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    let request_id = response.headers()["x-request-id"]
        .to_str()
        .unwrap()
        .to_string();
    let _ = response.into_body().collect().await.unwrap();
    let (status, replay) = post(
        &cluster.app,
        &format!("/admin/requests/{request_id}/replay"),
        json!({}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{replay}");
    let replay: Value = serde_json::from_str(&replay).unwrap();
    assert_eq!(replay["identical"], true, "{replay}");
    assert!(replay["original_tokens"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn router_policy_can_change_at_runtime() {
    let cluster = cluster(&one_worker(), |_| {}).await;
    let response = cluster
        .app
        .clone()
        .oneshot(
            Request::put("/admin/router")
                .header("content-type", "application/json")
                .body(Body::from(json!({"policy": "p2c"}).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        cluster.state.router.snapshot().policy,
        lev_router::Policy::PowerOfTwo
    );
}

#[tokio::test]
async fn api_keys_and_rate_limits_are_enforced() {
    let cluster = cluster(&one_worker(), |config| {
        config.auth.api_keys = vec![lev_gateway::config::ApiKeyConfig {
            key: "lev-test-key".into(),
            tenant: "acme".into(),
            name: None,
            limits: lev_gateway::config::LimitsConfig {
                rpm: Some(2),
                ..Default::default()
            },
        }];
    })
    .await;
    let body = json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hi"}],
        "thinking": {"type": "disabled"},
    });
    let send = |key: Option<&'static str>| {
        let app = cluster.app.clone();
        let body = body.clone();
        async move {
            let mut request =
                Request::post("/v1/chat/completions").header("content-type", "application/json");
            if let Some(key) = key {
                request = request.header("authorization", format!("Bearer {key}"));
            }
            app.oneshot(request.body(Body::from(body.to_string())).unwrap())
                .await
                .unwrap()
        }
    };
    assert_eq!(send(None).await.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(
        send(Some("lev-wrong")).await.status(),
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(send(Some("lev-test-key")).await.status(), StatusCode::OK);
    assert_eq!(send(Some("lev-test-key")).await.status(), StatusCode::OK);
    let throttled = send(Some("lev-test-key")).await;
    assert_eq!(throttled.status(), StatusCode::TOO_MANY_REQUESTS);
    let retry_after: u64 = throttled.headers()["retry-after"]
        .to_str()
        .unwrap()
        .parse()
        .unwrap();
    assert!((1..=30).contains(&retry_after), "retry-after {retry_after}");
    let bytes = throttled.into_body().collect().await.unwrap().to_bytes();
    let error: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(error["error"]["code"], "rate_limit_exceeded");
}

#[tokio::test]
async fn tenants_keys_and_usage_through_the_admin_api() {
    let cluster = cluster(&one_worker(), |config| {
        config.auth.allow_anonymous = Some(false)
    })
    .await;
    let (status, _) = post(
        &cluster.app,
        "/admin/tenants",
        json!({"id": "globex", "name": "Globex", "tpm": 100000}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, created) = post(
        &cluster.app,
        "/admin/tenants/globex/keys",
        json!({"name": "ci"}),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{created}");
    let created: Value = serde_json::from_str(&created).unwrap();
    let key = created["key"].as_str().unwrap().to_string();
    assert!(key.starts_with("lev-"));

    let response = cluster
        .app
        .clone()
        .oneshot(
            Request::post("/v1/messages")
                .header("content-type", "application/json")
                .header("x-api-key", &key)
                .body(Body::from(
                    json!({"model": MODEL, "max_tokens": 64, "messages": [{"role": "user", "content": "Hi"}]})
                        .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let _ = response.into_body().collect().await.unwrap();

    // The ledger is written asynchronously.
    let mut usage = Value::Null;
    for _ in 0..40 {
        tokio::time::sleep(Duration::from_millis(50)).await;
        let (_, value) = get(&cluster.app, "/admin/usage?tenant=globex").await;
        if value["usage"]
            .as_array()
            .is_some_and(|rows| !rows.is_empty())
        {
            usage = value;
            break;
        }
    }
    let row = &usage["usage"][0];
    assert_eq!(row["tenant_id"], "globex");
    assert_eq!(row["requests"], 1);
    assert!(row["prompt_tokens"].as_u64().unwrap() > 0);
    let (_, tenants) = get(&cluster.app, "/admin/tenants").await;
    assert!(
        tenants["tenants"]
            .as_array()
            .unwrap()
            .iter()
            .any(|t| t["tenant"]["id"] == "globex")
    );
}

#[tokio::test]
async fn audit_records_a_stage_timeline() {
    let cluster = cluster(&one_worker(), |_| {}).await;
    let (status, _) = post(
        &cluster.app,
        "/v1/chat/completions",
        json!({"model": MODEL, "messages": [{"role": "user", "content": "Hi"}], "thinking": {"type": "disabled"}}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (_, requests) = get(&cluster.app, "/admin/requests?limit=1").await;
    let names: Vec<&str> = requests["requests"][0]["timeline"]
        .as_array()
        .unwrap()
        .iter()
        .map(|stage| stage["name"].as_str().unwrap())
        .collect();
    for expected in [
        "auth",
        "parse",
        "convert",
        "render",
        "tokenize",
        "dispatch",
        "worker.prefill",
        "decode",
    ] {
        assert!(names.contains(&expected), "missing {expected} in {names:?}");
    }
}
