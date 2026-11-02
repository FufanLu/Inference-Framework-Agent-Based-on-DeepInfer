//! Synthetic workloads. Every request is a Chat Completions body; prompts are
//! made of deterministic pseudo-text so that runs are reproducible.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde_json::{Value, json};

/// A request and what the harness needs to know about it.
#[derive(Debug, Clone)]
pub struct Planned {
    pub body: Value,
    pub session: Option<String>,
    /// Workload-specific group, e.g. the agent or session index.
    pub group: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Kind {
    /// Agents with long shared system prompts and tool schemas; each request
    /// is one agent's prompt plus a unique question. Agent popularity is
    /// skewed (Zipf-like), so a few prefixes are hot.
    AgentSharedPrefix,
    /// Sessions whose prompts grow turn by turn; each turn shares the whole
    /// previous conversation as a prefix.
    MultiTurnChat,
    /// Unique prompts with no shared prefix: the control group.
    Unique,
}

#[derive(Debug, Clone)]
pub struct Spec {
    pub kind: Kind,
    pub requests: usize,
    pub model: String,
    pub max_tokens: u32,
    pub agents: usize,
    pub system_words: usize,
    pub question_words: usize,
    pub sessions: usize,
    pub thinking: bool,
    pub seed: u64,
}

const WORDS: &[&str] = &[
    "inference",
    "cache",
    "prefix",
    "router",
    "decode",
    "prefill",
    "token",
    "latency",
    "throughput",
    "replica",
    "kernel",
    "attention",
    "expert",
    "batch",
    "schedule",
    "memory",
    "bandwidth",
    "tensor",
    "parallel",
    "stream",
    "gateway",
    "tenant",
    "quota",
    "policy",
    "deploy",
    "cluster",
    "metric",
    "trace",
    "replay",
    "budget",
    "window",
    "context",
    "模型",
    "推理",
    "缓存",
    "路由",
    "调度",
    "延迟",
    "吞吐",
    "显存",
    "专家",
    "并行",
];

fn words(rng: &mut StdRng, n: usize) -> String {
    let mut out = String::with_capacity(n * 8);
    for i in 0..n {
        if i > 0 {
            out.push(' ');
        }
        out.push_str(WORDS[rng.random_range(0..WORDS.len())]);
        if rng.random_ratio(1, 12) {
            out.push('.');
        }
    }
    out
}

/// Zipf-like choice over `n` items with exponent 1.
fn zipf(rng: &mut StdRng, n: usize) -> usize {
    let harmonic: f64 = (1..=n).map(|k| 1.0 / k as f64).sum();
    let mut target = rng.random::<f64>() * harmonic;
    for k in 1..=n {
        target -= 1.0 / k as f64;
        if target <= 0.0 {
            return k - 1;
        }
    }
    n - 1
}

fn tool() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search the internal knowledge base and return the most relevant passages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {"type": "integer", "description": "Passages to return"},
                },
                "required": ["query"],
            },
        },
    })
}

/// Plan every request of a run. `salt` makes each run's prefixes distinct so
/// that one run cannot reuse another run's cache.
pub fn plan(spec: &Spec, salt: &str) -> Vec<Planned> {
    let mut rng = StdRng::seed_from_u64(spec.seed);
    let thinking = if spec.thinking { "enabled" } else { "disabled" };
    match spec.kind {
        Kind::AgentSharedPrefix => {
            let systems: Vec<String> = (0..spec.agents)
                .map(|agent| {
                    format!(
                        "[{salt}] You are agent #{agent} of the Leviathan operations team. {}",
                        words(&mut rng, spec.system_words)
                    )
                })
                .collect();
            (0..spec.requests)
                .map(|i| {
                    let agent = zipf(&mut rng, spec.agents);
                    let question = format!("Question {i}: {}", words(&mut rng, spec.question_words));
                    Planned {
                        body: json!({
                            "model": spec.model,
                            "messages": [
                                {"role": "system", "content": systems[agent]},
                                {"role": "user", "content": question},
                            ],
                            "tools": [tool()],
                            "tool_choice": "none",
                            "thinking": {"type": thinking},
                            "max_tokens": spec.max_tokens,
                            "stream": true,
                            "stream_options": {"include_usage": true},
                        }),
                        session: None,
                        group: agent,
                    }
                })
                .collect()
        }
        Kind::MultiTurnChat => {
            let sessions = spec.sessions.max(1);
            let mut histories: Vec<Vec<Value>> = (0..sessions)
                .map(|session| {
                    vec![json!({
                        "role": "system",
                        "content": format!("[{salt}] Session {session}. {}", words(&mut rng, spec.system_words / 4)),
                    })]
                })
                .collect();
            (0..spec.requests)
                .map(|i| {
                    let session = i % sessions;
                    let history = &mut histories[session];
                    history.push(json!({
                        "role": "user",
                        "content": words(&mut rng, spec.question_words),
                    }));
                    let body = json!({
                        "model": spec.model,
                        "messages": history.clone(),
                        "thinking": {"type": thinking},
                        "max_tokens": spec.max_tokens,
                        "stream": true,
                        "stream_options": {"include_usage": true},
                    });
                    // The next turn sees a fixed stand-in for this answer, so
                    // the planned conversation does not depend on outputs.
                    history.push(json!({
                        "role": "assistant",
                        "content": words(&mut rng, spec.question_words),
                    }));
                    Planned {
                        body,
                        session: Some(format!("{salt}-session-{session}")),
                        group: session,
                    }
                })
                .collect()
        }
        Kind::Unique => (0..spec.requests)
            .map(|i| Planned {
                body: json!({
                    "model": spec.model,
                    "messages": [{
                        "role": "user",
                        "content": format!("[{salt}-{i}] {}", words(&mut rng, spec.system_words + spec.question_words)),
                    }],
                    "thinking": {"type": thinking},
                    "max_tokens": spec.max_tokens,
                    "stream": true,
                    "stream_options": {"include_usage": true},
                }),
                session: None,
                group: i,
            })
            .collect(),
    }
}
