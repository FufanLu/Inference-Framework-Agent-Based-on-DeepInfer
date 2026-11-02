//! Leviathan gateway: OpenAI Chat Completions / Responses and Anthropic
//! Messages endpoints in front of token-in/token-out inference workers.
//!
//! deepseek-recipe converts each protocol request into a conversation,
//! renders the DeepSeek V4.1 prompt and parses the model output; this crate
//! supplies what recipe leaves to its caller: HTTP transport, authentication,
//! routing to workers, failover, metering and the audit log.

pub mod api;
pub mod audit;
pub mod config;
pub mod error;
pub mod events;
pub mod metrics;
pub mod model;
pub mod pipeline;
pub mod protocol;
pub mod state;
pub mod workers;
