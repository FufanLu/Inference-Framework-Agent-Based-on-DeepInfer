//! Leviathan gateway: OpenAI Chat Completions / Responses and Anthropic
//! Messages endpoints in front of token-in/token-out inference workers.

use axum::{Router, routing::get};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();
    let app = Router::new().route("/healthz", get(|| async { "ok" }));
    let addr = std::env::var("LEV_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".into());
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!(%addr, "lev-gateway listening");
    axum::serve(listener, app).await?;
    Ok(())
}
