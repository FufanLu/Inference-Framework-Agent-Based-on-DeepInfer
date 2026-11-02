use std::path::PathBuf;

use clap::Parser;
use lev_gateway::config::{Config, WorkerConfig};
use lev_gateway::state::AppState;
use tracing_subscriber::EnvFilter;

/// Leviathan gateway.
#[derive(Debug, Parser)]
#[command(version, about)]
struct Args {
    /// TOML configuration file.
    #[arg(long, env = "LEV_CONFIG")]
    config: Option<PathBuf>,
    /// Listen address, overriding the configuration.
    #[arg(long, env = "LEV_LISTEN")]
    listen: Option<String>,
    /// Worker gRPC endpoint (repeatable), added to the configured ones.
    #[arg(long = "worker", env = "LEV_WORKERS", value_delimiter = ',')]
    workers: Vec<String>,
    /// Routing policy: round_robin, least_load, p2c or cache_aware.
    #[arg(long, env = "LEV_POLICY")]
    policy: Option<String>,
    /// Prefix index mode: exact, approximate or off.
    #[arg(long, env = "LEV_INDEX_MODE")]
    index_mode: Option<String>,
    /// Log as JSON lines.
    #[arg(long, env = "LEV_LOG_JSON")]
    log_json: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    if args.log_json {
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .json()
            .init();
    } else {
        tracing_subscriber::fmt().with_env_filter(filter).init();
    }

    let mut config = match &args.config {
        Some(path) => Config::load(path)?,
        None => Config::default(),
    };
    if let Some(listen) = args.listen {
        config.listen = listen;
    }
    config.workers.extend(
        args.workers
            .into_iter()
            .filter(|endpoint| !endpoint.is_empty())
            .map(|endpoint| WorkerConfig { endpoint }),
    );
    if let Some(policy) = args.policy {
        config.router.policy = policy;
    }
    if let Some(mode) = args.index_mode {
        config.router.index_mode = mode;
    }

    let metrics = lev_gateway::metrics::install()?;
    let listen = config.listen.clone();
    let state = AppState::new(config, Some(metrics))?;
    let app = lev_gateway::api::router(state);
    let listener = tokio::net::TcpListener::bind(&listen).await?;
    tracing::info!(%listen, "lev-gateway listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut signal) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            signal.recv().await;
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }
    tracing::info!("shutting down");
}
