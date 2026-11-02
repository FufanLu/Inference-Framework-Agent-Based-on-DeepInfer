//! Gateway configuration: a TOML file, overridable from the command line.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::Context;
use lev_router::{IndexMode, Policy, RouterConfig, ShadowConfig};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default = "default_listen")]
    pub listen: String,
    #[serde(default = "default_models")]
    pub models: Vec<ModelConfig>,
    #[serde(default)]
    pub workers: Vec<WorkerConfig>,
    #[serde(default)]
    pub router: RouterSection,
    #[serde(default)]
    pub auth: AuthConfig,
    #[serde(default)]
    pub control: ControlConfig,
    #[serde(default)]
    pub dispatch: DispatchConfig,
    #[serde(default)]
    pub audit: AuditConfig,
}

fn default_listen() -> String {
    "127.0.0.1:8080".to_string()
}

/// A model the gateway serves.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    /// Canonical name, reported in responses and matched against workers.
    pub name: String,
    /// Other names clients may use; workers may report any of them too.
    #[serde(default)]
    pub aliases: Vec<String>,
    /// Prompt encoding: `v4.1` or `v4`.
    #[serde(default = "default_version")]
    pub version: String,
    /// `tokenizer.json` of the model.
    pub tokenizer: PathBuf,
    /// Context length of the deployment (prompt + output tokens).
    #[serde(default = "default_context_limit")]
    pub context_limit: u32,
    /// Thinking mode for requests that do not choose one.
    #[serde(default)]
    pub default_thinking: Option<bool>,
}

fn default_version() -> String {
    "v4.1".to_string()
}

fn default_context_limit() -> u32 {
    131_072
}

fn default_models() -> Vec<ModelConfig> {
    vec![ModelConfig {
        name: "DeepSeek-V4.1-Flash".to_string(),
        aliases: vec![
            "deepseek-flash".to_string(),
            "deepseek-ai/DeepSeek-V4.1-Flash".to_string(),
            "deepseek-ai/DeepSeek-V4.1-Flash-1017".to_string(),
        ],
        version: default_version(),
        tokenizer: PathBuf::from("assets/tokenizers/v41/tokenizer.json"),
        context_limit: default_context_limit(),
        default_thinking: None,
    }]
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerConfig {
    /// gRPC endpoint, e.g. `http://127.0.0.1:50051`.
    pub endpoint: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct RouterSection {
    pub policy: String,
    pub index_mode: String,
    pub disaggregation: bool,
    pub session_affinity: bool,
    pub shadow_ttl_s: u64,
    pub default_prefill_us_per_token: f64,
    pub decode_interference_us: f64,
    pub cpu_hit_cost_ratio: f64,
}

impl Default for RouterSection {
    fn default() -> Self {
        let defaults = RouterConfig::default();
        Self {
            policy: defaults.policy.as_str().to_string(),
            index_mode: "exact".to_string(),
            disaggregation: defaults.disaggregation,
            session_affinity: defaults.session_affinity,
            shadow_ttl_s: defaults.shadow.ttl.as_secs(),
            default_prefill_us_per_token: defaults.default_prefill_us_per_token,
            decode_interference_us: defaults.decode_interference_us,
            cpu_hit_cost_ratio: defaults.cpu_hit_cost_ratio,
        }
    }
}

impl RouterSection {
    pub fn to_router_config(&self) -> anyhow::Result<RouterConfig> {
        let policy: Policy = self.policy.parse().map_err(anyhow::Error::msg)?;
        let index_mode: IndexMode = self.index_mode.parse().map_err(anyhow::Error::msg)?;
        Ok(RouterConfig {
            policy,
            index_mode,
            disaggregation: self.disaggregation,
            session_affinity: self.session_affinity,
            shadow: ShadowConfig {
                capacity_blocks: None,
                ttl: Duration::from_secs(self.shadow_ttl_s),
            },
            default_prefill_us_per_token: self.default_prefill_us_per_token,
            decode_interference_us: self.decode_interference_us,
            cpu_hit_cost_ratio: self.cpu_hit_cost_ratio,
            ..RouterConfig::default()
        })
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct AuthConfig {
    /// Static API keys, provisioned into the control plane at startup.
    pub api_keys: Vec<ApiKeyConfig>,
    /// Accept requests without a valid key as tenant `anonymous`. Defaults to
    /// true only when no static keys are configured.
    pub allow_anonymous: Option<bool>,
    /// Limits of the anonymous tenant.
    pub anonymous_limits: LimitsConfig,
    /// Bearer token for `/admin` endpoints. Without one, admin endpoints are
    /// open (local development only).
    pub admin_token: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApiKeyConfig {
    pub key: String,
    pub tenant: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default, flatten)]
    pub limits: LimitsConfig,
}

#[derive(Debug, Clone, Copy, Default, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct LimitsConfig {
    pub rpm: Option<u64>,
    pub tpm: Option<u64>,
    pub monthly_token_quota: Option<u64>,
}

impl From<LimitsConfig> for lev_control::Limits {
    fn from(limits: LimitsConfig) -> Self {
        Self {
            rpm: limits.rpm,
            tpm: limits.tpm,
            monthly_token_quota: limits.monthly_token_quota,
        }
    }
}

/// Where control-plane state lives.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ControlConfig {
    /// `memory`, or a `postgres://` URL.
    pub store: String,
    /// `redis://` URL for rate limits shared by all gateway replicas;
    /// without one, limits are enforced per gateway process.
    pub redis: Option<String>,
}

impl Default for ControlConfig {
    fn default() -> Self {
        Self {
            store: "memory".to_string(),
            redis: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct DispatchConfig {
    /// Workers tried before a request fails, when earlier ones are unavailable.
    pub max_attempts: u32,
    /// Time allowed for a worker to accept a request and finish its prefill.
    pub ready_timeout_ms: u64,
}

impl Default for DispatchConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            ready_timeout_ms: 120_000,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct AuditConfig {
    /// Requests kept in memory for inspection and replay.
    pub capacity: usize,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self { capacity: 2_000 }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            listen: default_listen(),
            models: default_models(),
            workers: Vec::new(),
            router: RouterSection::default(),
            auth: AuthConfig::default(),
            control: ControlConfig::default(),
            dispatch: DispatchConfig::default(),
            audit: AuditConfig::default(),
        }
    }
}

impl Config {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config {}", path.display()))?;
        let mut config: Config =
            toml::from_str(&text).with_context(|| format!("parsing config {}", path.display()))?;
        // Relative tokenizer paths are relative to the config file.
        if let Some(base) = path.parent() {
            for model in &mut config.models {
                if model.tokenizer.is_relative() {
                    let candidate = base.join(&model.tokenizer);
                    if candidate.exists() {
                        model.tokenizer = candidate;
                    }
                }
            }
        }
        Ok(config)
    }
}
