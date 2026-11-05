//! Served models: prompt encoding, tokenizer and conversion options per model.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, bail};
use deepseek_recipe::request::{ConversionOptions, ModelVersion};
use deepseek_recipe_core::conversation::Conversation;
use deepseek_recipe_encoding::dsv4::DeepseekV4Encoding;
use deepseek_recipe_encoding::dsv41::DeepseekV41Encoding;
use deepseek_recipe_encoding::{PromptEncoding, PromptTokenizer, RenderedPrompt, TokenizerEncoder};
use tokenizers::Tokenizer;

use crate::config::ModelConfig;

enum Encoding {
    V4(DeepseekV4Encoding),
    V41(DeepseekV41Encoding),
}

/// One model the gateway serves.
pub struct ServedModel {
    /// Canonical name.
    pub name: String,
    pub aliases: Vec<String>,
    pub version: ModelVersion,
    pub context_limit: u32,
    /// Shared by every request's stream processor.
    pub tokenizer: Arc<Tokenizer>,
    pub eos_token_id: u32,
    /// Decode to `max_tokens` regardless of EOS (benchmark deployments).
    pub ignore_eos: bool,
    /// Conversion options that accept this model's names.
    pub options: ConversionOptions,
    prompt_tokenizer: PromptTokenizer,
    encoding: Encoding,
}

impl std::fmt::Debug for ServedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServedModel")
            .field("name", &self.name)
            .field("version", &self.version)
            .field("context_limit", &self.context_limit)
            .finish_non_exhaustive()
    }
}

impl ServedModel {
    pub fn from_config(config: &ModelConfig) -> anyhow::Result<Self> {
        let version = match config.version.as_str() {
            "v4.1" | "v41" => ModelVersion::DeepseekV41Flash,
            "v4" => ModelVersion::DeepseekV4Pro,
            other => bail!("model {}: unknown version `{other}`", config.name),
        };
        let tokenizer = Tokenizer::from_file(&config.tokenizer)
            .map_err(|error| anyhow::anyhow!("{error}"))
            .with_context(|| format!("loading tokenizer {}", config.tokenizer.display()))?;
        let eos_token_id = tokenizer
            .token_to_id("<｜end▁of▁sentence｜>")
            .context("tokenizer has no end-of-sentence token")?;
        let prompt_tokenizer = PromptTokenizer::new(&tokenizer);
        let encoding = match version {
            ModelVersion::DeepseekV4Pro => Encoding::V4(DeepseekV4Encoding::new()),
            _ => Encoding::V41(DeepseekV41Encoding::new()),
        };
        let names = std::iter::once(config.name.clone()).chain(config.aliases.iter().cloned());
        let mut options = ConversionOptions::default()
            .with_model_versions_by_name(names.map(|name| (name, version)));
        if let Some(thinking) = config.default_thinking {
            options = options.with_default_thinking_mode(thinking);
        }
        Ok(Self {
            name: config.name.clone(),
            aliases: config.aliases.clone(),
            version,
            context_limit: config.context_limit,
            tokenizer: Arc::new(tokenizer),
            eos_token_id,
            ignore_eos: config.ignore_eos,
            options,
            prompt_tokenizer,
            encoding,
        })
    }

    /// Render the conversation with this model's prompt encoding.
    pub fn render(&self, conversation: &Conversation) -> RenderedPrompt {
        match &self.encoding {
            Encoding::V4(encoding) => encoding.render_conversation(conversation),
            Encoding::V41(encoding) => encoding.render_conversation(conversation),
        }
    }

    /// Tokenize a rendered prompt: special ranges become their token IDs,
    /// plain text is encoded without the added vocabulary.
    pub fn encode(&self, rendered: &RenderedPrompt) -> Result<Vec<u32>, String> {
        self.prompt_tokenizer
            .encode_ids(&rendered.prompt, &rendered.special_ranges)
    }

    /// Every name this model answers to.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        std::iter::once(self.name.as_str()).chain(self.aliases.iter().map(String::as_str))
    }
}

/// All served models, looked up by name or alias.
#[derive(Debug, Default)]
pub struct ModelRegistry {
    models: Vec<Arc<ServedModel>>,
    by_name: HashMap<String, usize>,
}

impl ModelRegistry {
    pub fn from_configs(configs: &[ModelConfig]) -> anyhow::Result<Self> {
        let mut registry = Self::default();
        for config in configs {
            let model = Arc::new(ServedModel::from_config(config)?);
            let index = registry.models.len();
            for name in model.names() {
                if let Some(previous) = registry.by_name.insert(name.to_ascii_lowercase(), index)
                    && previous != index
                {
                    bail!("model name `{name}` is configured for two models");
                }
            }
            registry.models.push(model);
        }
        Ok(registry)
    }

    /// Resolve a requested model name, ignoring ASCII case.
    pub fn resolve(&self, name: &str) -> Option<Arc<ServedModel>> {
        self.by_name
            .get(&name.to_ascii_lowercase())
            .map(|&index| Arc::clone(&self.models[index]))
    }

    /// The canonical name for a name a worker reports, if it is served.
    pub fn canonical(&self, name: &str) -> Option<&str> {
        self.by_name
            .get(&name.to_ascii_lowercase())
            .map(|&index| self.models[index].name.as_str())
    }

    pub fn all(&self) -> &[Arc<ServedModel>] {
        &self.models
    }
}
