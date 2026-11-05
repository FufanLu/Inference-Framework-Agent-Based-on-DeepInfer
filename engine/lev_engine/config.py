"""Model and engine configuration.

`ModelSpec` is read from a Hugging Face ``config.json`` and holds only what
the engine needs. It accepts both the original DeepSeek-V2 checkpoint format
(``rope_scaling``) and the transformers 5 format (``rope_parameters``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RopeSpec:
    theta: float = 10000.0
    kind: str = "default"  # "default" or "yarn"
    factor: float = 1.0
    original_max_position: int = 4096
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    mscale: float = 1.0
    mscale_all_dim: float = 0.0


@dataclass(frozen=True)
class ModelSpec:
    architecture: str
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    intermediate_size: int
    rms_norm_eps: float
    max_position: int
    rope: RopeSpec
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    tie_word_embeddings: bool = False
    # GQA (Qwen3): cache holds K and V of every KV head.
    num_kv_heads: int = 0
    head_dim: int = 0
    qk_norm: bool = False
    #: RoPE pairs: interleaved (x[2i], x[2i+1]) for DeepSeek, halves for Qwen.
    rope_interleaved: bool = True
    # MLA
    q_lora_rank: int | None = None
    kv_lora_rank: int = 0
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    v_head_dim: int = 0
    # MoE
    n_routed_experts: int = 0
    n_shared_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    first_k_dense_replace: int = 0
    moe_layer_freq: int = 1
    topk_method: str = "greedy"
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = False
    routed_scaling_factor: float = 1.0

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def is_mla(self) -> bool:
        return self.kv_lora_rank > 0

    @property
    def kv_cache_dim(self) -> int:
        """Values cached per token per layer: the MLA latent plus the shared
        RoPE key, or K and V of every KV head (GQA)."""
        if self.is_mla:
            return self.kv_lora_rank + self.qk_rope_head_dim
        return 2 * self.num_kv_heads * self.head_dim

    def is_moe_layer(self, layer: int) -> bool:
        return (
            self.n_routed_experts > 0
            and layer >= self.first_k_dense_replace
            and layer % self.moe_layer_freq == 0
        )

    @classmethod
    def from_hf_config(cls, config: dict[str, Any]) -> ModelSpec:
        arch = (config.get("architectures") or [config.get("model_type", "")])[0]
        if arch == "Qwen3ForCausalLM":
            return cls._qwen3(config)
        if arch not in ("DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM"):
            raise ValueError(f"unsupported architecture {arch!r}")
        if arch == "DeepseekV3ForCausalLM" or config.get("scoring_func", "softmax") != "softmax":
            raise ValueError("only DeepSeek-V2 routing (softmax scores) is implemented")
        rope_cfg = config.get("rope_scaling") or config.get("rope_parameters") or {}
        kind = rope_cfg.get("rope_type") or rope_cfg.get("type") or "default"
        rope = RopeSpec(
            theta=float(rope_cfg.get("rope_theta", config.get("rope_theta", 10000.0))),
            kind=kind,
            factor=float(rope_cfg.get("factor", 1.0)),
            original_max_position=int(rope_cfg.get("original_max_position_embeddings", 4096)),
            beta_fast=float(rope_cfg.get("beta_fast", 32)),
            beta_slow=float(rope_cfg.get("beta_slow", 1)),
            mscale=float(rope_cfg.get("mscale", 1.0)),
            mscale_all_dim=float(rope_cfg.get("mscale_all_dim", 0.0)),
        )
        eos = config.get("eos_token_id")
        if isinstance(eos, list):
            eos = eos[0]
        return cls(
            architecture=arch,
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_hidden_layers"],
            num_heads=config["num_attention_heads"],
            intermediate_size=config["intermediate_size"],
            rms_norm_eps=config.get("rms_norm_eps", 1e-6),
            max_position=config.get("max_position_embeddings", 4096),
            rope=rope,
            bos_token_id=config.get("bos_token_id"),
            eos_token_id=eos,
            q_lora_rank=config.get("q_lora_rank"),
            kv_lora_rank=config["kv_lora_rank"],
            qk_nope_head_dim=config["qk_nope_head_dim"],
            qk_rope_head_dim=config["qk_rope_head_dim"],
            v_head_dim=config["v_head_dim"],
            n_routed_experts=config.get("n_routed_experts") or 0,
            n_shared_experts=config.get("n_shared_experts") or 0,
            num_experts_per_tok=config.get("num_experts_per_tok") or 0,
            moe_intermediate_size=config.get("moe_intermediate_size") or 0,
            first_k_dense_replace=config.get("first_k_dense_replace", 0),
            moe_layer_freq=config.get("moe_layer_freq", 1),
            topk_method=config.get("topk_method", "greedy"),
            n_group=config.get("n_group") or 1,
            topk_group=config.get("topk_group") or 1,
            norm_topk_prob=bool(config.get("norm_topk_prob", False)),
            routed_scaling_factor=float(config.get("routed_scaling_factor", 1.0)),
        )

    @classmethod
    def _qwen3(cls, config: dict[str, Any]) -> ModelSpec:
        if config.get("rope_scaling") or config.get("use_sliding_window"):
            raise ValueError("Qwen3 with rope scaling or sliding windows is not implemented")
        heads = config["num_attention_heads"]
        eos = config.get("eos_token_id")
        return cls(
            architecture="Qwen3ForCausalLM",
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_hidden_layers"],
            num_heads=heads,
            intermediate_size=config["intermediate_size"],
            rms_norm_eps=config.get("rms_norm_eps", 1e-6),
            max_position=config.get("max_position_embeddings", 4096),
            rope=RopeSpec(theta=float(config.get("rope_theta") or (config.get("rope_parameters") or {}).get("rope_theta", 1e6))),
            bos_token_id=config.get("bos_token_id"),
            eos_token_id=eos[0] if isinstance(eos, list) else eos,
            tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
            num_kv_heads=config["num_key_value_heads"],
            head_dim=config.get("head_dim") or config["hidden_size"] // heads,
            qk_norm=True,
            rope_interleaved=False,
        )

    @classmethod
    def from_pretrained(cls, path: str | Path) -> ModelSpec:
        return cls.from_hf_config(json.loads((Path(path) / "config.json").read_text()))


@dataclass
class EngineConfig:
    #: Tokens per KV block. Must match the router's block size for exact
    #: prefix routing.
    block_size: int = 16
    #: KV blocks per layer. 0 means: size from `kv_cache_memory_bytes`.
    num_blocks: int = 0
    kv_cache_memory_bytes: int = 0
    #: Token budget of one forward step (prefill chunks + decode tokens).
    max_num_batched_tokens: int = 2048
    max_num_seqs: int = 128
    #: Longest prompt + output; requests beyond it are rejected.
    max_model_len: int = 4096
    enable_prefix_caching: bool = True
    #: "reference" (PyTorch, any device) or "triton".
    kernel_backend: str = "reference"
    #: Batch-invariant numerics: a request's tokens do not depend on what else
    #: is in the batch (fixed reduction orders, no batch-size-dependent tiling).
    batch_invariant: bool = False
    #: Replay decode steps from CUDA graphs (Triton backend on CUDA only).
    cuda_graph: bool = False
    cuda_graph_max_batch: int = 256
    device: str = "cpu"
    dtype: str = "float32"
    extra: dict[str, Any] = field(default_factory=dict)
