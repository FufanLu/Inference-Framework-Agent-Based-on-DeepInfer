from __future__ import annotations

import pytest
import torch

TINY = dict(
    vocab_size=320,
    hidden_size=64,
    intermediate_size=96,
    moe_intermediate_size=24,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_key_value_heads=4,
    kv_lora_rank=32,
    q_lora_rank=None,
    qk_nope_head_dim=16,
    qk_rope_head_dim=8,
    v_head_dim=16,
    n_routed_experts=8,
    n_shared_experts=2,
    num_experts_per_tok=3,
    first_k_dense_replace=1,
    max_position_embeddings=512,
    rope_scaling={
        "type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 128,
        "beta_fast": 32,
        "beta_slow": 1,
        "mscale": 0.707,
        "mscale_all_dim": 0.707,
    },
    bos_token_id=0,
    eos_token_id=1,
)


def tiny_checkpoint(tmp_path_factory, name: str, **overrides):
    """A randomly initialised DeepSeek-V2 checkpoint (float64) and its HF model."""
    from transformers import DeepseekV2Config, DeepseekV2ForCausalLM

    torch.manual_seed(0)
    config = DeepseekV2Config(**{**TINY, **overrides})
    # The default grouped-mm experts do not take float64.
    config._experts_implementation = "eager"
    config._attn_implementation = "eager"
    model = DeepseekV2ForCausalLM(config).double().eval()
    # Default init leaves routers near uniform; spread them so routing matters.
    with torch.no_grad():
        for p_name, p in model.named_parameters():
            if p_name.endswith("mlp.gate.weight"):
                p.normal_(0, 0.5)
    path = tmp_path_factory.mktemp(name)
    model.save_pretrained(path)
    return path, model


@pytest.fixture(scope="session")
def tiny_v2(tmp_path_factory):
    return tiny_checkpoint(tmp_path_factory, "tiny-v2")


@pytest.fixture(scope="session")
def tiny_v2_qlora(tmp_path_factory):
    return tiny_checkpoint(
        tmp_path_factory, "tiny-v2-qlora", q_lora_rank=24, topk_method="group_limited_greedy", n_group=4, topk_group=2
    )
