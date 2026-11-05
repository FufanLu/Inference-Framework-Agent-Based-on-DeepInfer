"""Qwen3 dense models (GQA with per-head q/k RMSNorm) on the paged cache.

The cache row of a token holds K of every KV head then V of every KV head
(`2 * num_kv_heads * head_dim` values), so the block pool, scheduler and
prefix cache are the same as for MLA.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..batch import ForwardBatch
from ..config import ModelSpec
from ..ops import Ops
from ..rope import RotaryCache


@dataclass
class _Layer:
    input_norm: torch.Tensor
    post_attention_norm: torch.Tensor
    qkv: torch.Tensor  # [(Hq + 2 Hkv) * d, hidden]
    q_norm: torch.Tensor  # [d]
    k_norm: torch.Tensor  # [d]
    o: torch.Tensor  # [hidden, Hq * d]
    w13: torch.Tensor  # [2 I, hidden]
    w2: torch.Tensor  # [hidden, I]


class Qwen3:
    def __init__(
        self,
        spec: ModelSpec,
        weights: dict[str, torch.Tensor],
        ops: Ops,
        max_position: int,
        device: torch.device | str = "cpu",
    ):
        self.spec = spec
        self.ops = ops
        self.device = torch.device(device)
        w = {k: v.to(self.device) for k, v in weights.items()}
        self.dtype = w["model.embed_tokens.weight"].dtype
        self.embed = w["model.embed_tokens.weight"]
        self.norm = w["model.norm.weight"]
        self.lm_head = w.get("lm_head.weight", self.embed) if spec.tie_word_embeddings else w["lm_head.weight"]
        self.layers = []
        for i in range(spec.num_layers):
            p = f"model.layers.{i}"
            self.layers.append(
                _Layer(
                    input_norm=w[f"{p}.input_layernorm.weight"],
                    post_attention_norm=w[f"{p}.post_attention_layernorm.weight"],
                    qkv=torch.cat([w[f"{p}.self_attn.{n}_proj.weight"] for n in ("q", "k", "v")]),
                    q_norm=w[f"{p}.self_attn.q_norm.weight"],
                    k_norm=w[f"{p}.self_attn.k_norm.weight"],
                    o=w[f"{p}.self_attn.o_proj.weight"],
                    w13=torch.cat([w[f"{p}.mlp.gate_proj.weight"], w[f"{p}.mlp.up_proj.weight"]]),
                    w2=w[f"{p}.mlp.down_proj.weight"],
                )
            )
        self.rope = RotaryCache(spec.rope, spec.head_dim, max_position, self.device, interleaved=False)
        self.scale = spec.head_dim**-0.5

    def kv_cache_shape(self, num_blocks: int, block_size: int) -> tuple[int, int, int]:
        return (num_blocks, block_size, self.spec.kv_cache_dim)

    def _attention(self, layer: _Layer, h: torch.Tensor, batch: ForwardBatch, cache: torch.Tensor) -> torch.Tensor:
        s, ops = self.spec, self.ops
        T, d, hq, hkv = h.shape[0], s.head_dim, s.num_heads, s.num_kv_heads
        qkv = ops.linear(h, layer.qkv)
        q = qkv[:, : hq * d].reshape(T, hq, d)
        k = qkv[:, hq * d : (hq + hkv) * d].reshape(T, hkv, d)
        v = qkv[:, (hq + hkv) * d :].reshape(T, hkv, d)
        q = self.rope.apply(ops.rms_norm(q, layer.q_norm, s.rms_norm_eps), batch.positions)
        k = self.rope.apply(ops.rms_norm(k, layer.k_norm, s.rms_norm_eps), batch.positions)
        ops.write_kv(cache, batch.slot_mapping, torch.cat([k.reshape(T, -1), v.reshape(T, -1)], dim=-1))
        out = ops.gqa_attention(q.contiguous(), cache, batch, self.scale, hkv)
        return ops.linear(out.reshape(T, -1), layer.o)

    @torch.inference_mode()
    def forward(self, batch: ForwardBatch, kv_caches: list[torch.Tensor]) -> torch.Tensor:
        s, ops = self.spec, self.ops
        x = self.embed[batch.input_ids]
        for layer, cache in zip(self.layers, kv_caches, strict=True):
            h = ops.rms_norm(x, layer.input_norm, s.rms_norm_eps)
            x = x + self._attention(layer, h, batch, cache)
            h = ops.rms_norm(x, layer.post_attention_norm, s.rms_norm_eps)
            x = x + ops.linear(ops.silu_and_mul(ops.linear(h, layer.w13)), layer.w2)
        x = ops.rms_norm(x[batch.logits_indices], self.norm, s.rms_norm_eps)
        logits = ops.linear(x, self.lm_head)
        return logits if logits.dtype == torch.float64 else logits.float()
