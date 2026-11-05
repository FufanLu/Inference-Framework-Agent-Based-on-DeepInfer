"""DeepSeek-V2 (MLA + DeepSeekMoE) on a paged latent KV cache.

MLA with weight absorption. The cache holds, per token and layer, only the
normalised latent ``c_kv`` (kv_lora_rank values) and the shared RoPE key
(qk_rope_head_dim values): 576 numbers for V2-Lite instead of
2 * 16 heads * 128 for plain multi-head attention. Keys and values are never
expanded: ``W_UK`` is folded into the query and ``W_UV`` applied after the
attention, so attention runs directly over the latents::

    score(i, j) = (q_nope_i W_UK) . c_j  +  q_rope_i . k_rope_j
    out_i       = (sum_j p_ij c_j) W_UV

Prefill and decode take the same path, which is what makes a token's
numerics independent of whether it was computed in a prefill chunk or a
decode step (batch-invariant mode relies on that).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..batch import ForwardBatch
from ..config import ModelSpec
from ..ops import Ops
from ..parallel import Parallel, ep_dispatch_combine
from ..rope import RotaryCache, softmax_scale


@dataclass
class _Attention:
    q: torch.Tensor | None  # [H * qk, hidden] when q_lora_rank is None
    q_a: torch.Tensor | None
    q_a_norm: torch.Tensor | None
    q_b: torch.Tensor | None
    kv_a: torch.Tensor  # [rank + rope, hidden]
    kv_a_norm: torch.Tensor  # [rank]
    w_uk: torch.Tensor  # [H, nope, rank]
    w_uv: torch.Tensor  # [H, v, rank]
    o: torch.Tensor  # [hidden, H * v]


@dataclass
class _Dense:
    w13: torch.Tensor  # [2 * I, hidden]
    w2: torch.Tensor  # [hidden, I]


@dataclass
class _Moe:
    gate: torch.Tensor  # [E, hidden], used in fp32
    w13: torch.Tensor  # [E, 2 * I, hidden]
    w2: torch.Tensor  # [E, hidden, I]
    shared: _Dense | None


@dataclass
class _Layer:
    input_norm: torch.Tensor
    post_attention_norm: torch.Tensor
    attn: _Attention
    mlp: _Dense | _Moe


class DeepseekV2:
    def __init__(
        self,
        spec: ModelSpec,
        weights: dict[str, torch.Tensor],
        ops: Ops,
        max_position: int,
        device: torch.device | str = "cpu",
        parallel: Parallel | None = None,
    ):
        self.spec = spec
        self.ops = ops
        self.device = torch.device(device)
        self.par = parallel
        tp = parallel.size if parallel is not None and parallel.tp else 1
        self.local_heads = spec.num_heads // tp
        w = {k: v.to(self.device) for k, v in weights.items()}
        self.dtype = w["model.embed_tokens.weight"].dtype
        self.embed = w["model.embed_tokens.weight"]
        self.norm = w["model.norm.weight"]
        self.lm_head = w["lm_head.weight"]
        self.layers = [self._layer(w, i) for i in range(spec.num_layers)]
        self.rope = RotaryCache(spec.rope, spec.qk_rope_head_dim, max_position, self.device)
        self.scale = softmax_scale(spec.rope, spec.qk_head_dim)

    # ------------------------------------------------------------------ weights

    def _dense(self, w: dict[str, torch.Tensor], prefix: str) -> _Dense:
        gate, up, down = (w[f"{prefix}.{n}_proj.weight"] for n in ("gate", "up", "down"))
        if self.par is not None and self.par.tp:  # split the intermediate dimension
            cols = self.par.shard(gate.shape[0])
            gate, up, down = gate[cols], up[cols], down[:, cols]
        return _Dense(w13=torch.cat([gate, up], dim=0).contiguous(), w2=down.contiguous())

    def _layer(self, w: dict[str, torch.Tensor], i: int) -> _Layer:
        s, par = self.spec, self.par
        p = f"model.layers.{i}"
        tp = par is not None and par.tp
        heads = par.shard(s.num_heads) if tp else slice(None)

        def by_head(t: torch.Tensor, per_head: int) -> torch.Tensor:
            """Rows of a [H * per_head, in] projection for this rank's heads."""
            return t.view(s.num_heads, per_head, -1)[heads].reshape(-1, t.shape[-1]).contiguous()

        kv_b = w[f"{p}.self_attn.kv_b_proj.weight"].view(s.num_heads, s.qk_nope_head_dim + s.v_head_dim, s.kv_lora_rank)
        kv_b = kv_b[heads]
        o = w[f"{p}.self_attn.o_proj.weight"]
        o = o.view(o.shape[0], s.num_heads, s.v_head_dim)[:, heads].reshape(o.shape[0], -1).contiguous()
        lora = s.q_lora_rank is not None
        attn = _Attention(
            q=None if lora else by_head(w[f"{p}.self_attn.q_proj.weight"], s.qk_head_dim),
            q_a=w[f"{p}.self_attn.q_a_proj.weight"] if lora else None,
            q_a_norm=w[f"{p}.self_attn.q_a_layernorm.weight"] if lora else None,
            q_b=by_head(w[f"{p}.self_attn.q_b_proj.weight"], s.qk_head_dim) if lora else None,
            kv_a=w[f"{p}.self_attn.kv_a_proj_with_mqa.weight"],
            kv_a_norm=w[f"{p}.self_attn.kv_a_layernorm.weight"],
            w_uk=kv_b[:, : s.qk_nope_head_dim].contiguous(),
            w_uv=kv_b[:, s.qk_nope_head_dim :].contiguous(),
            o=o,
        )
        if s.is_moe_layer(i):
            experts = range(s.n_routed_experts)
            if par is not None and par.ep:
                experts = range(s.n_routed_experts)[par.shard(s.n_routed_experts)]
            routed = [self._expert(w, f"{p}.mlp.experts.{e}") for e in experts]
            mlp: _Dense | _Moe = _Moe(
                gate=w[f"{p}.mlp.gate.weight"].float(),
                w13=torch.stack([r.w13 for r in routed]),
                w2=torch.stack([r.w2 for r in routed]),
                shared=self._shared(w, f"{p}.mlp.shared_experts") if s.n_shared_experts else None,
            )
        else:
            mlp = self._layer_dense(w, f"{p}.mlp")
        return _Layer(
            input_norm=w[f"{p}.input_layernorm.weight"],
            post_attention_norm=w[f"{p}.post_attention_layernorm.weight"],
            attn=attn,
            mlp=mlp,
        )

    def _expert(self, w: dict[str, torch.Tensor], prefix: str) -> _Dense:
        """A routed expert: split like a dense MLP under TP, whole under EP."""
        return self._dense(w, prefix) if self.par is None or self.par.tp else self._whole(w, prefix)

    def _shared(self, w: dict[str, torch.Tensor], prefix: str) -> _Dense:
        """Shared experts are split under TP, replicated under EP (attention is local)."""
        return self._expert(w, prefix)

    def _layer_dense(self, w: dict[str, torch.Tensor], prefix: str) -> _Dense:
        return self._expert(w, prefix)

    @staticmethod
    def _whole(w: dict[str, torch.Tensor], prefix: str) -> _Dense:
        gate, up, down = (w[f"{prefix}.{n}_proj.weight"] for n in ("gate", "up", "down"))
        return _Dense(w13=torch.cat([gate, up], dim=0), w2=down)

    def kv_cache_shape(self, num_blocks: int, block_size: int) -> tuple[int, int, int]:
        return (num_blocks, block_size, self.spec.kv_cache_dim)

    # ------------------------------------------------------------------ forward

    def _attention(self, a: _Attention, h: torch.Tensor, batch: ForwardBatch, cache: torch.Tensor) -> torch.Tensor:
        s, ops = self.spec, self.ops
        tokens = h.shape[0]
        if a.q is not None:
            q = ops.linear(h, a.q)
        else:
            q = ops.linear(ops.rms_norm(ops.linear(h, a.q_a), a.q_a_norm, s.rms_norm_eps), a.q_b)
        q = q.view(tokens, self.local_heads, s.qk_head_dim)
        q_nope, q_rope = q[..., : s.qk_nope_head_dim], q[..., s.qk_nope_head_dim :]
        kv = ops.linear(h, a.kv_a)
        latent = ops.rms_norm(kv[:, : s.kv_lora_rank], a.kv_a_norm, s.rms_norm_eps)
        k_rope = self.rope.apply(kv[:, s.kv_lora_rank :], batch.positions)
        q_rope = self.rope.apply(q_rope, batch.positions)
        ops.write_kv(cache, batch.slot_mapping, torch.cat([latent, k_rope], dim=-1))
        q_latent = ops.head_matmul(q_nope, a.w_uk, transpose=False)
        out_latent = ops.mla_attention(q_latent, q_rope, cache, batch, self.scale)
        out = ops.head_matmul(out_latent, a.w_uv, transpose=True).reshape(tokens, -1)
        out = ops.linear(out, a.o)
        if self.par is not None and self.par.tp:  # each rank summed over its heads only
            self.par.all_reduce(out)
        return out

    def _dense_mlp(self, m: _Dense, h: torch.Tensor, reduce: bool = True) -> torch.Tensor:
        ops = self.ops
        out = ops.linear(ops.silu_and_mul(ops.linear(h, m.w13)), m.w2)
        if reduce and self.par is not None and self.par.tp:
            self.par.all_reduce(out)
        return out

    def route(self, m: _Moe, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Top-k experts and their weights (fp32) per token: DeepSeek-V2 softmax routing."""
        s = self.spec
        scores = (h.float() @ m.gate.t()).softmax(dim=-1)
        if s.topk_method == "group_limited_greedy":
            tokens = scores.shape[0]
            groups = scores.view(tokens, s.n_group, -1).max(dim=-1).values
            keep = torch.topk(groups, k=s.topk_group, dim=-1, sorted=False).indices
            mask = torch.zeros_like(groups).scatter_(1, keep, 1.0)
            mask = mask.unsqueeze(-1).expand(tokens, s.n_group, s.n_routed_experts // s.n_group).reshape(tokens, -1)
            scores = scores.masked_fill(mask == 0, 0.0)
        weights, ids = torch.topk(scores, k=s.num_experts_per_tok, dim=-1, sorted=False)
        if s.norm_topk_prob and s.num_experts_per_tok > 1:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return ids, weights * s.routed_scaling_factor

    def _moe(self, m: _Moe, h: torch.Tensor) -> torch.Tensor:
        ids, weights = self.route(m, h)
        par = self.par
        if par is not None and par.ep:
            local = m.w13.shape[0]

            def experts(rows: torch.Tensor, local_ids: torch.Tensor) -> torch.Tensor:
                ones = torch.ones((rows.shape[0], 1), dtype=torch.float32, device=rows.device)
                return self.ops.moe_routed(rows, local_ids[:, None], ones, m.w13, m.w2)

            out = ep_dispatch_combine(h, ids, weights, local, experts, par)
        else:
            out = self.ops.moe_routed(h, ids, weights, m.w13, m.w2)
        if m.shared is not None:
            out = out + self._dense_mlp(m.shared, h, reduce=False)
        if par is not None and par.tp:  # routed and shared partial sums, reduced once
            par.all_reduce(out)
        return out

    @torch.inference_mode()
    def forward(self, batch: ForwardBatch, kv_caches: list[torch.Tensor]) -> torch.Tensor:
        """Run one step; returns logits (fp32) for `batch.logits_indices`."""
        s, ops = self.spec, self.ops
        x = self.embed[batch.input_ids]
        for layer, cache in zip(self.layers, kv_caches, strict=True):
            h = ops.rms_norm(x, layer.input_norm, s.rms_norm_eps)
            x = x + self._attention(layer.attn, h, batch, cache)
            h = ops.rms_norm(x, layer.post_attention_norm, s.rms_norm_eps)
            x = x + (self._moe(layer.mlp, h) if isinstance(layer.mlp, _Moe) else self._dense_mlp(layer.mlp, h))
        x = ops.rms_norm(x[batch.logits_indices], self.norm, s.rms_norm_eps)
        logits = ops.linear(x, self.lm_head)
        # fp32 for sampling; float64 (tests) is kept as is.
        return logits if logits.dtype == torch.float64 else logits.float()
