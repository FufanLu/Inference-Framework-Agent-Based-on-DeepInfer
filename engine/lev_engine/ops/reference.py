"""Reference kernels in plain PyTorch: correct on any device, not fast.

These define the numerics the Triton kernels are tested against, and they
run the engine on CPU in tests. Every function takes and returns tensors in
the engine's working dtype; reductions that matter (norms, softmax) are done
in fp32.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..batch import ForwardBatch


def acc(x: torch.Tensor) -> torch.Tensor:
    """Accumulation dtype: at least fp32 (float64 stays float64, for tests)."""
    return x.to(torch.promote_types(x.dtype, torch.float32))


def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """x @ weight.T  (weight is [out, in], as stored in checkpoints)."""
    return x @ weight.t()


def head_matmul(x: torch.Tensor, weight: torch.Tensor, transpose: bool) -> torch.Tensor:
    """Per-head matmul: x [T, H, a] with weight [H, a, b] -> [T, H, b], or with
    weight [H, b, a] when `transpose` (x @ weight[h].T)."""
    return torch.einsum("tha,hba->thb" if transpose else "tha,hab->thb", x, weight)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = acc(x)
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return weight * xf.to(x.dtype)


def silu_and_mul(gate_up: torch.Tensor) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up


def write_kv(cache: torch.Tensor, slot_mapping: torch.Tensor, kv: torch.Tensor) -> None:
    """Store `kv[T, D]` into the paged cache `[num_blocks, block_size, D]`."""
    flat = cache.view(-1, cache.shape[-1])
    flat.index_copy_(0, slot_mapping, kv.to(cache.dtype))


def mla_attention(
    q_latent: torch.Tensor,  # [T, H, kv_lora_rank]   q_nope absorbed through W_UK
    q_rope: torch.Tensor,  # [T, H, rope_dim]        rotated
    cache: torch.Tensor,  # [num_blocks, block_size, kv_lora_rank + rope_dim]
    batch: ForwardBatch,
    scale: float,
) -> torch.Tensor:
    """Causal MLA attention in the latent space, over a paged cache.

    Every query token attends to every cached token of its sequence at a
    position not after its own (cached prefix, earlier chunks and the new
    tokens, which `write_kv` stored before this call). Returns the attention
    output still in the latent space, ``[T, H, kv_lora_rank]``.
    """
    rank = q_latent.shape[-1]
    block_size = cache.shape[1]
    out = torch.empty_like(q_latent)
    start = 0
    for seq, (q_len, ctx_len) in enumerate(zip(batch.query_lens_cpu, batch.seq_lens_cpu, strict=True)):
        if q_len == 0:
            continue
        blocks = batch.block_tables[seq, : (ctx_len + block_size - 1) // block_size].long()
        kv = acc(cache[blocks].reshape(-1, cache.shape[-1])[:ctx_len])  # [L, D]
        latent, k_rope = kv[:, :rank], kv[:, rank:]
        ql = acc(q_latent[start : start + q_len])  # [q, H, R]
        qr = acc(q_rope[start : start + q_len])
        scores = (torch.einsum("qhr,lr->hql", ql, latent) + torch.einsum("qhd,ld->hql", qr, k_rope)) * scale
        # Query i sits at position ctx_len - q_len + i.
        q_pos = torch.arange(ctx_len - q_len, ctx_len, device=scores.device)
        k_pos = torch.arange(ctx_len, device=scores.device)
        scores = scores.masked_fill(k_pos[None, :] > q_pos[:, None], float("-inf"))
        probs = scores.softmax(dim=-1)
        out[start : start + q_len] = torch.einsum("hql,lr->qhr", probs, latent).to(out.dtype)
        start += q_len
    return out


def moe_routed(
    x: torch.Tensor,  # [T, hidden]
    topk_ids: torch.Tensor,  # [T, k]
    topk_weights: torch.Tensor,  # [T, k] fp32
    w13: torch.Tensor,  # [E, 2 * I, hidden]  gate and up, stacked
    w2: torch.Tensor,  # [E, hidden, I]
) -> torch.Tensor:
    """Routed experts: tokens grouped by expert, one matmul per expert.

    Each (token, expert) pair's output is weighted and summed per token in the
    fixed order of the token's top-k list, so a token's result does not depend
    on which other tokens share its experts.
    """
    tokens, k = topk_ids.shape
    flat_ids = topk_ids.reshape(-1)
    order = torch.argsort(flat_ids, stable=True)
    counts = torch.bincount(flat_ids, minlength=w13.shape[0]).tolist()
    expert_out = torch.empty(tokens * k, x.shape[-1], dtype=x.dtype, device=x.device)
    token_of = order // k
    start = 0
    for expert, count in enumerate(counts):
        if count == 0:
            continue
        rows = order[start : start + count]
        h = silu_and_mul(x[token_of[start : start + count]] @ w13[expert].t())
        expert_out[rows] = h @ w2[expert].t()
        start += count
    weighted = acc(expert_out.view(tokens, k, -1)) * topk_weights.unsqueeze(-1)
    out = weighted[:, 0]
    for j in range(1, k):  # fixed summation order
        out = out + weighted[:, j]
    return out.to(x.dtype)
