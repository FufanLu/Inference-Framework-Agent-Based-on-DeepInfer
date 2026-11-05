"""Triton backend: launch wrappers around `kernels`.

`triton_ops(batch_invariant=False)` uses cuBLAS (torch.matmul) for dense
GEMMs; `batch_invariant=True` switches every GEMM to the fixed-tile Triton
matmul. Attention, norms and MoE always use the Triton kernels, which are
batch-invariant by construction.
"""

from __future__ import annotations

import torch
import triton

from .. import Ops, reference
from ..reference import acc
from . import kernels

# One tile configuration for every launch (see kernels.py on why it is fixed).
MM_BLOCK_M, MM_BLOCK_N, MM_BLOCK_K = 64, 64, 32
MOE_BLOCK_M, MOE_BLOCK_N, MOE_BLOCK_K = 16, 64, 32
ATTN_BLOCK_N = 32


def _pow2(n: int, floor: int = 16) -> int:
    return max(floor, triton.next_power_of_2(n))


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    out = torch.empty(x2.shape, dtype=x.dtype, device=x.device)
    if x2.shape[0]:
        kernels.rms_norm_kernel[(x2.shape[0],)](
            x2, weight, out, shape[-1], x2.stride(0), out.stride(0), eps, BLOCK=triton.next_power_of_2(shape[-1])
        )
    return out.view(shape)


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    n = x2.shape[-1] // 2
    out = torch.empty((x2.shape[0], n), dtype=x.dtype, device=x.device)
    block = min(1024, triton.next_power_of_2(n))
    if x2.shape[0]:
        kernels.silu_and_mul_kernel[(x2.shape[0], triton.cdiv(n, block))](
            x2, out, n, x2.stride(0), out.stride(0), BLOCK=block
        )
    return out.view(*x.shape[:-1], n)


def _ieee(t: torch.Tensor) -> bool:
    """fp32 inputs: full-precision dots, as torch's fp32 matmul (no TF32); bf16: tensor cores."""
    return t.dtype == torch.float32


def _matmul(a, b, c, M, N, K, batch, sa, sb, sc) -> None:
    """sa = (batch, m, k), sb = (batch, n, k), sc = (batch, m, n) strides."""
    if M == 0:
        return
    grid = (triton.cdiv(M, MM_BLOCK_M), triton.cdiv(N, MM_BLOCK_N), batch)
    kernels.matmul_kernel[grid](
        a, b, c, M, N, K, *sa, *sb, *sc, BLOCK_M=MM_BLOCK_M, BLOCK_N=MM_BLOCK_N, BLOCK_K=MM_BLOCK_K,
        IEEE=_ieee(a),
    )


def linear_invariant(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    M, K = x2.shape
    N = weight.shape[0]
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    _matmul(
        x2, weight, out, M, N, K, 1,
        (0, x2.stride(0), x2.stride(1)), (0, weight.stride(0), weight.stride(1)), (0, out.stride(0), out.stride(1)),
    )  # fmt: skip
    return out.view(*shape[:-1], N)


def head_matmul_invariant(x: torch.Tensor, weight: torch.Tensor, transpose: bool) -> torch.Tensor:
    T, H, a = x.shape
    if transpose:  # weight [H, b, a]
        b = weight.shape[1]
        sb = (weight.stride(0), weight.stride(1), weight.stride(2))
    else:  # weight [H, a, b]
        b = weight.shape[2]
        sb = (weight.stride(0), weight.stride(2), weight.stride(1))
    out = torch.empty((T, H, b), dtype=x.dtype, device=x.device)
    _matmul(
        x, weight, out, T, b, a, H,
        (x.stride(1), x.stride(0), x.stride(2)), sb, (out.stride(1), out.stride(0), out.stride(2)),
    )  # fmt: skip
    return out


def mla_attention(q_latent, q_rope, cache, batch, scale) -> torch.Tensor:
    T, H, R = q_latent.shape
    P = q_rope.shape[-1]
    assert q_latent.stride(-1) == 1 and q_rope.stride(-1) == 1 and cache.is_contiguous()
    out = torch.empty_like(q_latent)
    if T:
        kernels.mla_attention_kernel[(T,)](
            q_latent, q_rope, cache, out, batch.block_tables, batch.token_seq, batch.positions,
            H, R, P, cache.shape[1], scale,
            q_latent.stride(0), q_latent.stride(1), q_rope.stride(0), q_rope.stride(1),
            cache.stride(1), batch.block_tables.stride(0), out.stride(0), out.stride(1),
            BLOCK_H=_pow2(H), BLOCK_R=_pow2(R), BLOCK_P=_pow2(P), BLOCK_N=ATTN_BLOCK_N, IEEE=_ieee(q_latent),
        )  # fmt: skip
    return out


def gqa_attention(q, cache, batch, scale, num_kv_heads) -> torch.Tensor:
    T, Hq, D = q.shape
    G = Hq // num_kv_heads
    assert q.stride(-1) == 1 and cache.is_contiguous()
    out = torch.empty_like(q)
    if T:
        kernels.gqa_attention_kernel[(T, num_kv_heads)](
            q, cache, out, batch.block_tables, batch.token_seq, batch.positions,
            G, D, num_kv_heads, cache.shape[1], scale,
            q.stride(0), q.stride(1), cache.stride(1), batch.block_tables.stride(0), out.stride(0), out.stride(1),
            BLOCK_G=_pow2(G), BLOCK_D=_pow2(D), BLOCK_N=ATTN_BLOCK_N, IEEE=_ieee(q),
        )  # fmt: skip
    return out


def align_by_expert(topk_ids: torch.Tensor, num_experts: int, block_m: int):
    """Rows (positions in the flattened [T*k] list) grouped by expert, each
    group padded to a multiple of `block_m` with -1, and the expert of every
    block of `block_m` rows (-1 for unused blocks).

    Sized by the worst case (T*k + E*(block_m-1) rows) and computed with
    searchsorted, so it never reads a count back to the host: no
    synchronisation per MoE layer, and capturable in a CUDA graph."""
    flat = topk_ids.reshape(-1)
    n = flat.numel()
    order = torch.argsort(flat, stable=True)
    # Not torch.bincount: on CUDA it reads the maximum back to size its output.
    counts = torch.zeros(num_experts, dtype=torch.int64, device=flat.device)
    counts.scatter_add_(0, flat, torch.ones_like(flat))
    padded = (counts + block_m - 1) // block_m * block_m
    ends_padded = torch.cumsum(padded, 0)
    starts_padded = ends_padded - padded
    starts_sorted = torch.cumsum(counts, 0) - counts
    sorted_experts = flat[order]
    rank = torch.arange(n, device=flat.device) - starts_sorted[sorted_experts]
    capacity = (n + num_experts * (block_m - 1) + block_m - 1) // block_m * block_m
    rows = torch.full((capacity,), -1, dtype=torch.int64, device=flat.device)
    rows[starts_padded[sorted_experts] + rank] = order
    block_start = torch.arange(0, capacity, block_m, device=flat.device)
    block_expert = torch.searchsorted(ends_padded, block_start, right=True)
    block_expert = torch.where(block_expert < num_experts, block_expert, -1)
    return rows, block_expert.to(torch.int32)


def moe_routed(x, topk_ids, topk_weights, w13, w2) -> torch.Tensor:
    tokens, k = topk_ids.shape
    hidden = x.shape[-1]
    inter = w2.shape[-1]
    rows, block_expert = align_by_expert(topk_ids, w13.shape[0], MOE_BLOCK_M)
    src = torch.where(rows >= 0, rows // k, rows)
    x = x.contiguous()
    h = torch.empty((tokens * k, inter), dtype=x.dtype, device=x.device)
    y = torch.empty((tokens * k, hidden), dtype=x.dtype, device=x.device)
    blocks = block_expert.numel()
    if blocks:
        kernels.moe_gemm_kernel[(blocks, triton.cdiv(inter, MOE_BLOCK_N))](
            x, w13, h, rows, src, block_expert, inter, hidden, rows.numel(),
            x.stride(0), w13.stride(0), w13.stride(1), w13.stride(2), h.stride(0),
            FUSE_SILU=True, BLOCK_M=MOE_BLOCK_M, BLOCK_N=MOE_BLOCK_N, BLOCK_K=MOE_BLOCK_K, IEEE=_ieee(x),
        )  # fmt: skip
        kernels.moe_gemm_kernel[(blocks, triton.cdiv(hidden, MOE_BLOCK_N))](
            h, w2, y, rows, rows, block_expert, hidden, inter, rows.numel(),
            h.stride(0), w2.stride(0), w2.stride(1), w2.stride(2), y.stride(0),
            FUSE_SILU=False, BLOCK_M=MOE_BLOCK_M, BLOCK_N=MOE_BLOCK_N, BLOCK_K=MOE_BLOCK_K, IEEE=_ieee(x),
        )  # fmt: skip
    weighted = acc(y.view(tokens, k, hidden)) * topk_weights.unsqueeze(-1)
    out = weighted[:, 0]
    for j in range(1, k):  # fixed summation order, as in the reference
        out = out + weighted[:, j]
    return out.to(x.dtype)


def triton_ops(batch_invariant: bool) -> Ops:
    return Ops(
        name="triton-invariant" if batch_invariant else "triton",
        linear=linear_invariant if batch_invariant else reference.linear,
        head_matmul=head_matmul_invariant if batch_invariant else reference.head_matmul,
        rms_norm=rms_norm,
        silu_and_mul=silu_and_mul,
        write_kv=reference.write_kv,
        mla_attention=mla_attention,
        gqa_attention=gqa_attention,
        moe_routed=moe_routed,
    )
