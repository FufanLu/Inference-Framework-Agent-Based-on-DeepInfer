"""Triton kernels for the engine.

Batch invariance is a property of how each output element is reduced, so
every kernel here fixes its reduction order independently of how many rows
(tokens) the launch has:

- `rms_norm`: one program per row, the whole row in one block.
- `matmul`: one tile configuration for every M, no split-K: an output
  element always accumulates K in the same BLOCK_K steps in the same order.
- `mla_attention`: one program per query token (all heads at once: MLA's
  heads share one latent KV). The context is reduced in BLOCK_N chunks
  starting at position 0, so a token's result is the same in a prefill chunk
  of 1 or 4096 tokens, in a decode step of batch 1 or 256.
- `moe_*`: grouped GEMM over tokens sorted by expert; a row's result does not
  depend on the other rows in its expert group.

The cost of this (fixed tiles waste work at small M, no split-K or split-KV
parallelism for small batches) is measured in docs/illustrated.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(x_ptr, w_ptr, out_ptr, n_cols, stride_x, stride_out, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / n_cols
    y = x * tl.rsqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    # Same rounding as the reference (normalise in fp32, round to the weight
    # dtype, multiply in that dtype): the exact product of two bf16 values
    # fits in fp32, so multiplying in fp32 and rounding once is identical.
    y = y.to(w.dtype).to(tl.float32) * w.to(tl.float32)
    tl.store(out_ptr + row * stride_out + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def silu_and_mul_kernel(x_ptr, out_ptr, n_cols, stride_x, stride_out, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n_cols
    gate = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_ptr + row * stride_x + n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    y = gate * tl.sigmoid(gate) * up
    tl.store(out_ptr + row * stride_out + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _dot(a, b, IEEE: tl.constexpr):
    """tl.dot; IEEE multiplies fp32 inputs in full fp32 (the default is TF32
    on tensor-core GPUs), as torch's fp32 matmul does."""
    if IEEE:
        return tl.dot(a, b, input_precision="ieee")
    else:
        return tl.dot(a, b)


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_ab, stride_am, stride_ak, stride_bb, stride_bn, stride_bk, stride_cb, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, IEEE: tl.constexpr,
):  # fmt: skip
    """C[b, M, N] = A[b, M, K] @ B[b, N, K]^T, fp32 accumulation (b: grid axis 2).
    IEEE: see _dot."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2).to(tl.int64)
    a_ptr += pid_b * stride_ab
    b_ptr += pid_b * stride_bb
    c_ptr += pid_b * stride_cb
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + rk
        a = tl.load(a_ptr + rm[:, None] * stride_am + kk[None, :] * stride_ak,
                    mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.0)  # fmt: skip
        b = tl.load(b_ptr + rn[None, :] * stride_bn + kk[:, None] * stride_bk,
                    mask=(rn[None, :] < N) & (kk[:, None] < K), other=0.0)  # fmt: skip
        acc += _dot(a, b, IEEE)
    tl.store(c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
             acc.to(c_ptr.dtype.element_ty), mask=(rm[:, None] < M) & (rn[None, :] < N))  # fmt: skip


@triton.jit
def mla_attention_kernel(
    q_lat_ptr, q_rope_ptr, cache_ptr, out_ptr,
    block_tables_ptr, token_seq_ptr, positions_ptr,
    H, R, P, BS, scale,
    stride_ql_t, stride_ql_h, stride_qr_t, stride_qr_h,
    stride_cache_slot, stride_bt, stride_o_t, stride_o_h,
    BLOCK_H: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_N: tl.constexpr, IEEE: tl.constexpr,
):  # fmt: skip
    t = tl.program_id(0)
    seq = tl.load(token_seq_ptr + t)
    ctx = tl.load(positions_ptr + t) + 1  # causal: positions 0..pos
    h = tl.arange(0, BLOCK_H)
    r = tl.arange(0, BLOCK_R)
    p = tl.arange(0, BLOCK_P)
    hm = h < H
    q_lat = tl.load(q_lat_ptr + t * stride_ql_t + h[:, None] * stride_ql_h + r[None, :],
                    mask=hm[:, None] & (r[None, :] < R), other=0.0)  # fmt: skip
    q_rope = tl.load(q_rope_ptr + t * stride_qr_t + h[:, None] * stride_qr_h + p[None, :],
                     mask=hm[:, None] & (p[None, :] < P), other=0.0)  # fmt: skip
    m_i = tl.full((BLOCK_H,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_R), dtype=tl.float32)
    for start in range(0, ctx, BLOCK_N):
        n = start + tl.arange(0, BLOCK_N)
        valid = n < ctx
        block = tl.load(block_tables_ptr + seq * stride_bt + n // BS, mask=valid, other=0)
        slot = block.to(tl.int64) * BS + n % BS
        kv_lat = tl.load(cache_ptr + slot[:, None] * stride_cache_slot + r[None, :],
                         mask=valid[:, None] & (r[None, :] < R), other=0.0)  # fmt: skip
        kv_rope = tl.load(cache_ptr + slot[:, None] * stride_cache_slot + R + p[None, :],
                          mask=valid[:, None] & (p[None, :] < P), other=0.0)  # fmt: skip
        s = _dot(q_lat, tl.trans(kv_lat), IEEE) + _dot(q_rope, tl.trans(kv_rope), IEEE)
        s = tl.where(valid[None, :], s * scale, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        probs = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(probs, axis=1)
        acc = acc * alpha[:, None] + _dot(probs.to(kv_lat.dtype), kv_lat, IEEE)
        m_i = m_new
    out = acc / l_i[:, None]
    tl.store(out_ptr + t * stride_o_t + h[:, None] * stride_o_h + r[None, :],
             out.to(out_ptr.dtype.element_ty), mask=hm[:, None] & (r[None, :] < R))  # fmt: skip


@triton.jit
def moe_gemm_kernel(
    a_ptr, w_ptr, out_ptr, rows_ptr, src_rows_ptr, block_expert_ptr,
    N, K, num_valid,
    stride_am, stride_we, stride_wn, stride_wk, stride_om,
    FUSE_SILU: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, IEEE: tl.constexpr,
):  # fmt: skip
    """Grouped GEMM over rows sorted by expert.

    Program (m, n) computes BLOCK_M rows of one expert (every block of
    `rows_ptr` belongs to one expert, padded with -1). Row r reads activation
    row `src_rows[r]` and writes output row `rows[r]`. With FUSE_SILU, W is
    [E, 2N, K] (gate then up) and the output is silu(gate) * up.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    expert = tl.load(block_expert_ptr + pid_m).to(tl.int64)
    if expert < 0:  # past the last expert's rows (worst-case sizing)
        return
    idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    dst = tl.load(rows_ptr + idx, mask=idx < num_valid, other=-1)
    src = tl.load(src_rows_ptr + idx, mask=idx < num_valid, other=-1)
    live = dst >= 0
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    w_base = w_ptr + expert * stride_we
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + rk
        a = tl.load(a_ptr + src[:, None].to(tl.int64) * stride_am + kk[None, :],
                    mask=live[:, None] & (kk[None, :] < K), other=0.0)  # fmt: skip
        wmask = (rn[None, :] < N) & (kk[:, None] < K)
        w = tl.load(w_base + rn[None, :] * stride_wn + kk[:, None] * stride_wk, mask=wmask, other=0.0)
        acc += _dot(a, w, IEEE)
        if FUSE_SILU:
            wu = tl.load(w_base + (rn[None, :] + N) * stride_wn + kk[:, None] * stride_wk, mask=wmask, other=0.0)
            acc_up += _dot(a, wu, IEEE)
    if FUSE_SILU:
        acc = acc * tl.sigmoid(acc) * acc_up
    tl.store(out_ptr + dst[:, None].to(tl.int64) * stride_om + rn[None, :],
             acc.to(out_ptr.dtype.element_ty), mask=live[:, None] & (rn[None, :] < N))  # fmt: skip


@triton.jit
def gqa_attention_kernel(
    q_ptr, cache_ptr, out_ptr, block_tables_ptr, token_seq_ptr, positions_ptr,
    G, D, HKV, BS, scale,
    stride_q_t, stride_q_h, stride_cache_slot, stride_bt, stride_o_t, stride_o_h,
    BLOCK_G: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr, IEEE: tl.constexpr,
):  # fmt: skip
    """Program (t, kv_head): the G query heads sharing one KV head, for one
    token. Cache rows are [K head 0..HKV-1 | V head 0..HKV-1], D values each.
    Same fixed-chunk reduction as the MLA kernel, hence batch-invariant."""
    t = tl.program_id(0)
    kvh = tl.program_id(1)
    seq = tl.load(token_seq_ptr + t)
    ctx = tl.load(positions_ptr + t) + 1
    g = tl.arange(0, BLOCK_G)
    d = tl.arange(0, BLOCK_D)
    gm = g < G
    heads = kvh * G + g
    q = tl.load(q_ptr + t * stride_q_t + heads[:, None] * stride_q_h + d[None, :],
                mask=gm[:, None] & (d[None, :] < D), other=0.0)  # fmt: skip
    m_i = tl.full((BLOCK_G,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_G,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_G, BLOCK_D), dtype=tl.float32)
    k_off = kvh * D
    v_off = (HKV + kvh) * D
    for start in range(0, ctx, BLOCK_N):
        n = start + tl.arange(0, BLOCK_N)
        valid = n < ctx
        block = tl.load(block_tables_ptr + seq * stride_bt + n // BS, mask=valid, other=0)
        slot = block.to(tl.int64) * BS + n % BS
        kv_mask = valid[:, None] & (d[None, :] < D)
        k = tl.load(cache_ptr + slot[:, None] * stride_cache_slot + k_off + d[None, :], mask=kv_mask, other=0.0)
        v = tl.load(cache_ptr + slot[:, None] * stride_cache_slot + v_off + d[None, :], mask=kv_mask, other=0.0)
        s = _dot(q, tl.trans(k), IEEE)
        s = tl.where(valid[None, :], s * scale, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + _dot(p.to(v.dtype), v, IEEE)
        m_i = m_new
    out = acc / l_i[:, None]
    tl.store(out_ptr + t * stride_o_t + heads[:, None] * stride_o_h + d[None, :],
             out.to(out_ptr.dtype.element_ty), mask=gm[:, None] & (d[None, :] < D))  # fmt: skip
