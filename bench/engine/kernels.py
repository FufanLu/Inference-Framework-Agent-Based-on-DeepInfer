"""Kernel microbenchmarks on V2-Lite shapes (bf16, H200).

For each kernel: median time (triton.testing.do_bench), achieved TFLOP/s and
GB/s, against the PyTorch path it replaces (cuBLAS, or the reference loop).
Reported next to H200 peaks so the numbers read as a fraction of the roofline.

    python bench/engine/kernels.py --out results/kernels.json
"""

from __future__ import annotations

import argparse

import torch
import triton

from common import write

from lev_engine.batch import ForwardBatch
from lev_engine.ops import reference
from lev_engine.ops import triton as tops

# V2-Lite: hidden 2048, 16 heads, kv_lora 512, rope 64, 64 experts (top-6), moe_inter 1408.
HIDDEN, HEADS, RANK, ROPE, EXPERTS, TOPK, INTER = 2048, 16, 512, 64, 64, 6, 1408
LINEARS = {  # name: (N, K)
    "q_proj": (HEADS * 192, HIDDEN),
    "kv_a_proj": (RANK + ROPE, HIDDEN),
    "o_proj": (HIDDEN, HEADS * 128),
    "lm_head": (102400, HIDDEN),
}
DT = torch.bfloat16


def bench(fn) -> float:
    return triton.testing.do_bench(fn, warmup=25, rep=100)  # ms


def gemm_rows():
    rows = []
    for name, (N, K) in LINEARS.items():
        w = torch.randn(N, K, device="cuda", dtype=DT)
        for M in (1, 16, 64, 256, 1024, 4096):
            x = torch.randn(M, K, device="cuda", dtype=DT)
            flops = 2 * M * N * K
            byts = (M * K + N * K + M * N) * 2
            for impl, fn in (("cuBLAS", lambda: reference.linear(x, w)), ("triton-invariant", lambda: tops.linear_invariant(x, w))):
                ms = bench(fn)
                rows.append({"kernel": "linear", "shape": name, "M": M, "N": N, "K": K, "impl": impl, "ms": ms,
                             "tflops": flops / ms / 1e9, "gbps": byts / ms / 1e6})  # fmt: skip
            print(rows[-2], rows[-1], flush=True)
    return rows


def attention_rows():
    rows = []
    BS = 16
    for batch, ctx in ((1, 1024), (1, 8192), (16, 1024), (64, 1024), (64, 4096), (256, 1024)):
        blocks = (ctx + BS - 1) // BS
        cache = torch.randn(batch * blocks, BS, RANK + ROPE, device="cuda", dtype=DT)
        tables = [list(range(b * blocks, (b + 1) * blocks)) for b in range(batch)]
        fb = ForwardBatch.build([[1]] * batch, [ctx - 1] * batch, tables, BS, [True] * batch, "cuda")
        q_lat = torch.randn(batch, HEADS, RANK, device="cuda", dtype=DT)
        q_rope = torch.randn(batch, HEADS, ROPE, device="cuda", dtype=DT)
        byts = batch * ctx * (RANK + ROPE) * 2  # the cache is read once per decode token
        flops = batch * HEADS * ctx * 2 * (RANK + ROPE + RANK)
        for impl, fn in (
            ("reference (gather + einsum)", lambda: reference.mla_attention(q_lat, q_rope, cache, fb, 0.1)),
            ("triton", lambda: tops.mla_attention(q_lat, q_rope, cache, fb, 0.1)),
        ):
            ms = bench(fn)
            rows.append({"kernel": "mla_decode", "batch": batch, "context": ctx, "impl": impl, "ms": ms,
                         "gbps": byts / ms / 1e6, "tflops": flops / ms / 1e9})  # fmt: skip
            print(rows[-1], flush=True)
    return rows


def moe_rows():
    rows = []
    w13 = torch.randn(EXPERTS, 2 * INTER, HIDDEN, device="cuda", dtype=DT) * 0.02
    w2 = torch.randn(EXPERTS, HIDDEN, INTER, device="cuda", dtype=DT) * 0.02
    for tokens in (1, 16, 256, 2048, 8192):
        x = torch.randn(tokens, HIDDEN, device="cuda", dtype=DT)
        ids = torch.stack([torch.randperm(EXPERTS, device="cuda")[:TOPK] for _ in range(tokens)])
        weights = torch.rand(tokens, TOPK, device="cuda")
        flops = tokens * TOPK * 2 * HIDDEN * 3 * INTER
        for impl, fn in (
            ("reference (per-expert loop)", lambda: reference.moe_routed(x, ids, weights, w13, w2)),
            ("triton grouped GEMM", lambda: tops.moe_routed(x, ids, weights, w13, w2)),
        ):
            ms = bench(fn)
            rows.append({"kernel": "moe", "tokens": tokens, "impl": impl, "ms": ms, "tflops": flops / ms / 1e9})
            print(rows[-1], flush=True)
    return rows


def norm_rows():
    rows = []
    w = torch.randn(HIDDEN, device="cuda", dtype=DT)
    for M in (1, 256, 8192):
        x = torch.randn(M, HIDDEN, device="cuda", dtype=DT)
        byts = 2 * M * HIDDEN * 2
        for impl, fn in (("torch", lambda: reference.rms_norm(x, w, 1e-6)), ("triton", lambda: tops.rms_norm(x, w, 1e-6))):
            ms = bench(fn)
            rows.append({"kernel": "rms_norm", "M": M, "impl": impl, "ms": ms, "gbps": byts / ms / 1e6})
            print(rows[-1], flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    results = {"norm": norm_rows(), "linear": gemm_rows(), "mla_decode": attention_rows(), "moe": moe_rows()}
    write(args.out, {"dtype": "bfloat16", "peaks": {"note": "H200 SXM datasheet: ~989 dense BF16 TFLOP/s, 4.8 TB/s HBM3e"},
                     "results": results})  # fmt: skip


if __name__ == "__main__":
    main()
