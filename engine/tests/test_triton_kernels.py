"""Triton kernels against the reference, and their batch invariance.

Runs wherever Triton is installed; on a CPU-only machine set
TRITON_INTERPRET=1 (the Linux test image and CI do). On a GPU the same tests
check the compiled kernels.
"""

from __future__ import annotations

import os

import pytest
import torch

triton = pytest.importorskip("triton")

from lev_engine.batch import ForwardBatch  # noqa: E402
from lev_engine.ops import reference  # noqa: E402
from lev_engine.ops import triton as tops  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu" and os.environ.get("TRITON_INTERPRET") != "1":
    pytest.skip("no GPU: run with TRITON_INTERPRET=1", allow_module_level=True)

G = torch.Generator(device="cpu").manual_seed(0)


def rand(*shape, dtype=torch.float32):
    return torch.randn(*shape, generator=G).to(device=DEVICE, dtype=dtype)


GPU_ONLY = pytest.mark.skipif(DEVICE == "cpu", reason="the Triton interpreter cannot compute in bf16")


@GPU_ONLY
def test_bf16_kernels_match_reference():
    bf = torch.bfloat16
    x, w, wn = rand(37, 96, dtype=bf), rand(80, 96, dtype=bf), rand(96, dtype=bf)
    assert torch.equal(tops.rms_norm(x, wn, 1e-6), reference.rms_norm(x, wn, 1e-6))
    torch.testing.assert_close(tops.linear_invariant(x, w), reference.linear(x, w), rtol=2e-2, atol=2e-2)
    H, R, P, BS = 16, 512, 64, 16
    cache = rand(64, BS, R + P, dtype=bf)
    batch = paged_batch([[1] * 13, [2], [3] * 5], [0, 300, 9], BS, 64)
    q_lat, q_rope = rand(batch.num_tokens, H, R, dtype=bf), rand(batch.num_tokens, H, P, dtype=bf)
    torch.testing.assert_close(
        tops.mla_attention(q_lat, q_rope, cache, batch, 0.05).float(),
        reference.mla_attention(q_lat, q_rope, cache, batch, 0.05).float(),
        rtol=2e-2, atol=2e-2,
    )  # fmt: skip
    T, k, E = 40, 6, 64
    xm = rand(T, 2048, dtype=bf)
    ids = torch.stack([torch.randperm(E, generator=G)[:k] for _ in range(T)]).to(DEVICE)
    weights = torch.rand(T, k, generator=G).to(DEVICE)
    w13, w2 = rand(E, 2 * 1408, 2048, dtype=bf) * 0.02, rand(E, 2048, 1408, dtype=bf) * 0.02
    torch.testing.assert_close(
        tops.moe_routed(xm, ids, weights, w13, w2).float(), reference.moe_routed(xm, ids, weights, w13, w2).float(),
        rtol=3e-2, atol=3e-2,
    )  # fmt: skip


def test_rms_norm_and_silu_match_reference():
    x, w = rand(37, 96), rand(96)
    torch.testing.assert_close(tops.rms_norm(x, w, 1e-6), reference.rms_norm(x, w, 1e-6))
    gu = rand(37, 2 * 48)
    torch.testing.assert_close(tops.silu_and_mul(gu), reference.silu_and_mul(gu), rtol=1e-5, atol=1e-6)


def test_matmuls_match_reference_and_are_batch_invariant():
    x, w = rand(70, 96), rand(80, 96)
    got = tops.linear_invariant(x, w)
    torch.testing.assert_close(got, reference.linear(x, w), rtol=1e-4, atol=1e-4)
    # Row 5 computed in a batch of 1, of 9 and of 70 is the same, bit for bit.
    for rows in (slice(5, 6), slice(1, 10)):
        assert torch.equal(tops.linear_invariant(x[rows], w)[5 - rows.start], got[5])

    q, wk = rand(29, 4, 16), rand(4, 16, 32)
    torch.testing.assert_close(
        tops.head_matmul_invariant(q, wk, False), reference.head_matmul(q, wk, False), rtol=1e-4, atol=1e-4
    )
    o, wv = rand(29, 4, 32), rand(4, 16, 32)
    torch.testing.assert_close(
        tops.head_matmul_invariant(o, wv, True), reference.head_matmul(o, wv, True), rtol=1e-4, atol=1e-4
    )


def paged_batch(chunks, starts, block_size, num_blocks):
    """Random block tables (disjoint) for sequences with the given new tokens."""
    perm = torch.randperm(num_blocks, generator=G).tolist()
    tables = []
    for chunk, start in zip(chunks, starts, strict=True):
        need = (start + len(chunk) + block_size - 1) // block_size
        tables.append([perm.pop() for _ in range(need)])
    return ForwardBatch.build(chunks, starts, tables, block_size, [True] * len(chunks), DEVICE)


def test_mla_attention_matches_reference_and_is_batch_invariant():
    H, R, P, BS = 4, 32, 8, 4
    cache = rand(64, BS, R + P)
    # A long prefill chunk, a decode at position 70, a short chunk mid-sequence.
    batch = paged_batch([[1] * 13, [2], [3] * 5], [0, 70, 9], BS, 64)
    T = batch.num_tokens
    q_lat, q_rope = rand(T, H, R), rand(T, H, P)
    got = tops.mla_attention(q_lat, q_rope, cache, batch, 0.3)
    torch.testing.assert_close(got, reference.mla_attention(q_lat, q_rope, cache, batch, 0.3), rtol=1e-4, atol=1e-5)

    # The decode token alone, with the same block table: identical bits.
    alone = ForwardBatch.build([[2]], [70], [batch.block_tables[1].tolist()], BS, [True], DEVICE)
    solo = tops.mla_attention(q_lat[13:14], q_rope[13:14], cache, alone, 0.3)
    assert torch.equal(solo[0], got[13])


def test_gqa_attention_matches_reference_and_is_batch_invariant():
    Hq, Hkv, D, BS = 8, 2, 16, 4
    cache = rand(64, BS, 2 * Hkv * D)
    batch = paged_batch([[1] * 11, [2], [3] * 6], [0, 50, 13], BS, 64)
    q = rand(batch.num_tokens, Hq, D)
    got = tops.gqa_attention(q, cache, batch, 0.25, Hkv)
    torch.testing.assert_close(got, reference.gqa_attention(q, cache, batch, 0.25, Hkv), rtol=1e-4, atol=1e-5)
    alone = ForwardBatch.build([[2]], [50], [batch.block_tables[1].tolist()], BS, [True], DEVICE)
    assert torch.equal(tops.gqa_attention(q[11:12], cache, alone, 0.25, Hkv)[0], got[11])


def test_align_by_expert_is_complete_and_homogeneous():
    ids = torch.tensor([[3, 0], [3, 1], [0, 3], [7, 3]], device=DEVICE)
    rows, block_expert = tops.align_by_expert(ids, 8, 4)
    live = rows[rows >= 0].tolist()
    assert sorted(live) == list(range(8))  # every (token, slot) pair exactly once
    flat = ids.reshape(-1).tolist()
    for b, e in enumerate(block_expert.tolist()):
        block = rows[b * 4 : (b + 1) * 4].tolist()
        if e < 0:
            assert all(r == -1 for r in block)
        else:
            assert all(flat[r] == e for r in block if r >= 0)
    assert block_expert.tolist().count(-1) == len(block_expert) - 4  # experts 0, 1, 3, 7


def test_moe_matches_reference():
    T, k, E, hidden, inter = 23, 3, 8, 64, 24
    x = rand(T, hidden)
    ids = torch.stack([torch.randperm(E, generator=G)[:k] for _ in range(T)]).to(DEVICE)
    weights = torch.rand(T, k, generator=G).to(DEVICE)
    w13, w2 = rand(E, 2 * inter, hidden) * 0.1, rand(E, hidden, inter) * 0.1
    got = tops.moe_routed(x, ids, weights, w13, w2)
    torch.testing.assert_close(got, reference.moe_routed(x, ids, weights, w13, w2), rtol=1e-4, atol=1e-5)
    # Token 4 routed alone gives the same bits.
    assert torch.equal(tops.moe_routed(x[4:5], ids[4:5], weights[4:5], w13, w2)[0], got[4])


def test_engine_on_triton_matches_reference_backend(tiny_v2):
    from lev_engine.config import EngineConfig
    from lev_engine.engine import LLMEngine
    from lev_engine.sequence import SamplingParams

    path, _ = tiny_v2
    prompts = [torch.randint(2, 320, (n,), generator=G).tolist() for n in (11, 3, 25)]
    params = SamplingParams(temperature=0.0, max_tokens=6, ignore_eos=True)
    outs = {}
    for backend, invariant in (("reference", False), ("triton", True)):
        cfg = EngineConfig(block_size=4, num_blocks=128, max_num_batched_tokens=16, max_model_len=128,
                           dtype="float32", device=DEVICE, kernel_backend=backend, batch_invariant=invariant)  # fmt: skip
        outs[backend] = LLMEngine(path, cfg).generate(prompts, params)
    assert outs["triton"] == outs["reference"]


def test_speculation_is_bit_for_bit_plain_decoding_on_invariant_kernels(tiny_v2):
    """With batch-invariant kernels a verify row computes exactly what a
    decode row computes, so speculative tokens *and* logprobs equal plain
    decoding's bits, whichever proposer drafts."""
    from lev_engine.config import EngineConfig
    from lev_engine.engine import LLMEngine
    from lev_engine.sequence import SamplingParams

    path, _ = tiny_v2
    g = torch.Generator().manual_seed(21)
    chunk = torch.randint(2, 320, (5,), generator=g).tolist()
    prompts = [chunk * 3, torch.randint(2, 320, (9,), generator=g).tolist(), chunk + [7]]
    params = [SamplingParams(temperature=t, seed=i, max_tokens=10, ignore_eos=True, logprobs=True)
              for i, t in enumerate((0.0, 1.0, 0.7))]  # fmt: skip
    runs, stats = {}, {}
    for mode in (None, "ngram", "draft"):
        cfg = EngineConfig(block_size=4, num_blocks=64, max_num_batched_tokens=32, max_model_len=64, dtype="float32",
                           device=DEVICE, kernel_backend="triton", batch_invariant=True, speculative=mode,
                           draft_model=str(path) if mode == "draft" else "", num_speculative_tokens=3)  # fmt: skip
        engine = LLMEngine(path, cfg)
        for i, (p, sp) in enumerate(zip(prompts, params, strict=True)):
            engine.add_request(str(i), p, sp)
        while engine.has_unfinished:
            engine.step()
        runs[mode] = [(engine.requests[str(i)].output, engine.requests[str(i)].logprobs) for i in range(3)]
        stats[mode] = engine.stats
    assert runs["ngram"] == runs[None]
    assert runs["draft"] == runs[None]
    assert stats["draft"].accepted_tokens == stats["draft"].draft_tokens > 0


def test_the_whole_model_is_batch_invariant(tiny_v2):
    """Per-kernel invariance is not enough: any op outside the kernels (the
    MoE router's GEMM once was one) breaks it. A sequence's logits must be
    the same bits alone and next to others, in prefill and in decode."""
    from lev_engine.batch import ForwardBatch
    from lev_engine.config import ModelSpec
    from lev_engine.models.deepseek_v2 import DeepseekV2
    from lev_engine.ops import get_ops
    from lev_engine.weights import load_state_dict

    path, _ = tiny_v2
    spec = ModelSpec.from_pretrained(path)
    model = DeepseekV2(spec, load_state_dict(path, torch.float32, DEVICE), get_ops("triton", True), 128, DEVICE)
    g = torch.Generator().manual_seed(5)
    prompts = [torch.randint(2, 320, (n,), generator=g).tolist() for n in (13, 4, 29, 7)]

    def logits(idx: list[int], decode: bool) -> torch.Tensor:
        cache = [torch.zeros(model.kv_cache_shape(64, 4), device=DEVICE) for _ in model.layers]
        tables = [list(range(16 * i, 16 * (i + 1))) for i in range(len(idx))]
        ps = [prompts[i] for i in idx]
        out = model.forward(ForwardBatch.build(ps, [0] * len(ps), tables, 4, [True] * len(ps), DEVICE), cache)
        if decode:  # one more token each, its context in the cache
            nxt = [[int(t)] for t in out.argmax(-1).tolist()]
            out = model.forward(ForwardBatch.build(nxt, [len(p) for p in ps], tables, 4, [True] * len(ps), DEVICE), cache)
        return out

    for decode in (False, True):
        crowd = logits([0, 1, 2, 3], decode)
        for i in range(4):
            assert torch.equal(logits([i], decode)[0], crowd[i]), (decode, i)
