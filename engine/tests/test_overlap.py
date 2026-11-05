"""Overlapped scheduling produces exactly what step-by-step scheduling does."""

from __future__ import annotations

import pytest
import torch

from lev_engine.config import EngineConfig
from lev_engine.engine import LLMEngine
from lev_engine.kv_cache import BlockStored
from lev_engine.sequence import SamplingParams


def run(path, overlap: bool, **cfg):
    base = dict(block_size=4, num_blocks=256, max_num_batched_tokens=24, max_model_len=256, dtype="float64")
    engine = LLMEngine(path, EngineConfig(**{**base, **cfg}, overlap_schedule=overlap))
    g = torch.Generator().manual_seed(30)
    prompts = [torch.randint(2, 320, (n,), generator=g).tolist() for n in (7, 1, 26, 13, 40, 3)]
    for i, p in enumerate(prompts):
        # EOS (token 1) can end some requests early; sampled ones carry logprobs.
        params = SamplingParams(temperature=1.3 if i % 2 else 0.0, seed=i, max_tokens=10 + i, logprobs=bool(i % 2))
        engine.add_request(str(i), p, params, priority=i % 3)
    finished = {}
    while engine.has_unfinished:
        for out in engine.step():
            if out.finished:
                finished[out.request_id] = out.finish_reason
    seqs = {rid: (s.output, s.logprobs, s.finish_reason) for rid, s in engine.requests.items()}
    stored = sorted(h for e in engine.drain_events() if isinstance(e, BlockStored) for h in e.block_hashes)
    return seqs, finished, stored, engine.stats.preemptions


@pytest.mark.parametrize("num_blocks", [256, 22])
def test_overlap_matches_sequential(tiny_v2, num_blocks):
    path, _ = tiny_v2
    a = run(path, False, num_blocks=num_blocks)
    b = run(path, True, num_blocks=num_blocks)
    assert a[1] == b[1]
    for rid, (tokens, logprobs, reason) in a[0].items():
        assert b[0][rid][0] == tokens and b[0][rid][2] == reason
        # Preemption timing differs by a step (blocks are freed when a stop is
        # read back), so recomputed prefixes may run in other batches; the
        # reference kernels are not batch-invariant, and the MoE router's GEMM
        # runs in fp32 (as in HF), its CPU reduction order following the
        # batch shape: up to ~2e-9 drift in CI, same tokens.
        assert b[0][rid][1] == pytest.approx(logprobs, abs=1e-7)
    assert all(t >= 0 for out, _, _ in b[0].values() for t in out)  # no placeholder left
    # Overlap computes the KV of a finished request's last token (its extra
    # step), which can complete one more cacheable block; never fewer.
    assert set(a[2]) <= set(b[2])
    if num_blocks == 22:
        assert a[3] > 0 and b[3] > 0
