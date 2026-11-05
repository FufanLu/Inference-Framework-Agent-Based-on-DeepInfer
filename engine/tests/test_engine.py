"""End-to-end engine behaviour on the tiny float64 checkpoint."""

from __future__ import annotations

import torch

from lev_engine.config import EngineConfig
from lev_engine.engine import LLMEngine
from lev_engine.hashing import block_hashes
from lev_engine.kv_cache import BlockRemoved, BlockStored
from lev_engine.sequence import SamplingParams

GREEDY = SamplingParams(temperature=0.0, max_tokens=12, ignore_eos=True)


def prompts(n: int, seed: int = 3) -> list[list[int]]:
    g = torch.Generator().manual_seed(seed)
    lengths = torch.randint(1, 40, (n,), generator=g).tolist()
    return [torch.randint(2, 320, (k,), generator=g).tolist() for k in lengths]


def hf_greedy(hf, prompt: list[int], n: int) -> list[int]:
    with torch.no_grad():
        out = hf.generate(torch.tensor([prompt]), max_new_tokens=n, do_sample=False, min_new_tokens=n)
    return out[0, len(prompt) :].tolist()


def engine(path, **kw) -> LLMEngine:
    cfg = dict(block_size=4, num_blocks=256, max_num_batched_tokens=64, max_model_len=256, dtype="float64")
    return LLMEngine(path, EngineConfig(**{**cfg, **kw}))


def test_greedy_matches_transformers_with_chunked_prefill(tiny_v2):
    path, hf = tiny_v2
    ps = prompts(6)
    # A 16-token budget splits most prompts over several steps and mixes
    # prefill chunks with decodes of the other requests.
    outs = engine(path, max_num_batched_tokens=16).generate(ps, GREEDY)
    for prompt, out in zip(ps, outs, strict=True):
        assert out == hf_greedy(hf, prompt, 12)


def test_preemption_recomputes_to_the_same_tokens(tiny_v2):
    path, _ = tiny_v2
    ps = prompts(8, seed=5)
    roomy = engine(path).generate(ps, GREEDY)
    tight = engine(path, num_blocks=24)
    got = tight.generate(ps, GREEDY)
    assert tight.stats.preemptions > 0
    assert got == roomy


def test_prefix_cache_hits_and_events_follow_the_router_convention(tiny_v2):
    from leviathan_worker.hashing import block_hashes as worker_hashes

    path, _ = tiny_v2
    eng = engine(path)
    shared = prompts(1, seed=7)[0] * 3  # long enough for several full blocks
    first = eng.generate([shared + [5, 6]], GREEDY)
    events = eng.drain_events()
    stored = [h for e in events if isinstance(e, BlockStored) for h in e.block_hashes]
    expected = block_hashes(shared + [5, 6], 4)
    assert stored[: len(expected)] == expected
    assert expected == worker_hashes(shared + [5, 6], 4)

    eng.add_request("again", shared + [5, 6], GREEDY)
    out = []
    while eng.has_unfinished:
        for o in eng.step():
            out += o.new_tokens
            if o.num_cached_prompt is not None:
                cached = o.num_cached_prompt
    assert out == first[0]
    assert cached == (len(shared) + 2 - 1) // 4 * 4


def test_blocks_after_a_prefix_hit_are_cached_too(tiny_v2):
    """A request that extends a cached prefix registers (and announces) the
    blocks that follow the hit; the router learns the longer prefix."""
    path, _ = tiny_v2
    eng = engine(path)
    shared = prompts(1, seed=7)[0] * 3
    eng.generate([shared], GREEDY)
    longer = shared + torch.randint(2, 320, (40,), generator=torch.Generator().manual_seed(1)).tolist()
    eng.generate([longer], GREEDY)
    stored = {h for e in eng.drain_events() if isinstance(e, BlockStored) for h in e.block_hashes}
    assert set(block_hashes(longer, 4)) <= stored


def test_eviction_emits_block_removed(tiny_v2):
    path, _ = tiny_v2
    eng = engine(path, num_blocks=12)
    eng.generate(prompts(6, seed=9), GREEDY)
    events = eng.drain_events()
    assert any(isinstance(e, BlockRemoved) for e in events)
    live = set()
    for e in events:
        if isinstance(e, BlockStored):
            live.update(e.block_hashes)
        else:
            live.difference_update(e.block_hashes)
    assert len(live) == eng.pool.num_cached


def test_seeded_sampling_does_not_depend_on_the_batch(tiny_v2):
    path, _ = tiny_v2
    p = SamplingParams(temperature=0.9, top_p=0.9, top_k=50, max_tokens=10, seed=1234, ignore_eos=True)
    target = prompts(1, seed=11)[0]
    alone = engine(path).generate([target], p)[0]
    others = prompts(5, seed=12)
    crowd = engine(path, max_num_batched_tokens=24).generate(
        others[:2] + [target] + others[2:], [SamplingParams(temperature=1.0, seed=s) for s in range(2)] + [p]
        + [SamplingParams(temperature=1.0, seed=s) for s in range(3)]
    )
    assert crowd[2] == alone
    # A different seed gives a different continuation.
    other_seed = SamplingParams(temperature=0.9, top_p=0.9, top_k=50, max_tokens=10, seed=99, ignore_eos=True)
    assert engine(path).generate([target], other_seed)[0] != alone


def test_priority_admits_interactive_requests_first(tiny_v2):
    path, _ = tiny_v2
    eng = engine(path, max_num_seqs=1)
    eng.add_request("batch", prompts(1, seed=13)[0], SamplingParams(temperature=0, max_tokens=2), priority=1)
    eng.add_request("chat", prompts(1, seed=14)[0], SamplingParams(temperature=0, max_tokens=2), priority=0)
    finished = []
    while eng.has_unfinished:
        finished += [o.request_id for o in eng.step() if o.finished]
    assert finished == ["chat", "batch"]
