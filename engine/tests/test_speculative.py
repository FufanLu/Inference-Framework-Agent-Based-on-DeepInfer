"""Speculative decoding returns exactly what plain decoding returns."""

from __future__ import annotations

import pytest
import torch

from lev_engine.config import EngineConfig
from lev_engine.engine import LLMEngine
from lev_engine.kv_cache import BlockStored
from lev_engine.sequence import SamplingParams
from lev_engine.speculative import _NgramIndex, verify


def test_verify_keeps_the_target_samples_up_to_the_first_miss():
    # t_0 is always kept; t_{i+1} only while draft d_{i+1} == t_i.
    assert verify([4, 9, 9], [5, 6]) == [4]
    assert verify([5, 6, 7], [5, 8]) == [5, 6]
    assert verify([5, 6, 7], [5, 6]) == [5, 6, 7]  # all accepted, plus the bonus
    assert verify([3], []) == [3]


def test_ngram_lookup_takes_the_latest_earlier_occurrence():
    index = _NgramIndex(2, 3)
    tokens = [1, 2, 3, 9, 1, 2, 3, 7, 1, 2, 3]
    index.extend(tokens)
    assert index.lookup(tokens, 3) == [7, 1, 2]  # not the first match, not the suffix itself
    tokens += [4, 4]
    index.extend(tokens)
    assert index.lookup(tokens, 3) == []  # "4 4" and "3 4 4" never occurred before
    tokens += [8, 2, 3]
    index.extend(tokens)
    # No earlier "8 2 3"; the 2-gram "2 3" last occurred before "4 4".
    assert index.lookup(tokens, 2) == [4, 4]


def prompts() -> list[list[int]]:
    g = torch.Generator().manual_seed(40)
    out = []
    for n, reps in ((6, 3), (1, 1), (9, 2), (13, 1), (5, 4), (3, 2)):
        chunk = torch.randint(2, 320, (n,), generator=g).tolist()
        out.append(chunk * reps + torch.randint(2, 320, (2,), generator=g).tolist())
    return out


def run(path, **cfg):
    base = dict(block_size=4, num_blocks=256, max_num_batched_tokens=24, max_model_len=80, dtype="float64")
    engine = LLMEngine(path, EngineConfig(**{**base, **cfg}))
    for i, p in enumerate(prompts()):
        # Greedy and sampled requests; EOS (token 1) may stop some early; one
        # runs to the context limit (lookahead capped at max_model_len).
        params = SamplingParams(temperature=1.3 if i % 2 else 0.0, top_k=40 if i == 3 else 0, seed=i,
                                max_tokens=0 if i == 4 else 12 + 3 * i, logprobs=bool(i % 2), ignore_eos=i == 4)  # fmt: skip
        engine.add_request(str(i), p, params, priority=i % 3)
    finished, chunks = {}, {}
    while engine.has_unfinished:
        for out in engine.step():
            chunks.setdefault(out.request_id, []).extend(out.new_tokens)
            if out.finished:
                finished[out.request_id] = out.finish_reason
    seqs = {rid: (s.output, s.logprobs, s.finish_reason) for rid, s in engine.requests.items()}
    assert all(chunks[rid] == s.output for rid, s in engine.requests.items())  # what streams out
    stored = sorted(h for e in engine.drain_events() if isinstance(e, BlockStored) for h in e.block_hashes)
    return seqs, finished, stored, engine.stats


def same_outputs(plain, spec) -> None:
    assert spec[1] == plain[1]
    for rid, (tokens, logprobs, reason) in plain[0].items():
        assert spec[0][rid][0] == tokens and spec[0][rid][2] == reason
        # Verify batches differ in shape from decode batches and the reference
        # kernels are not batch-invariant; the MoE router runs in fp32 (as in
        # HF), whose CPU GEMM order follows the batch shape: ~1e-9 drift.
        # Bitwise equality with the invariant kernels: test_triton_kernels.
        assert spec[0][rid][1] == pytest.approx(logprobs, abs=1e-7)
    # A stop on an accepted draft leaves that token's KV computed, which can
    # complete one more cacheable block; never fewer.
    assert set(plain[2]) <= set(spec[2])


@pytest.mark.parametrize("num_blocks", [256, 22])
def test_ngram_speculation_matches_plain_decoding(tiny_v2, num_blocks):
    path, _ = tiny_v2
    plain = run(path, num_blocks=num_blocks)
    spec = run(path, num_blocks=num_blocks, speculative="ngram", num_speculative_tokens=3)
    same_outputs(plain, spec)
    s = spec[3]
    assert s.draft_tokens > 0 and 0 < s.accepted_tokens < s.draft_tokens
    assert s.steps < plain[3].steps  # accepted drafts save steps
    if num_blocks == 22:
        assert plain[3].preemptions > 0 and s.preemptions > 0


def test_a_draft_identical_to_the_target_is_always_accepted(tiny_v2):
    """Same weights as the target and the same noise: every draft is the
    token the target picks, sampled requests (T=1.3) included."""
    path, _ = tiny_v2
    plain = run(path)
    spec = run(path, speculative="draft", draft_model=str(path), num_speculative_tokens=4)
    same_outputs(plain, spec)
    s = spec[3]
    assert s.draft_tokens > 50 and s.accepted_tokens == s.draft_tokens
    # Drafting greedily instead loses the sampled requests' drafts.
    greedy = run(path, speculative="draft", draft_model=str(path), num_speculative_tokens=4, draft_coupled=False)
    same_outputs(plain, greedy)
    assert greedy[3].accepted_tokens < greedy[3].draft_tokens


def test_an_unrelated_draft_costs_speed_not_correctness(tiny_v2, tiny_qwen3):
    """A random Qwen3 drafting for the DeepSeek-V2 target (same vocabulary
    size, nothing else in common), with preemption in the mix."""
    path, _ = tiny_v2
    plain = run(path, num_blocks=22)
    spec = run(path, num_blocks=22, speculative="draft", draft_model=str(tiny_qwen3[0]), num_speculative_tokens=4)
    same_outputs(plain, spec)
    assert spec[3].preemptions > 0
    assert spec[3].accepted_tokens < spec[3].draft_tokens / 2


def test_speculation_rejects_overlap_scheduling(tiny_v2):
    path, _ = tiny_v2
    with pytest.raises(ValueError, match="overlap"):
        LLMEngine(path, EngineConfig(num_blocks=16, speculative="ngram", overlap_schedule=True, dtype="float64"))
