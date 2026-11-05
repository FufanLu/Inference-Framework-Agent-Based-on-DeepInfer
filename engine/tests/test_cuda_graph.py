"""Decode from CUDA graphs gives the same tokens and logits as eager (GPU only)."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA graphs need a GPU", allow_module_level=True)

from lev_engine.config import EngineConfig  # noqa: E402
from lev_engine.engine import LLMEngine  # noqa: E402
from lev_engine.sequence import SamplingParams  # noqa: E402


def test_graph_decode_matches_eager(tiny_v2):
    path, _ = tiny_v2
    g = torch.Generator().manual_seed(8)
    prompts = [torch.randint(2, 320, (n,), generator=g).tolist() for n in (3, 17, 9, 30, 1)]
    params = SamplingParams(temperature=0.7, seed=5, max_tokens=12, ignore_eos=True, logprobs=True)
    runs = {}
    for graphs in (False, True):
        cfg = EngineConfig(block_size=4, num_blocks=256, max_num_batched_tokens=64, max_model_len=128,
                           device="cuda", dtype="float32", kernel_backend="triton", batch_invariant=True,
                           cuda_graph=graphs, cuda_graph_max_batch=8)  # fmt: skip
        engine = LLMEngine(path, cfg)
        for i, p in enumerate(prompts):
            engine.add_request(str(i), p, params)
        while engine.has_unfinished:
            engine.step()
        runs[graphs] = [(engine.requests[str(i)].output, engine.requests[str(i)].logprobs) for i in range(len(prompts))]
    assert runs[True] == runs[False]  # bit for bit: same kernels, same reduction order


@pytest.mark.parametrize("mode", ["ngram", "draft"])
def test_speculative_verify_graphs_match_plain_eager_decoding(tiny_v2, mode):
    """Verify steps replayed from k+1-token graphs (drafts padded to k) and
    draft steps from the draft model's graphs: still plain decoding's bits."""
    path, _ = tiny_v2
    g = torch.Generator().manual_seed(9)
    chunk = torch.randint(2, 320, (6,), generator=g).tolist()
    prompts = [chunk * 3, torch.randint(2, 320, (11,), generator=g).tolist(), chunk + [5, 9], [3]]
    params = SamplingParams(temperature=0.8, seed=2, max_tokens=14, ignore_eos=True, logprobs=True)
    runs = {}
    for spec in (None, mode):
        cfg = EngineConfig(block_size=4, num_blocks=256, max_num_batched_tokens=64, max_model_len=128,
                           device="cuda", dtype="float32", kernel_backend="triton", batch_invariant=True,
                           cuda_graph=spec is not None, cuda_graph_max_batch=8, speculative=spec,
                           draft_model=str(path) if spec == "draft" else "", num_speculative_tokens=3)  # fmt: skip
        engine = LLMEngine(path, cfg)
        for i, p in enumerate(prompts):
            engine.add_request(str(i), p, params)
        while engine.has_unfinished:
            engine.step()
        runs[spec] = [(engine.requests[str(i)].output, engine.requests[str(i)].logprobs) for i in range(len(prompts))]
    assert runs[mode] == runs[None]
