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
