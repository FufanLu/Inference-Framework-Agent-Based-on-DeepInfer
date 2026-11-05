"""Qwen3 (GQA, q/k norm, half-rotation RoPE, tied embeddings) against transformers."""

from __future__ import annotations

import torch

from lev_engine.batch import ForwardBatch
from lev_engine.config import EngineConfig, ModelSpec
from lev_engine.engine import LLMEngine
from lev_engine.models.qwen3 import Qwen3
from lev_engine.ops import REFERENCE
from lev_engine.sequence import SamplingParams
from lev_engine.weights import load_state_dict

BLOCK = 4


def test_prompt_logits_match_transformers(tiny_qwen3):
    path, hf = tiny_qwen3
    spec = ModelSpec.from_pretrained(path)
    assert spec.kv_cache_dim == 2 * 2 * 16 and spec.tie_word_embeddings
    model = Qwen3(spec, load_state_dict(path, torch.float64), REFERENCE, max_position=512)
    tokens = torch.randint(2, 320, (30,), generator=torch.Generator().manual_seed(1)).tolist()
    with torch.no_grad():
        expected = hf(torch.tensor([tokens])).logits[0]
    for chunks in ([30], [7, 1, 22], [1] * 30):
        cache = [torch.zeros(model.kv_cache_shape(40, BLOCK), dtype=torch.float64) for _ in model.layers]
        table = list(range(39, 0, -4))[:8]
        got, start = [], 0
        for size in chunks:
            batch = ForwardBatch.build([tokens[start : start + size]], [start], [table], BLOCK, [True])
            batch.logits_indices = torch.arange(size)
            got.append(model.forward(batch, cache))
            start += size
        torch.testing.assert_close(torch.cat(got), expected, rtol=1e-5, atol=1e-6, msg=f"chunks {chunks}")


def test_greedy_generation_matches_transformers(tiny_qwen3):
    path, hf = tiny_qwen3
    g = torch.Generator().manual_seed(4)
    prompts = [torch.randint(2, 320, (n,), generator=g).tolist() for n in (5, 17, 31)]
    engine = LLMEngine(path, EngineConfig(block_size=4, num_blocks=128, max_num_batched_tokens=16, max_model_len=256,
                                          dtype="float64"))  # fmt: skip
    outs = engine.generate(prompts, SamplingParams(temperature=0.0, max_tokens=10, ignore_eos=True))
    for prompt, out in zip(prompts, outs, strict=True):
        with torch.no_grad():
            ref = hf.generate(torch.tensor([prompt]), max_new_tokens=10, do_sample=False,
                              eos_token_id=None, pad_token_id=0)  # fmt: skip
        assert out == ref[0, len(prompt) :].tolist()
