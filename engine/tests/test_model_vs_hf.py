"""The engine's absorbed-MLA forward pass against transformers' DeepseekV2.

Both run in float64 on CPU. transformers rounds RoPE and the attention softmax
through fp32, so the tolerance is fp32-sized (a real bug is off by 1e-2 or more):
prompt logits in one step, in uneven chunks, and token by token, all over the
paged cache with a scrambled block table.
"""

from __future__ import annotations

import pytest
import torch

from lev_engine.batch import ForwardBatch
from lev_engine.config import ModelSpec
from lev_engine.models.deepseek_v2 import DeepseekV2
from lev_engine.ops import REFERENCE
from lev_engine.rope import inv_freq
from lev_engine.weights import load_state_dict

BLOCK = 4


def engine_model(path) -> DeepseekV2:
    spec = ModelSpec.from_pretrained(path)
    return DeepseekV2(spec, load_state_dict(path, torch.float64), REFERENCE, max_position=512)


def hf_logits(model, tokens: list[int]) -> torch.Tensor:
    with torch.no_grad():
        return model(torch.tensor([tokens])).logits[0]


def run_chunks(model: DeepseekV2, tokens: list[int], chunks: list[int]) -> torch.Tensor:
    """Feed `tokens` in chunks of the given sizes; return logits of every position."""
    num_blocks = 64
    cache = [torch.zeros(model.kv_cache_shape(num_blocks, BLOCK), dtype=torch.float64) for _ in model.layers]
    # A deliberately non-contiguous block table.
    table = list(range(num_blocks - 1, 0, -3))[: (len(tokens) + BLOCK - 1) // BLOCK]
    out, start = [], 0
    for size in chunks:
        chunk = tokens[start : start + size]
        batch = ForwardBatch.build([chunk], [start], [table], BLOCK, [True])
        batch.logits_indices = torch.arange(len(chunk))  # every position, for the comparison
        out.append(model.forward(batch, cache))
        start += size
    assert start == len(tokens)
    return torch.cat(out)


@pytest.mark.parametrize("fixture", ["tiny_v2", "tiny_v2_qlora"])
def test_prompt_logits_match_transformers(fixture, request):
    path, hf = request.getfixturevalue(fixture)
    model = engine_model(path)
    tokens = torch.randint(2, 320, (37,), generator=torch.Generator().manual_seed(1)).tolist()
    expected = hf_logits(hf, tokens)
    for chunks in ([37], [5, 11, 1, 20], [1] * 37):
        got = run_chunks(model, tokens, chunks)
        torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6, msg=f"chunks {chunks}")


def test_yarn_frequencies_match_transformers(tiny_v2):
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    _, hf = tiny_v2
    spec = ModelSpec.from_pretrained(tiny_v2[0])
    ours, factor = inv_freq(spec.rope, spec.qk_rope_head_dim)
    theirs, their_factor = ROPE_INIT_FUNCTIONS["yarn"](hf.config)
    torch.testing.assert_close(ours, theirs.float())
    assert factor == pytest.approx(their_factor)


def test_batched_sequences_match_single_runs(tiny_v2):
    """Three sequences in one varlen batch give the same logits as alone."""
    path, hf = tiny_v2
    model = engine_model(path)
    g = torch.Generator().manual_seed(2)
    prompts = [torch.randint(2, 320, (n,), generator=g).tolist() for n in (9, 1, 22)]
    cache = [torch.zeros(model.kv_cache_shape(32, BLOCK), dtype=torch.float64) for _ in model.layers]
    tables = [[0, 1, 2], [3], [4, 5, 6, 7, 8, 9]]
    batch = ForwardBatch.build(prompts, [0, 0, 0], tables, BLOCK, [True, True, True])
    got = model.forward(batch, cache)
    for i, prompt in enumerate(prompts):
        torch.testing.assert_close(got[i], hf_logits(hf, prompt)[-1], rtol=1e-5, atol=1e-6)
