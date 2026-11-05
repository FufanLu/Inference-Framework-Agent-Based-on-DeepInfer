"""Does a request's output depend on what else is in the batch?

One target request (seeded sampling, logprobs) runs alone and then inside
batches of 1, 4, 16 and 64 other requests with different prompts and
lengths, under each kernel configuration. We compare the target's tokens and
logprob bits against its solo run.

Expected: batch-invariant Triton is identical everywhere; cuBLAS GEMMs change
reduction strategy with M, so logprobs drift and, eventually, tokens diverge.

    python bench/engine/invariance.py --model $MODEL_DIR --out results/invariance.json
"""

from __future__ import annotations

import argparse
import struct

import torch

from common import DEVICE, TEXTS, random_prompts, tokenizer, write

from lev_engine.config import EngineConfig
from lev_engine.engine import LLMEngine
from lev_engine.sequence import SamplingParams

CONFIGS = {
    "reference (cuBLAS)": dict(kernel_backend="reference", batch_invariant=False),
    "triton (cuBLAS GEMMs)": dict(kernel_backend="triton", batch_invariant=False),
    "triton batch-invariant": dict(kernel_backend="triton", batch_invariant=True),
}
CROWDS = [0, 1, 4, 16, 64]


def vocab(model: str) -> int:
    from lev_engine.config import ModelSpec

    return ModelSpec.from_pretrained(model).vocab_size


def engine_bos(model: str) -> int:
    from lev_engine.config import ModelSpec

    return ModelSpec.from_pretrained(model).bos_token_id


def bits(values: list[float]) -> list[int]:
    return [struct.unpack("<Q", struct.pack("<d", v))[0] for v in values]


def run(engine: LLMEngine, target: list[int], params: SamplingParams, crowd: list[list[int]]):
    engine.add_request("target", target, params)
    for i, p in enumerate(crowd):
        engine.add_request(f"crowd-{i}", p, SamplingParams(temperature=1.0, seed=i, max_tokens=params.max_tokens))
    while engine.has_unfinished:
        engine.step()
    seq = engine.requests.pop("target")
    for i in range(len(crowd)):
        engine.requests.pop(f"crowd-{i}")
    engine.pool.reset_prefix_cache()  # every run computes its prefill
    return seq.output, seq.logprobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--crowds", type=int, nargs="+", default=CROWDS)
    args = ap.parse_args()
    tok = tokenizer(args.model)
    target = [engine_bos(args.model)] + tok.encode(TEXTS[5]).ids
    params = SamplingParams(temperature=0.8, top_p=0.95, seed=2026, max_tokens=args.tokens, ignore_eos=True, logprobs=True)
    crowd_all = [p[: 50 + 37 * i % 400] for i, p in enumerate(random_prompts(64, 450, vocab(args.model), seed=9))]
    results = {}
    for name, kernels in CONFIGS.items():
        engine = LLMEngine(
            args.model,
            EngineConfig(
                device=DEVICE, dtype="bfloat16", block_size=16, num_blocks=4096, max_num_batched_tokens=4096,
                max_num_seqs=128, max_model_len=4096, **kernels,
            ),  # fmt: skip
        )
        solo_tokens, solo_lp = run(engine, target, params, [])
        rows = []
        for n in args.crowds:
            tokens, lp = run(engine, target, params, crowd_all[:n])
            first = next((i for i, (a, b) in enumerate(zip(tokens, solo_tokens)) if a != b), None)
            same_lp = sum(a == b for a, b in zip(bits(lp), bits(solo_lp)))
            rows.append({
                "crowd": n,
                "tokens_identical": tokens == solo_tokens,
                "first_divergent_token": first,
                "logprobs_bitwise_equal": same_lp,
                "logprobs_compared": min(len(lp), len(solo_lp)),
                "max_logprob_diff": max((abs(a - b) for a, b in zip(lp, solo_lp)), default=0.0),
            })  # fmt: skip
            print(name, rows[-1], flush=True)
        results[name] = rows
        del engine
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    write(args.out, {"model": args.model, "tokens": args.tokens, "target_prompt_tokens": len(target),
                     "results": results})  # fmt: skip


if __name__ == "__main__":
    main()
