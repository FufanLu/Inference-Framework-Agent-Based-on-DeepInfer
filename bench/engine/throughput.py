"""Offline throughput and single-stream decode speed: lev-engine vs vLLM.

Same checkpoint, same GPU, same token-ID prompts (random, so no prefix
reuse), fixed output length. Each engine runs in its own process
(`--engine`), so GPU memory and CUDA state never mix.

    python bench/engine/throughput.py --engine lev:triton --model $M --out r/lev-triton.json
    python bench/engine/throughput.py --engine vllm --model $M --out r/vllm.json
"""

from __future__ import annotations

import argparse
import time

import torch

from common import DEVICE, random_prompts, sync, write

WORKLOADS = {
    # name: (requests, prompt tokens, output tokens)
    "batch-256x512x256": (256, 512, 256),
    "single-1x512x256": (1, 512, 256),
}


def run_lev(args, kernels: str):
    from lev_engine.config import EngineConfig
    from lev_engine.engine import LLMEngine
    from lev_engine.sequence import SamplingParams

    flags = set(kernels.split("+")[1:])
    kernels = kernels.split("+")[0]
    graphs, overlap = "graph" in flags, "overlap" in flags
    backend, invariant = {"reference": ("reference", False), "triton": ("triton", False),
                          "triton-invariant": ("triton", True)}[kernels]  # fmt: skip
    engine = LLMEngine(
        args.model,
        EngineConfig(
            device=DEVICE, dtype="bfloat16", block_size=16, max_num_batched_tokens=8192, max_num_seqs=256,
            max_model_len=4096, kv_cache_memory_bytes=40 << 30, kernel_backend=backend, batch_invariant=invariant,
            cuda_graph=graphs, overlap_schedule=overlap,
        ),  # fmt: skip
    )
    out = {}
    for name, (n, plen, olen) in WORKLOADS.items():
        prompts = random_prompts(n, plen, engine.spec.vocab_size, seed=1)
        params = SamplingParams(temperature=0.0, max_tokens=olen, ignore_eos=True)
        engine.generate(prompts[:1], SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True))  # warm up
        engine.pool.reset_prefix_cache()
        steps0, sched0, fwd0 = engine.stats.steps, 0.0, 0.0
        for rid, p in enumerate(prompts):
            engine.add_request(f"{name}-{rid}", p, params)
        sync()
        start = time.perf_counter()
        first_token = None
        while engine.has_unfinished:
            outputs = engine.step()
            if outputs and first_token is None:
                first_token = time.perf_counter()
            sched0 += engine.stats.last.schedule_s
            fwd0 += engine.stats.last.forward_s
        sync()
        wall = time.perf_counter() - start
        for rid in range(n):
            engine.requests.pop(f"{name}-{rid}", None)
        out[name] = {
            "requests": n, "prompt_tokens": plen, "output_tokens": olen, "wall_s": wall,
            "output_tokens_per_s": n * olen / wall, "total_tokens_per_s": n * (plen + olen) / wall,
            "steps": engine.stats.steps - steps0, "schedule_s": sched0, "forward_and_sample_s": fwd0,
            "time_to_first_token_s": (first_token - start) if first_token else None,
            "decode_ms_per_token": (wall - (first_token - start)) * 1e3 / max(olen - 1, 1) if n == 1 else None,
        }  # fmt: skip
        print(kernels, name, {k: round(v, 3) if isinstance(v, float) else v for k, v in out[name].items()}, flush=True)
        engine.pool.reset_prefix_cache()
    return out


def run_vllm(args):
    from vllm import LLM, SamplingParams, TokensPrompt

    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.85,
              enable_prefix_caching=False, trust_remote_code=True)  # fmt: skip
    out = {}
    vocab = llm.llm_engine.model_config.get_vocab_size()
    for name, (n, plen, olen) in WORKLOADS.items():
        prompts = [TokensPrompt(prompt_token_ids=p) for p in random_prompts(n, plen, vocab, seed=1)]
        params = SamplingParams(temperature=0.0, max_tokens=olen, ignore_eos=True)
        llm.generate(prompts[:1], SamplingParams(max_tokens=8, ignore_eos=True), use_tqdm=False)
        sync()
        start = time.perf_counter()
        results = llm.generate(prompts, params, use_tqdm=False)
        sync()
        wall = time.perf_counter() - start
        produced = sum(len(r.outputs[0].token_ids) for r in results)
        entry = {
            "requests": n, "prompt_tokens": plen, "output_tokens": olen, "wall_s": wall,
            "output_tokens_per_s": produced / wall, "total_tokens_per_s": (n * plen + produced) / wall,
        }  # fmt: skip
        m = results[0].metrics
        if n == 1 and m is not None and getattr(m, "first_token_time", None) and getattr(m, "last_token_time", None):
            entry["decode_ms_per_token"] = (m.last_token_time - m.first_token_time) * 1e3 / max(olen - 1, 1)
        out[name] = entry
        print("vllm", name, {k: round(v, 3) if isinstance(v, float) else v for k, v in entry.items()}, flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True,
                    help="vllm | lev:reference | lev:triton | lev:triton-invariant, with +graph (CUDA graphs) and/or +overlap (overlapped scheduling)")  # fmt: skip
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.engine == "vllm":
        results = run_vllm(args)
    else:
        results = run_lev(args, args.engine.split(":", 1)[1])
    write(args.out, {"engine": args.engine, "model": args.model, "workloads": results})


if __name__ == "__main__":
    main()
