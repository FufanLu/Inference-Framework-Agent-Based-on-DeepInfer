"""Speculative decoding on real text: speed, acceptance, and output identity.

For each proposer (none, n-gram prompt lookup, a draft model sampling with
the target's noise, the same draft sampling greedily), on two workloads:

  edit   "add comments to this code and return all of it": the answer copies
         its prompt (this repo's own source files), the n-gram case
  chat   open questions: little to copy, the draft-model case

at a few batch sizes and temperatures 0 and 1. Every speculative run is
compared with plain decoding on the same kernels: with batch-invariant
kernels the tokens and logprobs must be identical bit for bit.

    python bench/engine/speculative.py --model $QWEN3_8B --draft $QWEN3_06B --template qwen3 --out r/spec-qwen3.json
    python bench/engine/speculative.py --model $V2_LITE --template v2 --out r/spec-v2lite.json
    # local smoke run on a tiny checkpoint (no tokenizer): --template ids
"""

from __future__ import annotations

import argparse
import gc
import random
import time

import torch

from common import DEVICE, ROOT, sync, tokenizer, write

QUESTIONS = [
    "Explain how a hash map handles collisions, with a short example.",
    "What are the trade-offs between TCP and UDP for a multiplayer game?",
    "Describe the water cycle to a ten-year-old.",
    "Why does a binary search need a sorted array? Give an example.",
    "Summarize the causes of the French Revolution in a paragraph.",
    "How does public-key cryptography let two strangers agree on a secret?",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "What is the difference between a process and a thread?",
    "Give three tips for writing clear technical documentation, with reasons.",
    "Explain gradient descent and why the learning rate matters.",
    "How do vaccines train the immune system?",
    "Compare Python lists and tuples, and say when to use each.",
    "What happens, step by step, when you type a URL into a browser?",
    "Describe how a compiler turns source code into machine code.",
    "Why is the ocean salty but rivers are not?",
    "Explain what a database index is and when it does not help.",
]


def edit_tasks(n: int) -> list[str]:
    """Real source files of this repository, ~50 lines each."""
    files = sorted(p for p in (ROOT / "engine" / "lev_engine").rglob("*.py") if p.stat().st_size > 1500)
    tasks = []
    for i in range(n):
        lines = files[i % len(files)].read_text().splitlines()
        start = 0 if i < len(files) else min(len(lines) - 1, 50 * (i // len(files)))
        code = "\n".join(lines[start : start + 50])
        tasks.append(
            "Add a short comment above every function and class in the following Python code. Return the "
            f"complete code, otherwise unchanged, in one code block.\n\n```python\n{code}\n```"
        )
    return tasks


def render(template: str, tok, text: str, bos: int | None) -> list[int]:
    if template == "qwen3":  # non-thinking mode
        s = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return tok.encode(s, add_special_tokens=False).ids
    if template == "v2":  # as the gateway renders V2 chats
        return [bos] + tok.encode(f"User: {text}\n\nAssistant:", add_special_tokens=False).ids
    raise ValueError(template)


def id_prompts(n: int, vocab: int, seed: int) -> list[list[int]]:
    """Smoke-run stand-in for text: repetitive token-ID prompts."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        chunk = [rng.randrange(2, vocab) for _ in range(rng.randrange(4, 12))]
        out.append(chunk * rng.randrange(1, 4) + [rng.randrange(2, vocab)])
    return out


def build(args, mode: str | None, invariant: bool, coupled: bool = True):
    from lev_engine.config import EngineConfig
    from lev_engine.engine import LLMEngine

    cfg = EngineConfig(
        device=DEVICE, dtype=args.dtype, block_size=16, max_num_batched_tokens=8192, max_num_seqs=max(args.batches),
        max_model_len=4096, kv_cache_memory_bytes=args.kv_gib << 30, kernel_backend=args.kernels,
        batch_invariant=invariant, cuda_graph=args.graphs, cuda_graph_max_batch=max(args.batches),
        speculative=mode, num_speculative_tokens=args.k, draft_model=args.draft if mode == "draft" else "",
        draft_coupled=coupled,
    )  # fmt: skip
    if DEVICE == "cpu":
        cfg.kv_cache_memory_bytes, cfg.num_blocks = 0, 512
    return LLMEngine(args.model, cfg)


def run(engine, prompts: list[list[int]], temperature: float, max_tokens: int) -> dict:
    from lev_engine.sequence import SamplingParams

    engine.pool.reset_prefix_cache()  # same prefill work in every run
    s0 = {k: getattr(engine.stats, k) for k in ("steps", "generated_tokens", "verify_rows", "draft_tokens", "accepted_tokens")}
    ids = [f"r{i}-{time.monotonic_ns()}" for i in range(len(prompts))]
    for i, (rid, p) in enumerate(zip(ids, prompts, strict=True)):
        engine.add_request(rid, p, SamplingParams(temperature=temperature, seed=i, max_tokens=max_tokens, logprobs=True))
    sync()
    t0 = time.perf_counter()
    while engine.has_unfinished:
        engine.step()
    sync()
    seconds = time.perf_counter() - t0
    seqs = [engine.requests.pop(rid) for rid in ids]
    d = {k: getattr(engine.stats, k) - v for k, v in s0.items()}
    row = {
        "seconds": seconds,
        "tokens": d["generated_tokens"],
        "tokens_per_s": d["generated_tokens"] / seconds,
        "steps": d["steps"],
        "verify_rows": d["verify_rows"],
        "draft_tokens": d["draft_tokens"],
        "accepted_tokens": d["accepted_tokens"],
    }
    if d["draft_tokens"]:
        row["acceptance"] = d["accepted_tokens"] / d["draft_tokens"]
        row["tokens_per_verify"] = 1 + d["accepted_tokens"] / d["verify_rows"]
    return row, [(s.output, s.logprobs) for s in seqs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft", default="", help="draft model dir (same tokenizer); omit for n-gram only")
    ap.add_argument("--template", choices=["qwen3", "v2", "ids"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--invariant-batches", type=int, nargs="*", default=[8], help="also run these on invariant kernels")
    ap.add_argument("--temperatures", type=float, nargs="+", default=[0.0, 1.0])
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--kernels", default="triton")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--kv-gib", type=int, default=30)
    ap.add_argument("--no-graphs", dest="graphs", action="store_false")
    args = ap.parse_args()
    if DEVICE == "cpu":
        args.graphs, args.dtype = False, "float32"

    n = max(args.batches)
    if args.template == "ids":
        from lev_engine.config import ModelSpec

        vocab = ModelSpec.from_pretrained(args.model).vocab_size
        workloads = {"edit": id_prompts(n, vocab, 1), "chat": id_prompts(n, vocab, 2)}
    else:
        from lev_engine.config import ModelSpec

        tok, bos = tokenizer(args.model), ModelSpec.from_pretrained(args.model).bos_token_id
        workloads = {
            "edit": [render(args.template, tok, t, bos) for t in edit_tasks(n)],
            "chat": [render(args.template, tok, QUESTIONS[i % len(QUESTIONS)], bos) for i in range(n)],
        }
    modes = [("plain", None, True), ("ngram", "ngram", True)]
    if args.draft:
        modes += [("draft", "draft", True), ("draft-greedy", "draft", False)]

    rows, reference = [], {}
    for invariant in (False, True):
        batches = args.batches if not invariant else [b for b in args.invariant_batches if b <= n]
        if not batches:
            continue
        for name, mode, coupled in modes:
            engine = build(args, mode, invariant, coupled)
            for w in workloads.values():  # warm up: compile every kernel shape the runs use
                run(engine, w[:2], 1.0, 16)
            for wname, prompts in workloads.items():
                for B in batches:
                    for T in args.temperatures:
                        if name == "draft-greedy" and T == 0:
                            continue  # same as coupled under greedy
                        row, outs = run(engine, prompts[:B], T, args.max_tokens)
                        key = (invariant, wname, B, T)
                        if name == "plain":
                            reference[key] = outs
                        else:
                            ref = reference[key]
                            row["same_tokens_vs_plain"] = sum(a[0] == b[0] for a, b in zip(outs, ref)) / B
                            row["same_logprobs_vs_plain"] = sum(a == b for a, b in zip(outs, ref)) / B
                        row.update(mode=name, invariant=invariant, workload=wname, batch=B, temperature=T)
                        rows.append(row)
                        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
            del engine
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    write(args.out, {"model": args.model, "draft": args.draft, "k": args.k, "kernels": args.kernels,
                     "graphs": args.graphs, "max_tokens": args.max_tokens, "results": rows})  # fmt: skip


if __name__ == "__main__":
    main()
