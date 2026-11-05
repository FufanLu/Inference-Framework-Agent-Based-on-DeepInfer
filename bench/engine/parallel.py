"""DeepSeek-V2-Lite step times on one GPU, TP2 and EP2 (DP attention).

Every rank drives `DeepseekV2.forward` directly (same code as the engine):
a prefill of the whole batch, then decode steps, timed with CUDA events.
The global batch is B sequences:

    single  1 GPU,  B sequences
    tp      2 GPUs, the same B sequences on both (heads / intermediate split)
    ep      2 GPUs, B/2 sequences each, routed experts split, all_to_all per MoE layer

One decode step is profiled to split GPU time into NCCL and the rest.

    python bench/engine/parallel.py --mode single --model $M --out r/single.json
    torchrun --nproc-per-node 2 bench/engine/parallel.py --mode tp --model $M --out r/tp.json
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist

import time

from common import DEVICE, random_prompts, write

from lev_engine.batch import ForwardBatch
from lev_engine.config import ModelSpec
from lev_engine.models.deepseek_v2 import DeepseekV2
from lev_engine.ops import get_ops
from lev_engine.parallel import Parallel
from lev_engine.weights import load_state_dict

BS = 16


class Runner:
    def __init__(self, model: DeepseekV2, prompts: list[list[int]], max_len: int):
        self.model = model
        self.lens = [len(p) for p in prompts]
        per_seq = (max_len + BS - 1) // BS
        self.tables = [list(range(i * per_seq, (i + 1) * per_seq)) for i in range(len(prompts))]
        shape = model.kv_cache_shape(per_seq * len(prompts), BS)
        self.cache = [torch.zeros(shape, dtype=model.dtype, device=DEVICE) for _ in model.layers]
        self.prompts = prompts

    def prefill(self) -> torch.Tensor:
        b = ForwardBatch.build(self.prompts, [0] * len(self.prompts), self.tables, BS, [True] * len(self.prompts), DEVICE)
        return self.model.forward(b, self.cache).argmax(-1)

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        ids = tokens.tolist()
        b = ForwardBatch.build([[t] for t in ids], self.lens, self.tables, BS, [True] * len(ids), DEVICE)
        self.lens = [n + 1 for n in self.lens]
        return self.model.forward(b, self.cache).argmax(-1)


def timed(fn, *args):
    if DEVICE == "cpu":
        t = time.perf_counter()
        out = fn(*args)
        return out, (time.perf_counter() - t) * 1e3
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn(*args)
    end.record()
    torch.cuda.synchronize()
    return out, start.elapsed_time(end)


def nccl_share(runner: Runner, tokens: torch.Tensor) -> dict:
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        runner.decode(tokens)
        torch.cuda.synchronize()
    total = nccl = 0.0
    for e in prof.key_averages():
        t = getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)
        total += t
        if "nccl" in e.key.lower():
            nccl += t
    return {"gpu_time_us": total, "nccl_us": nccl, "nccl_share": nccl / total if total else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "tp", "ep"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", type=int, nargs="+", default=[2, 16, 64, 128])
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--decode-steps", type=int, default=32)
    args = ap.parse_args()
    par = None
    rank, world = 0, 1
    if args.mode != "single":
        local = int(os.environ["LOCAL_RANK"])
        if DEVICE == "cpu":
            dist.init_process_group("gloo")
        else:
            torch.cuda.set_device(local)
            dist.init_process_group("nccl", device_id=torch.device("cuda", local))
        rank, world = dist.get_rank(), dist.get_world_size()
        par = Parallel.from_env(args.mode)
    spec = ModelSpec.from_pretrained(args.model)
    dtype = torch.bfloat16 if DEVICE != "cpu" else torch.float32
    weights = load_state_dict(args.model, dtype, DEVICE)
    model = DeepseekV2(spec, weights, get_ops("triton" if DEVICE != "cpu" else "reference", False), 4096, DEVICE, par)
    del weights
    rows = []
    for B in args.batches:
        everyone = random_prompts(B, args.prompt, spec.vocab_size, seed=B)
        mine = everyone[rank::world] if args.mode == "ep" else everyone
        runner = Runner(model, mine, args.prompt + args.decode_steps + 8)
        tokens, prefill_ms = timed(runner.prefill)
        tokens, _ = timed(runner.decode, tokens)  # warm-up step
        times = []
        for _ in range(args.decode_steps - 2):
            tokens, ms = timed(runner.decode, tokens)
            times.append(ms)
        times.sort()
        share = nccl_share(runner, tokens) if par is not None and DEVICE != "cpu" else {}
        row = {"global_batch": B, "local_batch": len(mine), "prompt": args.prompt, "prefill_ms": prefill_ms,
               "decode_step_ms_p50": times[len(times) // 2], "decode_tokens_per_s": B / (times[len(times) // 2] / 1e3),
               "prefill_tokens_per_s": B * args.prompt / (prefill_ms / 1e3), **share}  # fmt: skip
        rows.append(row)
        if rank == 0:
            print(args.mode, {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
        del runner
        if DEVICE != "cpu":
            torch.cuda.empty_cache()
    if rank == 0:
        write(args.out, {"mode": args.mode, "world_size": world, "model": args.model, "results": rows})
    if par is not None:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
