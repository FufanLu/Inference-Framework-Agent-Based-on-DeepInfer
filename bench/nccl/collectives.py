"""NCCL collective bandwidth, nccl-tests style, through torch.distributed.

For each collective and message size: mean time over `--iters` (CUDA
events, after warm-up), algorithm bandwidth (bytes / time) and bus bandwidth
(nccl-tests' formula, comparable across rank counts):

    all_reduce            busbw = algbw * 2 (n-1) / n
    all_gather, reduce_scatter, all_to_all   algbw * (n-1) / n
    sendrecv (rank 0 -> 1)                   algbw

NCCL_* environment variables pass through, so the Slurm job sweeps
NCCL_ALGO / NCCL_PROTO / NCCL_P2P_DISABLE by rerunning this script.

    torchrun --nproc-per-node 2 bench/nccl/collectives.py --out r/nccl-default.json
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time

import torch
import torch.distributed as dist

OPS = ("all_reduce", "all_gather", "reduce_scatter", "all_to_all", "sendrecv")


def bus_factor(op: str, n: int) -> float:
    if op == "all_reduce":
        return 2 * (n - 1) / n
    if op == "sendrecv":
        return 1.0
    return (n - 1) / n


def make(op: str, numel: int, n: int, device):
    x = torch.ones(numel, dtype=torch.bfloat16, device=device)
    if op == "all_reduce":
        return lambda: dist.all_reduce(x)
    if op == "all_gather":
        out = torch.empty(numel * n, dtype=x.dtype, device=device)
        part = x[: numel // n]
        return lambda: dist.all_gather_into_tensor(out[: part.numel() * n], part)
    if op == "reduce_scatter":
        out = torch.empty(numel // n, dtype=x.dtype, device=device)
        return lambda: dist.reduce_scatter_tensor(out, x[: out.numel() * n])
    if op == "all_to_all":
        out = torch.empty_like(x[: numel // n * n])
        return lambda: dist.all_to_all_single(out, x[: numel // n * n])
    if op == "sendrecv":
        rank = dist.get_rank()

        def run():
            if rank == 0:
                dist.send(x, 1)
            elif rank == 1:
                dist.recv(x, 0)

        return run
    raise ValueError(op)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-bytes", type=int, default=1 << 10)
    ap.add_argument("--max-bytes", type=int, default=1 << 30)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--ops", nargs="+", default=list(OPS))
    args = ap.parse_args()
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    n, rank = dist.get_world_size(), dist.get_rank()
    device = torch.device("cuda", local)
    rows = []
    size = args.min_bytes
    while size <= args.max_bytes:
        numel = size // 2
        for op in args.ops:
            fn = make(op, numel, n, device)
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            dist.barrier()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.iters):
                fn()
            end.record()
            torch.cuda.synchronize()
            t = start.elapsed_time(end) / args.iters / 1e3  # s
            alg = size / t / 1e9
            rows.append({"op": op, "bytes": size, "time_us": t * 1e6, "algbw_GBps": alg,
                         "busbw_GBps": alg * bus_factor(op, n)})  # fmt: skip
            if rank == 0:
                print(f"{op:15s} {size:>12d} B  {t * 1e6:10.1f} us  bus {rows[-1]['busbw_GBps']:8.2f} GB/s", flush=True)
        size *= 4
    if rank == 0:
        env = {k: v for k, v in os.environ.items() if k.startswith("NCCL_")}
        payload = {"world_size": n, "hosts": socket.gethostname(), "nccl_env": env,
                   "nccl_version": ".".join(map(str, torch.cuda.nccl.version())), "torch": torch.__version__,
                   "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "results": rows}  # fmt: skip
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote {args.out}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
