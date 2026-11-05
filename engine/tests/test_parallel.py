"""TP2 and EP2 (DP attention) against one process, on CPU with gloo, float64."""

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from lev_engine.batch import ForwardBatch
from lev_engine.config import ModelSpec
from lev_engine.models.deepseek_v2 import DeepseekV2
from lev_engine.ops import REFERENCE
from lev_engine.parallel import Parallel
from lev_engine.weights import load_state_dict

BLOCK = 4


def prompts() -> list[list[int]]:
    g = torch.Generator().manual_seed(21)
    return [torch.randint(2, 320, (n,), generator=g).tolist() for n in (13, 5, 22, 9)]


def last_logits(model: DeepseekV2, batch_prompts: list[list[int]]) -> torch.Tensor:
    tables, nxt = [], 0
    for p in batch_prompts:
        n = (len(p) + BLOCK - 1) // BLOCK
        tables.append(list(range(nxt, nxt + n)))
        nxt += n
    cache = [torch.zeros(model.kv_cache_shape(max(nxt, 1), BLOCK), dtype=torch.float64) for _ in model.layers]
    batch = ForwardBatch.build(batch_prompts, [0] * len(batch_prompts), tables, BLOCK, [True] * len(batch_prompts))
    return model.forward(batch, cache)


def worker(rank: int, world: int, port: int, path: str, mode: str, out: str) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        spec = ModelSpec.from_pretrained(path)
        model = DeepseekV2(spec, load_state_dict(path, torch.float64), REFERENCE, 512, "cpu", Parallel.from_env(mode))
        ps = prompts()
        mine = ps if mode == "tp" else ps[rank::world]  # EP: each rank serves its own requests
        torch.save(last_logits(model, mine), f"{out}/{mode}-{rank}.pt")
    finally:
        dist.destroy_process_group()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("mode", ["tp", "ep"])
def test_parallel_matches_one_process(mode, tiny_v2, tmp_path):
    path, _ = tiny_v2
    spec = ModelSpec.from_pretrained(path)
    single = last_logits(DeepseekV2(spec, load_state_dict(path, torch.float64), REFERENCE, 512), prompts())
    mp.start_processes(worker, args=(2, free_port(), str(path), mode, str(tmp_path)), nprocs=2, join=True,
                       start_method="spawn")  # fmt: skip
    model = DeepseekV2(spec, load_state_dict(path, torch.float64), REFERENCE, 512)
    for rank in range(2):
        got = torch.load(tmp_path / f"{mode}-{rank}.pt")
        # EP: compare with one process running the same requests. (Against the
        # 4-request batch it is off by ~1e-8: the reference router's fp32 GEMM
        # is not batch-invariant on CPU BLAS, which is chapter 10's point.)
        expected = single if mode == "tp" else last_logits(model, prompts()[rank::2])
        torch.testing.assert_close(got, expected, rtol=1e-9, atol=1e-9)
