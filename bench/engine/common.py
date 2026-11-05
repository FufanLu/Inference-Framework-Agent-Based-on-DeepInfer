"""Shared helpers for the engine benchmarks (bench/engine/*.py).

Every script writes one JSON file with its raw numbers and the environment
(GPU, driver, torch/triton/vLLM versions), so docs quote measured values only.
"""

from __future__ import annotations

import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
#: cuda on the cluster; cpu (with TRITON_INTERPRET=1) for a local smoke run.
DEVICE = os.environ.get("LEV_BENCH_DEVICE", "cuda")
sys.path.insert(0, str(ROOT / "engine"))

# Real text for parity checks: logits on natural text are what a user sees.
TEXTS = [
    "The Hopper architecture introduced the Tensor Memory Accelerator, which moves tiles between global and "
    "shared memory asynchronously so that warps can overlap data movement with matrix multiply-accumulate.",
    "def merge_sort(xs):\n    if len(xs) <= 1:\n        return xs\n    mid = len(xs) // 2\n"
    "    left, right = merge_sort(xs[:mid]), merge_sort(xs[mid:])\n    out = []\n",
    "混合专家模型在每个 token 上只激活少数专家，因此总参数量很大而单 token 计算量较小，"
    "但专家之间的负载不均衡会让部分 GPU 成为瓶颈。",
    "Paged attention stores the key-value cache in fixed-size blocks, so memory for a sequence does not need "
    "to be contiguous and blocks holding a shared prefix can be reused by many requests at once.",
    "In 1905 Einstein published four papers: on the photoelectric effect, Brownian motion, special relativity "
    "and the equivalence of mass and energy. Each of them",
    "User: Explain why the sky is blue in two sentences.\n\nAssistant:",
    "SELECT customer_id, SUM(amount) AS total FROM orders WHERE created_at >= '2026-01-01' GROUP BY customer_id "
    "HAVING SUM(amount) > 1000 ORDER BY total DESC LIMIT 10;",
    "The router keeps a radix tree of block hashes per worker and sends each request to the worker whose cache "
    "already holds the longest prefix of its prompt, unless that worker is overloaded.",
]


def environment() -> dict:
    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        env.update(gpu=props.name, gpu_memory_gib=round(props.total_memory / 2**30, 1), sm=f"{props.major}{props.minor}")
    try:
        import triton

        env["triton"] = triton.__version__
    except ImportError:
        pass
    try:
        import vllm

        env["vllm"] = vllm.__version__
    except ImportError:
        pass
    return env


def write(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"environment": environment(), **payload}
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path}")


def random_prompts(n: int, length: int, vocab: int, seed: int = 0) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randrange(100, vocab - 100) for _ in range(length)] for _ in range(n)]


def tokenizer(model_dir: str | Path):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
