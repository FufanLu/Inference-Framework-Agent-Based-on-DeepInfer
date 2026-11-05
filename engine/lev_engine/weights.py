"""Load Hugging Face safetensors checkpoints (single file or sharded)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import torch
from safetensors import safe_open


def checkpoint_files(path: str | Path) -> list[Path]:
    path = Path(path)
    index = path / "model.safetensors.index.json"
    if index.exists():
        shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        return [path / shard for shard in shards]
    single = path / "model.safetensors"
    if single.exists():
        return [single]
    raise FileNotFoundError(f"no safetensors checkpoint in {path}")


def iter_tensors(path: str | Path, device: str | torch.device = "cpu") -> Iterator[tuple[str, torch.Tensor]]:
    """Yield every tensor, loaded straight to `device`, one at a time."""
    for file in checkpoint_files(path):
        with safe_open(str(file), framework="pt", device=str(device)) as f:
            for name in f.keys():
                yield name, f.get_tensor(name)


def load_state_dict(path: str | Path, dtype: torch.dtype, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    return {name: t.to(dtype) for name, t in iter_tensors(path, device)}
