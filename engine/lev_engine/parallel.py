"""Tensor and expert parallelism for DeepSeek-V2 (torch.distributed).

TP (tensor parallel, every rank sees the same batch):
    attention heads are split across ranks (q_proj rows, W_UK/W_UV, o_proj
    columns); the MLA latent cache is the same on every rank (it is shared by
    all heads), so it is replicated. Dense MLPs and every expert are split on
    the intermediate dimension. Two all-reduces per layer: after o_proj and
    after the MLP.

EP (expert parallel, each rank runs its own requests = DP attention):
    attention, the dense first layer and the shared experts are local; routed
    experts are partitioned (rank r owns experts [r*E/n, (r+1)*E/n)). Each MoE
    layer sends every (token, expert) pair to the expert's owner with one
    all_to_all, computes, and sends the rows back with another; the top-k
    outputs are summed in the same fixed order as on one GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class Parallel:
    mode: str  # "tp" or "ep"
    size: int
    rank: int
    group: dist.ProcessGroup | None = None

    @classmethod
    def from_env(cls, mode: str) -> Parallel:
        if not dist.is_initialized():
            raise RuntimeError("init torch.distributed first (torchrun sets RANK/WORLD_SIZE)")
        if mode not in ("tp", "ep"):
            raise ValueError(f"unknown parallel mode {mode!r}")
        return cls(mode, dist.get_world_size(), dist.get_rank())

    @property
    def tp(self) -> bool:
        return self.mode == "tp" and self.size > 1

    @property
    def ep(self) -> bool:
        return self.mode == "ep" and self.size > 1

    def shard(self, n: int) -> slice:
        if n % self.size:
            raise ValueError(f"{n} does not split over {self.size} ranks")
        step = n // self.size
        return slice(self.rank * step, (self.rank + 1) * step)

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(x, group=self.group)
        return x


def ep_dispatch_combine(
    x: torch.Tensor,  # [T, hidden]
    topk_ids: torch.Tensor,  # [T, k] global expert ids
    topk_weights: torch.Tensor,  # [T, k] fp32
    experts_per_rank: int,
    expert_fn,  # (rows [R, hidden], local_ids [R]) -> [R, hidden]
    par: Parallel,
) -> torch.Tensor:
    """Routed experts across ranks: dispatch, compute, combine. Returns [T, hidden]."""
    T, k = topk_ids.shape
    hidden = x.shape[-1]
    flat = topk_ids.reshape(-1)
    owner = flat // experts_per_rank
    order = torch.argsort(owner, stable=True)  # pairs grouped by destination rank
    send_counts = torch.bincount(owner, minlength=par.size)
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts, group=par.group)
    send_split, recv_split = send_counts.tolist(), recv_counts.tolist()

    send_rows = x[order // k]
    recv_rows = x.new_empty((sum(recv_split), hidden))
    dist.all_to_all_single(recv_rows, send_rows, recv_split, send_split, group=par.group)
    send_ids = flat[order] % experts_per_rank  # local expert id at the destination
    recv_ids = send_ids.new_empty((sum(recv_split),))
    dist.all_to_all_single(recv_ids, send_ids, recv_split, send_split, group=par.group)

    computed = expert_fn(recv_rows, recv_ids)
    back = x.new_empty((T * k, hidden))
    dist.all_to_all_single(back, computed.contiguous(), send_split, recv_split, group=par.group)
    pair_out = torch.empty_like(back)
    pair_out[order] = back  # undo the sort: row j is pair j = (token j // k, slot j % k)
    weighted = pair_out.view(T, k, hidden).to(torch.promote_types(x.dtype, torch.float32)) * topk_weights.unsqueeze(-1)
    out = weighted[:, 0]
    for j in range(1, k):  # fixed summation order, as on one GPU
        out = out + weighted[:, j]
    return out.to(x.dtype)
