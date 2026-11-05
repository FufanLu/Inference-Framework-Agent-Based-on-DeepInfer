"""Kernel dispatch.

`Ops` bundles the kernels one model uses. The reference backend is plain
PyTorch; the Triton backend (`lev_engine.ops.triton`) replaces the hot ones
and is checked against the reference in tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from . import reference


@dataclass(frozen=True)
class Ops:
    name: str
    linear: Callable
    head_matmul: Callable
    rms_norm: Callable
    silu_and_mul: Callable
    write_kv: Callable
    mla_attention: Callable
    moe_routed: Callable


REFERENCE = Ops(
    name="reference",
    linear=reference.linear,
    head_matmul=reference.head_matmul,
    rms_norm=reference.rms_norm,
    silu_and_mul=reference.silu_and_mul,
    write_kv=reference.write_kv,
    mla_attention=reference.mla_attention,
    moe_routed=reference.moe_routed,
)


def get_ops(backend: str, batch_invariant: bool = False) -> Ops:
    if backend == "reference":
        return REFERENCE
    if backend == "triton":
        from .triton import triton_ops  # noqa: PLC0415  (needs triton installed)

        return triton_ops(batch_invariant)
    raise ValueError(f"unknown kernel backend {backend!r}")
