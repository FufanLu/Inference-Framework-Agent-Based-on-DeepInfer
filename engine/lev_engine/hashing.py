"""Prefix block hashes: the convention of crates/lev-router/src/hashing.rs.

Identical to ``leviathan_worker.hashing`` (a test checks it), so the engine's
KV events need no translation before the router sees them, unlike vLLM's.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence

import xxhash

ROOT_SEED = 0x4C45_5649_4154_4841  # b"LEVIATHA"


def block_hash(tokens: Sequence[int], parent: int) -> int:
    return xxhash.xxh3_64_intdigest(struct.pack(f"<{len(tokens)}I", *tokens), seed=parent)


def block_hashes(tokens: Sequence[int], block_size: int, parent: int = ROOT_SEED) -> list[int]:
    hashes: list[int] = []
    for start in range(0, len(tokens) - block_size + 1, block_size):
        parent = block_hash(tokens[start : start + block_size], parent)
        hashes.append(parent)
    return hashes
