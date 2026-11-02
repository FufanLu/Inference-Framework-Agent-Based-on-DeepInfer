"""The block hash convention shared with the router (crates/lev-router/src/hashing.rs).

Only full blocks of ``block_size`` prompt tokens are hashed. Each block hash
seeds the next one, so a hash identifies the whole prefix up to and including
its block::

    h[0] = xxh3_64(le_u32_bytes(tokens[0:B]),        seed=ROOT_SEED)
    h[i] = xxh3_64(le_u32_bytes(tokens[iB:(i+1)B]),  seed=h[i-1])
"""

from __future__ import annotations

import struct
from collections.abc import Sequence

import xxhash

ROOT_SEED = 0x4C45_5649_4154_4841  # b"LEVIATHA"


def block_hashes(tokens: Sequence[int], block_size: int, parent: int = ROOT_SEED) -> list[int]:
    """Return the chain hash of every full block of ``tokens``.

    ``parent`` continues an existing chain: pass the hash of the block that
    precedes ``tokens[0]``.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    pack = struct.Struct(f"<{block_size}I").pack
    hashes: list[int] = []
    for start in range(0, len(tokens) - block_size + 1, block_size):
        parent = xxhash.xxh3_64_intdigest(pack(*tokens[start : start + block_size]), seed=parent)
        hashes.append(parent)
    return hashes
