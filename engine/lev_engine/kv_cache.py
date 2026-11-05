"""Paged KV block pool with prefix caching.

Each block holds `block_size` tokens of every layer's latent KV. A block that
is full gets the router's chain hash of its prefix (`hashing.block_hash`) and
stays findable by that hash after its last user lets go of it, until the pool
needs the space back:

    in use (ref > 0)  --last free-->  evictable (LRU, still cached)
           ^                                 |
           +------- prefix hit --------------+--- evicted when allocating --> free

Every change to the set of cached hashes is recorded as a KV event, in the
same form the worker publishes to the router (BlockStored / BlockRemoved).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BlockStored:
    block_hashes: tuple[int, ...]
    parent_hash: int | None


@dataclass(frozen=True)
class BlockRemoved:
    block_hashes: tuple[int, ...]


@dataclass
class _Block:
    id: int
    ref: int = 0
    hash: int | None = None
    parent: int | None = None


class OutOfBlocksError(RuntimeError):
    pass


@dataclass
class BlockPool:
    num_blocks: int
    block_size: int
    enable_prefix_caching: bool = True
    blocks: list[_Block] = field(init=False)
    _free: list[int] = field(init=False)  # never cached, or evicted
    _evictable: OrderedDict[int, None] = field(init=False)  # cached, ref 0, LRU first
    _by_hash: dict[int, int] = field(init=False)
    events: list[BlockStored | BlockRemoved] = field(init=False)

    def __post_init__(self) -> None:
        self.blocks = [_Block(i) for i in range(self.num_blocks)]
        self._free = list(range(self.num_blocks - 1, -1, -1))  # pop() hands out 0, 1, 2, ...
        self._evictable = OrderedDict()
        self._by_hash = {}
        self.events = []

    # ------------------------------------------------------------------ queries

    @property
    def num_free(self) -> int:
        """Blocks an allocation can use now (free plus evictable)."""
        return len(self._free) + len(self._evictable)

    @property
    def num_cached(self) -> int:
        return len(self._by_hash)

    def usage(self) -> float:
        return 1.0 - self.num_free / self.num_blocks

    # ------------------------------------------------------------------ prefix cache

    def match(self, hashes: list[int]) -> list[int]:
        """Take a reference on the longest cached run of `hashes` (a prefix chain)."""
        if not self.enable_prefix_caching:
            return []
        found: list[int] = []
        for h in hashes:
            block = self._by_hash.get(h)
            if block is None:
                break
            found.append(block)
        for block in found:
            self._ref(block)
        return found

    def register(self, block_id: int, block_hash: int, parent: int | None) -> None:
        """Mark a full block as holding the prefix `block_hash`."""
        if not self.enable_prefix_caching:
            return
        block = self.blocks[block_id]
        if block.hash is not None or block_hash in self._by_hash:
            return  # already cached (by this or an identical block)
        block.hash, block.parent = block_hash, parent
        self._by_hash[block_hash] = block_id
        self.events.append(BlockStored((block_hash,), parent))

    # ------------------------------------------------------------------ allocation

    def allocate(self, n: int) -> list[int]:
        if n > self.num_free:
            raise OutOfBlocksError(f"need {n} blocks, {self.num_free} free")
        out = []
        for _ in range(n):
            if self._free:
                block_id = self._free.pop()
            else:
                block_id, _ = self._evictable.popitem(last=False)
                self._evict(block_id)
            self.blocks[block_id].ref = 1
            out.append(block_id)
        return out

    def free(self, block_ids: list[int]) -> None:
        """Drop one reference on each block. Later blocks of a sequence are
        released first so that the LRU evicts a prefix's tail before its head."""
        for block_id in reversed(block_ids):
            block = self.blocks[block_id]
            block.ref -= 1
            assert block.ref >= 0, f"block {block_id} freed twice"
            if block.ref == 0:
                if block.hash is not None:
                    self._evictable[block_id] = None
                else:
                    self._free.append(block_id)

    def reset_prefix_cache(self) -> None:
        for block_id in list(self._evictable):
            del self._evictable[block_id]
            self._evict(block_id)
            self._free.append(block_id)

    def drain_events(self) -> list[BlockStored | BlockRemoved]:
        events, self.events = self.events, []
        return events

    # ------------------------------------------------------------------ internals

    def _ref(self, block_id: int) -> None:
        block = self.blocks[block_id]
        if block.ref == 0:
            self._evictable.pop(block_id, None)
        block.ref += 1

    def _evict(self, block_id: int) -> None:
        block = self.blocks[block_id]
        if block.hash is not None:
            del self._by_hash[block.hash]
            self.events.append(BlockRemoved((block.hash,)))
            block.hash = block.parent = None
