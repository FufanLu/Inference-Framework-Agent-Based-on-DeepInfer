"""A block-level prefix cache with a GPU tier and a CPU tier.

Blocks are identified by the chain hashes of :mod:`leviathan_worker.hashing`,
so a hash names a whole prefix and matching is a walk along the prompt's hash
chain.

GPU tier (``gpu_blocks`` slots)
    A slot holds either a *hashed* block, which any request with that prefix
    can share, or an *unhashed* block that one running request is filling
    (its partial last prompt block and its output). Hashed blocks referenced by
    a running request are pinned; unreferenced ones wait in an LRU and are
    evicted only when a slot is needed, as in vLLM.

CPU tier (``cpu_blocks`` slots)
    An LRU of hashed blocks. GPU evictions demote here and CPU evictions drop
    the block. A block lives in exactly one tier; a CPU hit is promoted back.

Admission control is exact: a request acquires a :class:`Lease` with a budget
of every block it will ever hold, and :attr:`TieredPrefixCache.free_blocks`
counts what is neither pinned nor promised, so an admitted request never runs
out of KV and never needs preemption. Slots are claimed physically only as
the request's KV grows, so cached blocks survive until space is really needed.

Every change is recorded as a KV event; :meth:`drain_events` hands them over.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..engine import BlockRemoved, BlockStored, KvEvent, Medium


class CacheOvercommitError(RuntimeError):
    """A caller acquired more blocks than :attr:`TieredPrefixCache.free_blocks` allowed."""


@dataclass
class _GpuBlock:
    parent: int | None
    refs: int = 0


@dataclass(eq=False)
class Lease:
    """The GPU blocks held by one running request."""

    #: Blocks the request may hold over its lifetime (prompt plus planned output).
    budget: int
    #: Pinned hashed blocks, in chain order: the request's cached prefix so far.
    hashes: list[int] = field(default_factory=list)
    #: Unhashed blocks allocated so far.
    materialized: int = 0
    #: Blocks promoted from the CPU tier when the lease was acquired.
    loaded_from_cpu: int = 0
    released: bool = False

    @property
    def promised(self) -> int:
        """Budgeted blocks that do not hold a hashed block yet."""
        return self.budget - len(self.hashes)

    @property
    def last_hash(self) -> int | None:
        return self.hashes[-1] if self.hashes else None


class TieredPrefixCache:
    def __init__(self, gpu_blocks: int, cpu_blocks: int) -> None:
        if gpu_blocks <= 0:
            raise ValueError("gpu_blocks must be positive")
        if cpu_blocks < 0:
            raise ValueError("cpu_blocks must not be negative")
        self.gpu_capacity = gpu_blocks
        self.cpu_capacity = cpu_blocks
        self._gpu: dict[int, _GpuBlock] = {}
        self._gpu_lru: OrderedDict[int, None] = OrderedDict()  # unreferenced, least recent first
        self._cpu: OrderedDict[int, int | None] = OrderedDict()  # hash -> parent, least recent first
        self._promised = 0  # sum of lease.promised over live leases
        self._materialized = 0  # sum of lease.materialized over live leases
        self._log = _EventLog()

    # ------------------------------------------------------------------ accounting

    @property
    def used_blocks(self) -> int:
        """GPU blocks that are pinned or promised to running requests."""
        return len(self._gpu) - len(self._gpu_lru) + self._promised

    @property
    def free_blocks(self) -> int:
        """GPU blocks available to new requests (cached but unreferenced blocks count as free)."""
        return self.gpu_capacity - self.used_blocks

    @property
    def usage(self) -> float:
        return self.used_blocks / self.gpu_capacity

    def gpu_hashes(self) -> set[int]:
        return set(self._gpu)

    def cpu_hashes(self) -> set[int]:
        return set(self._cpu)

    def refs(self, block_hash: int) -> int:
        block = self._gpu.get(block_hash)
        return block.refs if block else 0

    # ------------------------------------------------------------------ requests

    def match(self, hashes: Sequence[int]) -> int:
        """Length of the longest prefix of ``hashes`` cached on either tier."""
        n = 0
        for h in hashes:
            if h not in self._gpu and h not in self._cpu:
                break
            n += 1
        return n

    def slots_needed(self, hits: Sequence[int], budget: int) -> int:
        """Free blocks a request needs to pin the matched prefix ``hits`` within ``budget`` blocks.

        Promised blocks count once each; a hit counts unless it is already
        pinned by another request (then it is shared for free).
        """
        needed = budget - len(hits)
        for h in hits:
            block = self._gpu.get(h)
            if block is None or block.refs == 0:
                needed += 1
        return needed

    def acquire(self, hits: Sequence[int], budget: int) -> Lease:
        """Pin ``hits`` (a prefix returned by :meth:`match`), promoting CPU blocks, and promise the rest."""
        if budget < len(hits):
            raise ValueError("budget is smaller than the matched prefix")
        if self.slots_needed(hits, budget) > self.free_blocks:
            raise CacheOvercommitError("not enough free KV blocks for this lease")
        # Pin GPU hits first so the evictions below cannot take them, and lift
        # CPU hits out of the CPU tier so the demotions below cannot drop them.
        for h in hits:
            if h in self._gpu:
                self._pin(h)
        promoted = [h for h in hits if h not in self._gpu]
        parents = {h: self._cpu.pop(h) for h in promoted}
        if promoted:
            self._log.removed(Medium.CPU, promoted)
            self._make_room(len(promoted))
            for h in promoted:
                self._gpu[h] = _GpuBlock(parent=parents[h], refs=1)
                self._log.stored(Medium.GPU, h, parents[h])
        lease = Lease(budget=budget, hashes=list(hits), loaded_from_cpu=len(promoted))
        self._promised += lease.promised
        return lease

    def commit(self, lease: Lease, hashes: Sequence[int], parent: int | None) -> None:
        """Record that ``lease`` filled the blocks ``hashes``, chained after ``parent``.

        A block another request already cached (on either tier) is shared
        instead of stored twice.
        """
        self._check_live(lease)
        if len(hashes) > lease.promised:
            raise CacheOvercommitError("commit exceeds the lease budget")
        for h in hashes:
            if lease.materialized:  # this unhashed block now carries the hash
                lease.materialized -= 1
                self._materialized -= 1
            if h in self._gpu:
                self._pin(h)
            else:
                if h in self._cpu:
                    del self._cpu[h]
                    self._log.removed(Medium.CPU, [h])
                self._make_room(1)
                self._gpu[h] = _GpuBlock(parent=parent, refs=1)
                self._log.stored(Medium.GPU, h, parent)
            lease.hashes.append(h)
            self._promised -= 1
            parent = h

    def allocate(self, lease: Lease, blocks: int) -> None:
        """The request's KV now spans ``blocks`` blocks; claim slots for the unhashed ones."""
        self._check_live(lease)
        target = min(max(0, blocks - len(lease.hashes)), lease.promised)
        delta = target - lease.materialized
        if delta > 0:
            self._make_room(delta)
        lease.materialized += delta
        self._materialized += delta

    def release(self, lease: Lease) -> None:
        """Unpin everything ``lease`` holds; its hashed blocks stay cached."""
        if lease.released:
            return
        lease.released = True
        self._promised -= lease.promised
        self._materialized -= lease.materialized
        # Tail blocks enter the LRU first, so they are evicted before the
        # (more widely shared) head of the prefix, as in vLLM.
        for h in reversed(lease.hashes):
            block = self._gpu[h]
            block.refs -= 1
            if block.refs == 0:
                self._gpu_lru[h] = None
        lease.hashes = []
        lease.materialized = 0
        lease.budget = 0

    def drain_events(self) -> list[KvEvent]:
        return self._log.drain()

    def check_invariants(self) -> None:
        """Raise AssertionError if the bookkeeping is inconsistent (for tests)."""
        assert len(self._gpu) + self._materialized <= self.gpu_capacity, "GPU slots overfilled"
        assert self.used_blocks <= self.gpu_capacity, "GPU blocks overcommitted"
        assert len(self._cpu) <= self.cpu_capacity, "CPU tier overfilled"
        assert not set(self._gpu) & set(self._cpu), "a block is on both tiers"
        for h, block in self._gpu.items():
            assert block.refs >= 0
            assert (block.refs == 0) == (h in self._gpu_lru), "LRU out of sync with refcounts"

    # ------------------------------------------------------------------ internals

    def _check_live(self, lease: Lease) -> None:
        if lease.released:
            raise ValueError("lease was released")

    def _pin(self, h: int) -> None:
        block = self._gpu[h]
        if block.refs == 0:
            del self._gpu_lru[h]
        block.refs += 1

    def _make_room(self, n: int) -> None:
        """Evict unreferenced GPU blocks (oldest first) until ``n`` more slots fit."""
        excess = len(self._gpu) + self._materialized + n - self.gpu_capacity
        if excess <= 0:
            return
        if excess > len(self._gpu_lru):
            raise CacheOvercommitError("GPU tier has no evictable blocks left")
        victims = []
        for _ in range(excess):
            h, _ = self._gpu_lru.popitem(last=False)
            victims.append((h, self._gpu.pop(h).parent))
        self._log.removed(Medium.GPU, [h for h, _ in victims])
        self._demote(victims)

    def _demote(self, victims: list[tuple[int, int | None]]) -> None:
        if self.cpu_capacity == 0:
            return
        victims = victims[-self.cpu_capacity :]  # more victims than the tier holds: keep the newest
        overflow = len(self._cpu) + len(victims) - self.cpu_capacity
        if overflow > 0:
            dropped = [self._cpu.popitem(last=False)[0] for _ in range(overflow)]
            self._log.removed(Medium.CPU, dropped)
        for h, parent in victims:
            self._cpu[h] = parent
            self._log.stored(Medium.CPU, h, parent)


@dataclass
class _PendingStored:
    medium: Medium
    parent: int | None
    hashes: list[int]


@dataclass
class _PendingRemoved:
    medium: Medium
    hashes: list[int]


class _EventLog:
    """Records cache changes, merging adjacent events that extend each other."""

    def __init__(self) -> None:
        self._events: list[_PendingStored | _PendingRemoved] = []

    def stored(self, medium: Medium, h: int, parent: int | None) -> None:
        last = self._events[-1] if self._events else None
        if isinstance(last, _PendingStored) and last.medium is medium and parent is not None and last.hashes[-1] == parent:
            last.hashes.append(h)
        else:
            self._events.append(_PendingStored(medium, parent, [h]))

    def removed(self, medium: Medium, hashes: Sequence[int]) -> None:
        if not hashes:
            return
        last = self._events[-1] if self._events else None
        if isinstance(last, _PendingRemoved) and last.medium is medium:
            last.hashes.extend(hashes)
        else:
            self._events.append(_PendingRemoved(medium, list(hashes)))

    def drain(self) -> list[KvEvent]:
        events: list[KvEvent] = []
        for e in self._events:
            if isinstance(e, _PendingStored):
                events.append(BlockStored(block_hashes=tuple(e.hashes), parent_hash=e.parent, medium=e.medium))
            else:
                events.append(BlockRemoved(block_hashes=tuple(e.hashes), medium=e.medium))
        self._events.clear()
        return events
