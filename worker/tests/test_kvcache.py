"""Unit tests of the tiered prefix cache."""

from __future__ import annotations

import random

import pytest

from leviathan_worker.engine import BlockRemoved, BlockStored, Medium
from leviathan_worker.events import CacheMirror
from leviathan_worker.kvcache import CacheOvercommitError, TieredPrefixCache

GPU, CPU = Medium.GPU, Medium.CPU


def test_committed_chain_is_published_and_stays_cached_after_release():
    cache = TieredPrefixCache(8, 4)
    lease = cache.acquire([], 4)
    assert (cache.used_blocks, cache.free_blocks) == (4, 4)
    cache.commit(lease, [1, 2, 3], None)
    assert cache.drain_events() == [BlockStored((1, 2, 3), None, GPU)]
    assert cache.used_blocks == 4  # three pinned blocks plus one promised
    cache.release(lease)
    assert (cache.used_blocks, cache.usage) == (0, 0.0)
    assert cache.gpu_hashes() == {1, 2, 3}
    assert cache.match([1, 2, 3, 4]) == 3
    assert cache.drain_events() == []


def test_a_prefix_pinned_by_another_request_is_shared_for_free():
    cache = TieredPrefixCache(8, 0)
    a = cache.acquire([], 3)
    cache.commit(a, [1, 2], None)
    hits = [1, 2][: cache.match([1, 2, 9])]
    assert cache.slots_needed(hits, 4) == 2
    b = cache.acquire(hits, 4)
    assert cache.used_blocks == 2 + 1 + 2  # shared blocks, a's promise, b's promise
    assert cache.refs(1) == cache.refs(2) == 2
    cache.release(a)
    cache.release(b)
    assert cache.used_blocks == 0
    cache.check_invariants()


def test_concurrent_commits_of_the_same_block_store_it_once():
    cache = TieredPrefixCache(8, 0)
    a, b = cache.acquire([], 2), cache.acquire([], 2)
    cache.commit(a, [7], None)
    cache.commit(b, [7], None)
    assert cache.drain_events() == [BlockStored((7,), None, GPU)]
    assert cache.refs(7) == 2
    assert cache.used_blocks == 3


def test_eviction_is_lazy_lru_tail_first_and_demotes_to_cpu():
    cache = TieredPrefixCache(4, 2)
    a = cache.acquire([], 4)
    cache.commit(a, [1, 2, 3, 4], None)
    cache.release(a)
    cache.drain_events()

    b = cache.acquire([], 2)
    assert cache.gpu_hashes() == {1, 2, 3, 4}, "a promise alone claims no slot"
    cache.allocate(b, 1)  # b's first KV block needs a slot: the tail of a's chain goes first
    assert cache.drain_events() == [BlockRemoved((4,), GPU), BlockStored((4,), 3, CPU)]
    cache.commit(b, [10], None)  # the allocated block gets its hash; no new slot needed
    assert cache.drain_events() == [BlockStored((10,), None, GPU)]

    cache.allocate(b, 2)
    assert cache.drain_events() == [BlockRemoved((3,), GPU), BlockStored((3,), 2, CPU)]
    c = cache.acquire([], 2)
    cache.allocate(c, 2)  # two more slots: blocks 2 and 1 demote, the CPU tier drops 4 and 3
    assert cache.drain_events() == [
        BlockRemoved((2, 1), GPU),
        BlockRemoved((4, 3), CPU),
        BlockStored((2,), 1, CPU),
        BlockStored((1,), None, CPU),
    ]
    assert cache.gpu_hashes() == {10} and cache.cpu_hashes() == {1, 2}
    cache.check_invariants()


def test_cpu_hits_are_promoted_back_to_the_gpu():
    cache = TieredPrefixCache(4, 8)
    a = cache.acquire([], 4)
    cache.commit(a, [1, 2, 3], None)
    cache.release(a)
    b = cache.acquire([], 4)
    cache.allocate(b, 4)  # pushes 1, 2, 3 to the CPU tier
    cache.release(b)
    assert cache.cpu_hashes() == {1, 2, 3}
    cache.drain_events()

    chain = [1, 2, 3, 4]
    hits = chain[: cache.match(chain)]
    assert hits == [1, 2, 3]
    lease = cache.acquire(hits, 4)
    assert lease.loaded_from_cpu == 3
    assert cache.drain_events() == [BlockRemoved((1, 2, 3), CPU), BlockStored((1, 2, 3), None, GPU)]
    assert cache.gpu_hashes() == {1, 2, 3} and not cache.cpu_hashes()
    cache.check_invariants()


def test_matching_walks_both_tiers():
    cache = TieredPrefixCache(2, 4)
    a = cache.acquire([], 2)
    cache.commit(a, [1, 2], None)
    cache.release(a)
    b = cache.acquire([], 1)
    cache.allocate(b, 1)  # demotes 2 (the tail), keeps 1 on the GPU
    assert cache.gpu_hashes() == {1} and cache.cpu_hashes() == {2}
    assert cache.match([1, 2, 3]) == 2


def test_overcommit_is_refused():
    cache = TieredPrefixCache(4, 0)
    cache.acquire([], 3)
    assert cache.slots_needed([], 2) > cache.free_blocks
    with pytest.raises(CacheOvercommitError):
        cache.acquire([], 2)


def test_random_workload_keeps_invariants_and_events_in_sync():
    rng = random.Random(1234)
    cache = TieredPrefixCache(16, 8)
    mirror = CacheMirror()
    live = []

    def chain() -> list[int]:
        root, branch = rng.randrange(4), rng.randrange(4)
        trunk = [root * 1000 + i for i in range(rng.randint(0, 5))]
        return trunk + [root * 1000 + 100 * (branch + 1) + len(trunk) * 10 + i for i in range(rng.randint(1, 4))]

    for _ in range(3000):
        op = rng.random()
        if op < 0.35 and len(live) < 6:
            blocks = chain()
            hits = blocks[: cache.match(blocks)]
            budget = len(blocks) + rng.randint(0, 3)
            if cache.slots_needed(hits, budget) <= cache.free_blocks:
                live.append((cache.acquire(hits, budget), blocks))
        elif op < 0.75 and live:
            lease, blocks = rng.choice(live)
            n = len(lease.hashes)
            if n < len(blocks) and rng.random() < 0.7:
                cache.commit(lease, [blocks[n]], lease.last_hash)
            cache.allocate(lease, min(lease.budget, len(lease.hashes) + rng.randint(0, 2)))
        elif live:
            lease, _ = live.pop(rng.randrange(len(live)))
            cache.release(lease)
        mirror.apply(cache.drain_events())
        cache.check_invariants()
        assert mirror.blocks(GPU) == cache.gpu_hashes()
        assert mirror.blocks(CPU) == cache.cpu_hashes()
    for lease, _ in live:
        cache.release(lease)
    assert cache.used_blocks == 0
