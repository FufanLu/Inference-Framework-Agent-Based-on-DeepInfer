"""Unit tests of the event hub: sequences, overflow, snapshots and load windows."""

from __future__ import annotations

import asyncio
import dataclasses

import pytest

from leviathan_worker.engine import (
    AllCleared,
    BlockRemoved,
    BlockStored,
    KvEventBatch,
    LoadCounters,
    LoadReport,
    Medium,
    SubscriberOverflowError,
)
from leviathan_worker.events import CacheMirror, EventHub, snapshot_batches

GPU, CPU = Medium.GPU, Medium.CPU
IDLE = LoadCounters(running=0, waiting=0, kv_usage=0.0, free_blocks=10, prefill_tokens=0, decode_tokens=0,
                    spec_draft_tokens=0, spec_accepted_tokens=0)


async def next_event(subscription):
    return await asyncio.wait_for(anext(subscription), 2.0)


async def subscribe(hub: EventHub, *, snapshot: bool = False, interval_ms: int = 60_000):
    subscription = hub.subscribe(snapshot=snapshot, load_report_interval_ms=interval_ms)
    kv = []
    while not isinstance(event := await next_event(subscription), LoadReport):
        kv.append(event)
    return subscription, kv


async def test_each_subscriber_numbers_its_own_batches():
    hub = EventHub(lambda: IDLE)
    a, _ = await subscribe(hub)
    hub.publish([BlockStored((1,), None, GPU)])
    b, _ = await subscribe(hub)
    hub.publish([BlockStored((2,), 1, GPU)])
    hub.publish([])  # nothing to send
    assert [(e.seq, e.events) for e in [await next_event(a), await next_event(a)]] == [
        (1, (BlockStored((1,), None, GPU),)),
        (2, (BlockStored((2,), 1, GPU),)),
    ]
    assert await next_event(b) == KvEventBatch(1, (BlockStored((2,), 1, GPU),))


async def test_a_subscriber_that_falls_behind_is_dropped():
    hub = EventHub(lambda: IDLE, queue_size=2)
    slow, _ = await subscribe(hub)
    fast, _ = await subscribe(hub)
    for i in range(2):
        hub.publish([BlockStored((i,), None, GPU)])
        await next_event(fast)
    hub.publish([BlockStored((9,), None, GPU)])
    assert hub.subscriber_count == 1
    with pytest.raises(SubscriberOverflowError):
        await next_event(slow)
    assert (await next_event(fast)).seq == 3


async def test_close_ends_a_waiting_subscriber():
    hub = EventHub(lambda: IDLE)
    subscription, _ = await subscribe(hub)
    waiter = asyncio.create_task(anext(subscription))
    await asyncio.sleep(0.01)
    hub.close()
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(waiter, 2.0)
    with pytest.raises(RuntimeError):
        hub.subscribe(snapshot=False, load_report_interval_ms=100)


async def test_snapshot_subscription_starts_from_a_clean_state():
    hub = EventHub(lambda: IDLE)
    hub.publish([BlockStored((1, 2, 3), None, GPU), BlockRemoved((3,), GPU), BlockStored((3,), 2, CPU)])
    subscription, kv = await subscribe(hub, snapshot=True)
    assert kv[0].seq == 1 and kv[0].events[0] == AllCleared()
    mirror = CacheMirror()
    for batch in kv:
        mirror.apply(batch.events)
    assert mirror.blocks(GPU) == {1, 2} and mirror.blocks(CPU) == {3}
    hub.publish([BlockRemoved((3,), CPU)])
    assert (await next_event(subscription)).seq == kv[-1].seq + 1


def test_snapshot_emits_parents_before_children_across_tiers_and_branches():
    mirror = CacheMirror()
    # 1 -> 2 -> 3 on the GPU, 2 -> 4 -> 5 with 4 on the CPU, 6 in both tiers, 9 an orphan.
    mirror.apply(
        [
            BlockStored((1, 2, 3, 4, 5), None, GPU),
            BlockStored((9,), 8, CPU),
            BlockRemoved((4,), GPU),
            BlockStored((4,), 2, CPU),
            BlockStored((6,), 5, GPU),
            BlockStored((6,), 5, CPU),
            BlockRemoved((2,), GPU),
            BlockStored((2,), 1, GPU),  # re-stored after its children
        ]
    )
    events = mirror.snapshot()
    replayed = CacheMirror()
    replayed.apply(events)
    for medium in (GPU, CPU):
        assert replayed.blocks(medium) == mirror.blocks(medium)
    cached = mirror.blocks(GPU) | mirror.blocks(CPU)
    seen: set[int] = set()
    for event in events:
        if event.parent_hash in cached:
            assert event.parent_hash in seen, f"{event} precedes its parent"
        seen.update(event.block_hashes)


def test_snapshot_batches_respect_the_size_limit():
    mirror = CacheMirror()
    mirror.apply([BlockStored(tuple(range(1, 11)), None, GPU), BlockStored((100, 101), 5, CPU)])
    batches = snapshot_batches(mirror, chunk=4)
    assert batches[0][0] == AllCleared()
    assert all(sum(len(e.block_hashes) for e in b if isinstance(e, BlockStored)) <= 4 for b in batches)
    replayed = CacheMirror()
    for batch in batches:
        replayed.apply(batch)
    assert replayed.blocks(GPU) == set(range(1, 11)) and replayed.blocks(CPU) == {100, 101}


async def test_load_reports_turn_counters_into_rates():
    counters = IDLE
    hub = EventHub(lambda: counters)
    subscription, _ = await subscribe(hub, interval_ms=20)
    counters = dataclasses.replace(
        IDLE, running=2, decode_tokens=200, prefill_tokens=500, spec_draft_tokens=50, spec_accepted_tokens=30
    )
    report = await next_event(subscription)
    assert isinstance(report, LoadReport)
    assert report.running == 2
    assert report.decode_tokens_per_s > 0 and report.prefill_tokens_per_s > 0
    assert report.spec_accept_rate == pytest.approx(0.6)
    idle_window = await next_event(subscription)  # no drafts in this window: the rate is kept
    assert idle_window.spec_accept_rate == pytest.approx(0.6)
    assert idle_window.decode_tokens_per_s < report.decode_tokens_per_s
    assert idle_window.timestamp_ms >= report.timestamp_ms
    subscription.close()
