"""Prefix caching, KV events, snapshots and load reports, observed over gRPC."""

from __future__ import annotations

import asyncio

from support import EventRecorder, collect, make_request, prompt_ids

from leviathan_worker.hashing import block_hashes
from leviathan_worker.pb import worker_pb2 as pb

GPU, CPU = pb.MEDIUM_GPU, pb.MEDIUM_CPU


def parents(recorder: EventRecorder) -> dict[int, int | None]:
    """Each stored block's parent, as the events describe it."""
    result: dict[int, int | None] = {}
    for event in recorder.events():
        if event.HasField("stored"):
            parent = event.stored.parent_hash if event.stored.HasField("parent_hash") else None
            for h in event.stored.block_hashes:
                result[h] = parent
                parent = h
    return result


def engine_state(worker) -> dict[int, set[int]]:
    return {GPU: worker.engine.cache.gpu_hashes(), CPU: worker.engine.cache.cpu_hashes()}


async def until(predicate, timeout: float = 5.0) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(poll(), timeout)


async def test_shared_prefix_hits_the_cache_and_blocks_are_published(worker, tokenizer):
    recorder = await EventRecorder.start(worker.stub)
    size = worker.block_size
    system = "You are a careful assistant for an inference platform. " * 6
    first = prompt_ids(tokenizer, "How does routing work?", system=system)
    second = prompt_ids(tokenizer, "How does batching work?", system=system)
    shared = next(i for i, (a, b) in enumerate(zip(first, second)) if a != b)
    assert shared // size >= 3

    r1 = await collect(worker.stub, make_request("first", first))
    assert r1.ready.cached_tokens == 0
    hashes = block_hashes(first, size)
    await recorder.wait_for(lambda rec: set(hashes) <= set(rec.stored(GPU)))
    chain = parents(recorder)
    assert chain[hashes[0]] is None
    assert all(chain[h] == p for p, h in zip(hashes, hashes[1:]))

    r2 = await collect(worker.stub, make_request("second", second))
    assert r2.ready.cached_tokens == (shared // size) * size
    assert r2.ready.prompt_tokens == len(second)

    await recorder.wait_for(lambda rec: set(block_hashes(second, size)) <= set(rec.stored(GPU)))
    assert [b.seq for b in recorder.kv] == list(range(1, len(recorder.kv) + 1))
    assert recorder.mirror() == engine_state(worker)
    await recorder.close()


async def test_identical_prompt_reuses_all_but_the_last_token(worker, tokenizer):
    prompt = prompt_ids(tokenizer, "Repeat after me. " * 10)
    await collect(worker.stub, make_request("once", prompt))
    again = await collect(worker.stub, make_request("twice", prompt))
    full_blocks = len(prompt) // worker.block_size * worker.block_size
    assert again.ready.cached_tokens == min(full_blocks, len(prompt) - 1)


async def test_eviction_demotes_to_cpu_then_drops(make_worker, tokenizer):
    worker = await make_worker(total_blocks=8, cpu_blocks=3)
    recorder = await EventRecorder.start(worker.stub)
    for i in range(6):
        await collect(worker.stub, make_request(f"evict-{i}", prompt_ids(tokenizer, f"Distinct question {i}. " * 4)))
    await recorder.wait_for(lambda rec: rec.mirror() == engine_state(worker))
    demoted = recorder.removed(GPU)
    assert demoted, "a full GPU tier must evict"
    # Every block evicted from the GPU went to the CPU tier next.
    assert set(demoted) <= set(recorder.stored(CPU))
    assert recorder.removed(CPU), "an overflowing CPU tier must drop blocks"
    assert len(worker.engine.cache.cpu_hashes()) <= 3
    worker.engine.cache.check_invariants()
    await recorder.close()


async def test_cpu_tier_hit_is_promoted_and_counts_as_cached(make_worker, tokenizer):
    worker = await make_worker(total_blocks=8, cpu_blocks=64)
    recorder = await EventRecorder.start(worker.stub)
    target = prompt_ids(tokenizer, "Remember this prompt for later, please. " * 5)
    hashes = block_hashes(target, worker.block_size)
    assert len(hashes) >= 2
    await collect(worker.stub, make_request("target", target))
    for i in range(10):
        if not set(hashes) & worker.engine.cache.gpu_hashes():
            break
        await collect(worker.stub, make_request(f"filler-{i}", prompt_ids(tokenizer, f"Filler request {i}. " * 4)))
    assert set(hashes) <= worker.engine.cache.cpu_hashes()

    again = await collect(worker.stub, make_request("target-again", target))
    assert again.ready.cached_tokens == min(len(hashes) * worker.block_size, len(target) - 1)
    assert set(hashes) <= worker.engine.cache.gpu_hashes()

    def promoted(rec: EventRecorder) -> bool:
        events = rec.events()
        removed_at = max(
            (i for i, e in enumerate(events) if e.HasField("removed") and e.removed.medium == CPU
             and set(hashes) <= set(e.removed.block_hashes)),
            default=None,
        )
        return removed_at is not None and any(
            e.HasField("stored") and e.stored.medium == GPU and set(hashes) <= set(e.stored.block_hashes)
            for e in events[removed_at:]
        )

    await recorder.wait_for(promoted)
    await recorder.close()


async def test_snapshot_replays_the_cache(make_worker, tokenizer):
    worker = await make_worker(total_blocks=10, cpu_blocks=6)
    live = await EventRecorder.start(worker.stub)
    system = "Shared system prompt for snapshot testing. " * 3
    for i in range(5):
        await collect(worker.stub, make_request(f"s{i}", prompt_ids(tokenizer, f"question {i}", system=system)))
    await live.wait_for(lambda rec: rec.mirror() == engine_state(worker))

    late = await EventRecorder.start(worker.stub, snapshot=True, interval_ms=50)
    await late.wait_for(lambda rec: rec.loads)  # the first report follows the snapshot
    snapshot = list(late.kv)
    assert snapshot[0].seq == 1 and snapshot[0].events[0].HasField("cleared")
    assert [b.seq for b in snapshot] == list(range(1, len(snapshot) + 1))
    assert late.mirror() == engine_state(worker) == live.mirror()
    # Chains are ordered parents first.
    seen: set[int] = set()
    cached = engine_state(worker)[GPU] | engine_state(worker)[CPU]
    for event in late.events():
        if event.HasField("stored"):
            stored = event.stored
            if stored.HasField("parent_hash") and stored.parent_hash in cached:
                assert stored.parent_hash in seen
            seen.update(stored.block_hashes)

    # Live events keep flowing after the snapshot, with contiguous sequence numbers.
    await collect(worker.stub, make_request("after", prompt_ids(tokenizer, "after the snapshot", system=system)))
    await late.wait_for(lambda rec: rec.mirror() == engine_state(worker) and len(rec.kv) > len(snapshot))
    assert [b.seq for b in late.kv] == list(range(1, len(late.kv) + 1))
    await live.close()
    await late.close()


async def test_load_reports(make_worker, tokenizer):
    worker = await make_worker(decode_step_base_us=3_000, mean_output_tokens=300)
    recorder = await EventRecorder.start(worker.stub, interval_ms=20)
    await recorder.wait_for(lambda rec: rec.loads)
    idle = recorder.loads[0]
    assert (idle.running, idle.waiting, idle.kv_usage, idle.free_blocks) == (0, 0, 0.0, 512)

    request = asyncio.create_task(collect(worker.stub, make_request("load", prompt_ids(tokenizer, "Tell me a lot."))))
    await recorder.wait_for(lambda rec: any(r.running == 1 and r.kv_usage > 0 for r in rec.loads))
    busy = next(r for r in recorder.loads if r.running == 1 and r.kv_usage > 0)
    assert busy.free_blocks < 512
    assert abs(busy.kv_usage - (512 - busy.free_blocks) / 512) < 1e-6
    await recorder.wait_for(lambda rec: any(r.decode_tokens_per_s > 0 and r.spec_accept_rate > 0 for r in rec.loads))
    result = await request
    assert result.finished.reason == pb.FINISH_REASON_STOP
    count = len(recorder.loads)
    await recorder.wait_for(lambda rec: len(rec.loads) > count)
    last = recorder.loads[-1]
    assert (last.running, last.kv_usage) == (0, 0.0)
    assert all(0.0 <= r.spec_accept_rate <= 1.0 for r in recorder.loads)
    stamps = [r.timestamp_ms for r in recorder.loads]
    assert stamps == sorted(stamps)
    await recorder.close()


async def test_subscribers_have_independent_sequences(worker, tokenizer):
    a = await EventRecorder.start(worker.stub)
    await collect(worker.stub, make_request("one", prompt_ids(tokenizer, "first " * 20)))
    await a.wait_for(lambda rec: rec.kv)
    b = await EventRecorder.start(worker.stub)
    await until(lambda: worker.engine.hub.subscriber_count == 2)
    await collect(worker.stub, make_request("two", prompt_ids(tokenizer, "second " * 20)))
    await b.wait_for(lambda rec: rec.kv)
    assert b.kv[0].seq == 1
    assert a.kv[-1].seq == len(a.kv) > b.kv[-1].seq
    await a.close()
    await b.close()
