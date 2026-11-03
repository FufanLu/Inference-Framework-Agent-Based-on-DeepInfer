"""The sim scheduler under contention, driven through the Engine interface directly."""

from __future__ import annotations

import asyncio
import contextlib

from support import TIMEOUT, prompt_ids

from leviathan_worker.backends.sim import SimEngine
from leviathan_worker.backends.sim_profile import SimProfile
from leviathan_worker.engine import (
    Finished,
    FinishReason,
    GenerateRequest,
    KvEventBatch,
    KvTransfer,
    Medium,
    Ready,
    Role,
    SamplingParams,
    TokenBatch,
)
from leviathan_worker.events import CacheMirror


async def start_engine(tokenizer, **overrides) -> SimEngine:
    engine = SimEngine(profile=SimProfile.load("fast-dev").override(**overrides), tokenizer=tokenizer)
    await engine.start()
    return engine


async def run(engine: SimEngine, request: GenerateRequest) -> list:
    async def consume() -> list:
        return [event async for event in engine.generate(request)]

    return await asyncio.wait_for(consume(), TIMEOUT)


def tokens(events: list) -> list[int]:
    return [t for e in events if isinstance(e, TokenBatch) for t in e.token_ids]


async def test_concurrent_requests_under_kv_pressure(tokenizer):
    engine = await start_engine(tokenizer, total_blocks=48, cpu_blocks=16, max_num_seqs=8)
    mirror = CacheMirror()
    subscription = engine.subscribe(snapshot=False, load_report_interval_ms=60_000)

    async def follow() -> None:
        async for event in subscription:
            if isinstance(event, KvEventBatch):
                mirror.apply(event.events)

    follower = asyncio.create_task(follow())
    systems = [f"System prompt number {k}. " * 8 for k in range(3)]
    requests = [
        GenerateRequest(
            request_id=f"r{i}",
            prompt_token_ids=tuple(prompt_ids(tokenizer, f"question {i}", system=systems[i % 3])),
            sampling=SamplingParams(seed=i, max_tokens=30 if i % 4 == 0 else 0),
        )
        for i in range(40)
    ]
    results = await asyncio.gather(*(run(engine, r) for r in requests))
    for request, events in zip(requests, results):
        assert isinstance(events[0], Ready) and isinstance(events[-1], Finished)
        assert events[-1].reason in (FinishReason.STOP, FinishReason.LENGTH)
        assert events[-1].completion_tokens == len(tokens(events))
        if request.sampling.max_tokens:
            assert len(tokens(events)) <= request.sampling.max_tokens
    assert sum(events[0].cached_tokens > 0 for events in results) >= 30
    engine.cache.check_invariants()
    assert engine.cache.used_blocks == 0
    assert engine.snapshot_state()["requests"] == 0

    async def synced() -> None:
        while mirror.blocks(Medium.GPU) != engine.cache.gpu_hashes() or mirror.blocks(Medium.CPU) != engine.cache.cpu_hashes():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(synced(), TIMEOUT)

    # One at a time on a fresh engine, every request produces the same tokens.
    fresh = await start_engine(tokenizer, total_blocks=48, cpu_blocks=16, max_num_seqs=8)
    for request, events in zip(requests, results):
        assert tokens(await run(fresh, request)) == tokens(events)
    follower.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await follower
    await engine.stop()
    await fresh.stop()


async def test_aborting_a_waiting_request(tokenizer):
    engine = await start_engine(tokenizer, max_num_seqs=1, decode_step_base_us=10_000, mean_output_tokens=200)
    first = GenerateRequest("first", tuple(prompt_ids(tokenizer, "one")), SamplingParams(seed=1))
    second = GenerateRequest("second", tuple(prompt_ids(tokenizer, "two")), SamplingParams(seed=2))
    first_task = asyncio.create_task(run(engine, first))
    second_task = asyncio.create_task(run(engine, second))
    while engine.snapshot_state()["waiting"] != 1:
        await asyncio.sleep(0.001)
    assert await engine.abort("second")
    events = await second_task
    assert len(events) == 1 and events[0].reason is FinishReason.ABORT and events[0].completion_tokens == 0
    assert (await first_task)[-1].reason is FinishReason.STOP
    await engine.stop()


async def test_aborting_during_a_kv_transfer_releases_the_blocks(tokenizer):
    prefill = SimEngine(profile=SimProfile.load("fast-dev"), tokenizer=tokenizer, role=Role.PREFILL)
    await prefill.start()
    prompt = tuple(prompt_ids(tokenizer, "context " * 60))
    ready = (await run(prefill, GenerateRequest("p", prompt, stage=Role.PREFILL)))[0]
    assert isinstance(ready.kv_transfer, KvTransfer)

    decode = await start_engine(tokenizer, kv_transfer_gbps=0.001)  # a transfer that takes seconds
    task = asyncio.create_task(run(decode, GenerateRequest("d", prompt, stage=Role.DECODE, kv_transfer=ready.kv_transfer)))
    while decode.snapshot_state()["running"] != 1:
        await asyncio.sleep(0.001)
    assert decode.cache.used_blocks > 0
    assert await decode.abort("d")
    events = await task
    assert [type(e) for e in events] == [Finished] and events[0].reason is FinishReason.ABORT
    assert decode.cache.used_blocks == 0
    decode.cache.check_invariants()
    await prefill.stop()
    await decode.stop()


async def test_stop_aborts_in_flight_requests(tokenizer):
    engine = await start_engine(tokenizer, decode_step_base_us=10_000, mean_output_tokens=200)
    task = asyncio.create_task(run(engine, GenerateRequest("long", tuple(prompt_ids(tokenizer, "hi")))))
    while engine.snapshot_state()["running"] != 1:
        await asyncio.sleep(0.001)
    await engine.stop()
    events = await task
    assert events[-1].reason is FinishReason.ABORT and events[-1].error == "worker stopped"
    assert not engine.health().ready


async def test_lower_priority_value_is_admitted_first(tokenizer):
    # One sequence at a time: the waiting queue order is the admission order.
    engine = await start_engine(tokenizer, max_num_seqs=1)
    order: list[str] = []

    async def track(request: GenerateRequest) -> None:
        async for event in engine.generate(request):
            if isinstance(event, Ready):
                order.append(request.request_id)

    def request(name: str, priority: int) -> GenerateRequest:
        return GenerateRequest(
            request_id=name,
            prompt_token_ids=tuple(prompt_ids(tokenizer, f"question {name}")),
            sampling=SamplingParams(seed=1, max_tokens=8),
            priority=priority,
        )

    tasks = [asyncio.create_task(track(request("blocker", 0)))]
    await asyncio.sleep(0)
    for name in ("job-1", "job-2", "job-3"):
        tasks.append(asyncio.create_task(track(request(name, 1))))
        await asyncio.sleep(0)
    tasks.append(asyncio.create_task(track(request("interactive", 0))))
    await asyncio.wait_for(asyncio.gather(*tasks), TIMEOUT)
    assert order == ["blocker", "interactive", "job-1", "job-2", "job-3"]
    await engine.stop()
