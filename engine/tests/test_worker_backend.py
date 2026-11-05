"""The worker's lev backend: the engine behind the backend-neutral worker API."""

from __future__ import annotations

import asyncio

import pytest
import torch

from leviathan_worker.backends.lev_backend import LevConfig, LevEngine
from leviathan_worker.engine import (
    BlockStored,
    Finished,
    FinishReason,
    GenerateRequest,
    KvEventBatch,
    Ready,
    SamplingParams,
    TokenBatch,
)

ENGINE = dict(block_size=4, num_blocks=128, max_num_batched_tokens=32, max_model_len=256, dtype="float64")


def prompt(n: int, seed: int) -> list[int]:
    return torch.randint(2, 320, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


async def collect(engine: LevEngine, request: GenerateRequest) -> list:
    return [event async for event in engine.generate(request)]


def greedy(rid: str, tokens: list[int], n: int = 8) -> GenerateRequest:
    return GenerateRequest(rid, tuple(tokens), SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True))


def test_concurrent_requests_stream_ready_tokens_finished(tiny_v2):
    from lev_engine.config import EngineConfig
    from lev_engine.engine import LLMEngine
    from lev_engine.sequence import SamplingParams as LevSampling

    path, _ = tiny_v2
    direct = LLMEngine(path, EngineConfig(**ENGINE))

    async def main():
        engine = LevEngine(LevConfig("lev-0", "tiny", str(path), ENGINE))
        await engine.start()
        sub = engine.subscribe(snapshot=False, load_report_interval_ms=60_000)
        prompts = [prompt(n, s) for n, s in ((9, 1), (21, 2), (3, 3))]
        runs = await asyncio.gather(*(collect(engine, greedy(f"r{i}", p)) for i, p in enumerate(prompts)))
        for p, events in zip(prompts, runs, strict=True):
            assert isinstance(events[0], Ready) and events[0].prompt_tokens == len(p)
            assert isinstance(events[-1], Finished) and events[-1].reason is FinishReason.LENGTH
            tokens = [t for e in events if isinstance(e, TokenBatch) for t in e.token_ids]
            # The engine called directly (its transformers parity is tested elsewhere;
            # transformers' min_new_tokens would also mask EOS, which ignore_eos keeps).
            assert tokens == direct.generate([p], LevSampling(temperature=0.0, max_tokens=8, ignore_eos=True))[0]
            assert events[-1].completion_tokens == 8
        # The same prompt again is served from the prefix cache.
        again = await collect(engine, greedy("again", prompts[1]))
        assert again[0].cached_tokens == (21 - 1) // 4 * 4
        batch = await asyncio.wait_for(sub.__anext__(), 1)
        while not isinstance(batch, KvEventBatch):  # load reports come first
            batch = await asyncio.wait_for(sub.__anext__(), 1)
        assert isinstance(batch.events[0], BlockStored)
        sub.close()
        await engine.stop()

    asyncio.run(main())


def test_abort_and_eos(tiny_v2):
    path, _ = tiny_v2

    async def main():
        engine = LevEngine(LevConfig("lev-0", "tiny", str(path), ENGINE))
        await engine.start()
        stream = engine.generate(greedy("long", prompt(5, 4), n=200))
        first = await stream.__anext__()
        assert isinstance(first, Ready)
        assert await engine.abort("long")
        events = [e async for e in stream]
        assert isinstance(events[-1], Finished) and events[-1].reason is FinishReason.ABORT
        # EOS (token 1 in the tiny config) ends a request and is not streamed.
        eos_req = GenerateRequest("eos", tuple(prompt(5, 5)), SamplingParams(temperature=0.0, max_tokens=50))
        events = await collect(engine, eos_req)
        tokens = [t for e in events if isinstance(e, TokenBatch) for t in e.token_ids]
        assert 1 not in tokens
        await engine.stop()

    asyncio.run(main())


def test_requests_refused_up_front(tiny_v2):
    from leviathan_worker.engine import InvalidRequestError, OutputConstraint, ConstraintKind

    path, _ = tiny_v2

    async def main():
        engine = LevEngine(LevConfig("lev-0", "tiny", str(path), ENGINE))
        await engine.start()
        with pytest.raises(InvalidRequestError):
            await collect(engine, greedy("big", [5] * 300))
        bad = GenerateRequest("c", (5, 6), constraint=OutputConstraint(ConstraintKind.JSON, "{}"))
        with pytest.raises(InvalidRequestError):
            await collect(engine, bad)
        await engine.stop()

    asyncio.run(main())
