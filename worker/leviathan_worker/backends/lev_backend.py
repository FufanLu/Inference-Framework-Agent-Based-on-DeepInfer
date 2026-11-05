"""lev backend: Leviathan's own engine (`engine/lev_engine`) behind the worker API.

The engine is synchronous; one asyncio task owns it. Requests and aborts are
queued and applied between steps, each step runs in a thread so the event
loop keeps serving gRPC, and the step's outputs are fanned out to the
per-request streams. KV events come from the engine's block pool already in
the router's hash convention, so they are published as they are.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from ..engine import (
    BlockRemoved,
    BlockStored,
    Engine,
    EngineInfo,
    EventSubscription,
    Finished,
    FinishReason,
    GenerateEvent,
    GenerateRequest,
    Health,
    InvalidRequestError,
    LoadCounters,
    Medium,
    Ready,
    Role,
    TokenBatch,
    TokenLogprob,
    UnavailableError,
    DuplicateRequestError,
)
from ..events import EventHub

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LevConfig:
    worker_id: str
    model_name: str
    model_path: str
    #: `lev_engine.config.EngineConfig` fields.
    engine: dict[str, Any] = field(default_factory=dict)
    role: Role = Role.UNIFIED


@dataclass
class _Stream:
    request: GenerateRequest
    queue: asyncio.Queue
    submitted: float
    started: float | None = None
    first_token: float | None = None
    emitted: int = 0


class LevEngine(Engine):
    def __init__(self, config: LevConfig) -> None:
        if config.role is not Role.UNIFIED:
            raise ValueError("the lev backend serves unified workers only (PD comes later)")
        self._config = config
        self._engine: Any = None
        self._hub = EventHub(self._load)
        self._streams: dict[str, _Stream] = {}
        self._pending: list[GenerateRequest] = []
        self._aborts: list[str] = []
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._ready = False
        self._error: str | None = None

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        from lev_engine.config import EngineConfig  # noqa: PLC0415 - needs torch
        from lev_engine.engine import LLMEngine  # noqa: PLC0415

        cfg = EngineConfig(**self._config.engine)
        self._engine = await asyncio.to_thread(LLMEngine, self._config.model_path, cfg)
        self._task = asyncio.create_task(self._loop(), name="lev-engine-loop")
        self._ready = True
        log.info(
            "lev engine ready: %s, %d blocks of %d, backend %s%s",
            self._config.model_path,
            self._engine.pool.num_blocks,
            cfg.block_size,
            cfg.kernel_backend,
            " (batch-invariant)" if cfg.batch_invariant else "",
        )

    async def stop(self) -> None:
        self._ready = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        for stream in self._streams.values():
            stream.queue.put_nowait(Finished(FinishReason.ABORT, stream.emitted, "worker stopped"))
        self._streams.clear()
        self._hub.close()

    # ------------------------------------------------------------------ info

    def info(self) -> EngineInfo:
        cfg = self._engine.config if self._engine else None
        return EngineInfo(
            worker_id=self._config.worker_id,
            backend="lev",
            model=self._config.model_name,
            role=self._config.role,
            block_size=cfg.block_size if cfg else 0,
            context_limit=cfg.max_model_len if cfg else 0,
            total_blocks=self._engine.pool.num_blocks if self._engine else 0,
            supports_kv_events=True,
            supports_constraints=False,
            supports_logprobs=True,
            deterministic=bool(cfg and cfg.batch_invariant),
            labels={"kernels": cfg.kernel_backend if cfg else ""},
        )

    def health(self) -> Health:
        if self._error:
            return Health(ready=False, status=f"engine dead: {self._error}")
        if not self._ready:
            return Health(ready=False, status="starting")
        return Health(ready=True, status="ok")

    def _load(self) -> LoadCounters:
        e = self._engine
        if e is None:
            return LoadCounters(0, 0, 0.0, 0, 0, 0, 0, 0)
        return LoadCounters(
            running=len(e.scheduler.running),
            waiting=len(e.scheduler.waiting) + len(self._pending),
            kv_usage=e.pool.usage(),
            free_blocks=e.pool.num_free,
            prefill_tokens=e.stats.prompt_tokens - e.stats.cached_prompt_tokens,
            decode_tokens=e.stats.generated_tokens,
            spec_draft_tokens=0,
            spec_accepted_tokens=0,
        )

    def subscribe(self, *, snapshot: bool, load_report_interval_ms: int) -> EventSubscription:
        return self._hub.subscribe(snapshot=snapshot, load_report_interval_ms=load_report_interval_ms)

    async def abort(self, request_id: str) -> bool:
        if request_id not in self._streams:
            return False
        self._aborts.append(request_id)
        self._wake.set()
        return True

    # ------------------------------------------------------------------ generate

    async def generate(self, request: GenerateRequest) -> AsyncIterator[GenerateEvent]:
        if not self._ready:
            raise UnavailableError("lev engine is not ready")
        if not request.prompt_token_ids:
            raise InvalidRequestError("empty prompt")
        if request.request_id in self._streams:
            raise DuplicateRequestError(request.request_id)
        limit = self._engine.config.max_model_len
        if len(request.prompt_token_ids) >= limit:
            raise InvalidRequestError(f"prompt of {len(request.prompt_token_ids)} tokens exceeds {limit}")
        if request.constraint is not None:
            raise InvalidRequestError("the lev backend does not support output constraints yet")
        stream = _Stream(request, asyncio.Queue(), time.monotonic())
        self._streams[request.request_id] = stream
        self._pending.append(request)
        self._wake.set()
        try:
            while True:
                event = await stream.queue.get()
                yield event
                if isinstance(event, Finished):
                    return
        finally:
            if request.request_id in self._streams:  # the consumer went away
                self._aborts.append(request.request_id)
                self._wake.set()

    # ------------------------------------------------------------------ engine loop

    def _admit(self) -> None:
        from lev_engine.sequence import SamplingParams  # noqa: PLC0415

        e = self._engine
        for request in self._pending:
            s = request.sampling
            params = SamplingParams(
                temperature=s.temperature,
                top_p=s.top_p,
                max_tokens=s.max_tokens,
                seed=s.seed,
                ignore_eos=s.ignore_eos,
                stop_token_ids=tuple(s.stop_token_ids),
                logprobs=s.logprobs,
            )
            try:
                e.add_request(request.request_id, list(request.prompt_token_ids), params, request.priority)
            except ValueError as error:
                stream = self._streams.pop(request.request_id, None)
                if stream is not None:
                    stream.queue.put_nowait(Finished(FinishReason.ERROR, 0, str(error)))
                continue
            self._streams[request.request_id].started = time.monotonic()
        self._pending.clear()
        for request_id in self._aborts:
            e.abort(request_id)
            stream = self._streams.pop(request_id, None)
            if stream is not None:
                stream.queue.put_nowait(Finished(FinishReason.ABORT, stream.emitted))
        self._aborts.clear()

    def _dispatch(self, outputs: list[Any]) -> None:
        from lev_engine.sequence import FinishReason as LevFinish  # noqa: PLC0415

        now = time.monotonic()
        eos = self._engine.spec.eos_token_id
        for out in outputs:
            stream = self._streams.get(out.request_id)
            if stream is None:
                continue
            req = stream.request
            if stream.first_token is None:
                stream.first_token = now
                started = stream.started or stream.submitted
                stream.queue.put_nowait(
                    Ready(
                        prompt_tokens=len(req.prompt_token_ids),
                        cached_tokens=out.num_cached_prompt or 0,
                        queue_time_us=int((started - stream.submitted) * 1e6),
                        prefill_time_us=int((now - started) * 1e6),
                    )
                )
            tokens = list(out.new_tokens)
            logprobs = [TokenLogprob(lp) for lp in out.logprobs] if out.logprobs else []
            if out.finished and out.finish_reason is LevFinish.STOP and tokens:
                last = tokens[-1]
                if (last == eos and not req.sampling.ignore_eos) or last in req.sampling.stop_token_ids:
                    tokens, logprobs = tokens[:-1], logprobs[:-1]
            if tokens:
                stream.emitted += len(tokens)
                stream.queue.put_nowait(TokenBatch(tuple(tokens), tuple(logprobs)))
            if out.finished:
                reason = FinishReason.STOP if out.finish_reason is LevFinish.STOP else FinishReason.LENGTH
                stream.queue.put_nowait(
                    Finished(reason, stream.emitted, decode_time_us=int((now - stream.first_token) * 1e6))
                )
                self._streams.pop(out.request_id, None)
                self._engine.requests.pop(out.request_id, None)

    def _publish_kv_events(self) -> None:
        from lev_engine.kv_cache import BlockStored as LevStored  # noqa: PLC0415

        events = []
        for e in self._engine.drain_events():
            if isinstance(e, LevStored):
                events.append(BlockStored(e.block_hashes, e.parent_hash, Medium.GPU))
            else:
                events.append(BlockRemoved(e.block_hashes, Medium.GPU))
        if events:
            self._hub.publish(events)

    async def _loop(self) -> None:
        e = self._engine
        try:
            while True:
                if not (self._pending or self._aborts or e.has_unfinished):
                    self._wake.clear()
                    await self._wake.wait()
                self._admit()
                if not e.has_unfinished:
                    continue
                outputs = await asyncio.to_thread(e.step)
                self._dispatch(outputs)
                self._publish_kv_events()
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - the engine is dead; fail every stream
            log.exception("lev engine step failed")
            self._error = repr(error)
            self._ready = False
            for stream in self._streams.values():
                stream.queue.put_nowait(Finished(FinishReason.ERROR, stream.emitted, self._error))
            self._streams.clear()
