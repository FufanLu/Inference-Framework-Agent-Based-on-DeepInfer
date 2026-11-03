"""A simulated continuous-batching engine.

One asyncio task plays the engine's scheduler. Each iteration plans a step the
way vLLM's scheduler does (every decoding sequence, then prefill chunks within
the token budget, then admissions from the FIFO queue), sleeps for the step's
modelled duration, and applies its results: prompt blocks enter the prefix
cache as chunks complete, finished prefills send ``Ready`` and their first
token, and each decoding sequence emits one token plus its accepted DSpark
drafts.

Output is decided when a request arrives: synthetic text (see
:mod:`leviathan_worker.synth`) tokenized with the V4.1 tokenizer, truncated by
stop tokens, ``max_tokens`` and the context window. It depends only on the
prompt, the seed, the constraint and the hints, so it is identical whatever
else the engine is running; per-request random streams make the speculation
statistics and logprobs deterministic too.

KV loads (from the CPU tier, or from a prefill worker in PD disaggregation)
happen asynchronously per request, like vLLM's KV connectors: a request waits
for its fetch timer while the batch keeps running.
"""

from __future__ import annotations

import asyncio
import contextlib
import enum
import json
import logging
import math
import random
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

from ..engine import (
    DuplicateRequestError,
    Engine,
    EngineInfo,
    EventSubscription,
    Finished,
    FinishReason,
    GenerateEvent,
    GenerateRequest,
    Health,
    InvalidRequestError,
    KvTransfer,
    LoadCounters,
    OverloadedError,
    Ready,
    Role,
    TokenBatch,
    UnavailableError,
)
from ..events import DEFAULT_QUEUE_SIZE, EventHub
from ..hashing import ROOT_SEED, block_hashes
from ..kvcache import Lease, TieredPrefixCache
from ..synth import OutputSynthesizer, SynthesisError, derive_seed, fake_logprobs, request_seed
from ..tokens import V41Tokenizer
from .sim_profile import SimProfile

log = logging.getLogger(__name__)

HANDLE_FORMAT = "leviathan-sim-kv/1"
DEFAULT_MAX_WAITING = 1024

T = TypeVar("T")


class Phase(enum.Enum):
    WAITING = "waiting"  # queued, holds no KV
    FETCHING = "fetching"  # admitted, waiting for KV from the CPU tier or a prefill worker
    PREFILLING = "prefilling"
    DECODING = "decoding"
    FINISHED = "finished"


@dataclass(frozen=True)
class _TransferPlan:
    """The prompt blocks a DECODE-stage request receives from its prefill worker."""

    source_worker_id: str
    blocks: int


class _OutputCollector:
    """The unconsumed output of one request.

    Token batches merge while the consumer lags (as vLLM's output collector
    does), so memory is bounded by the request's own output, not by time.
    """

    def __init__(self) -> None:
        self._ready: Ready | None = None
        self._tokens: TokenBatch | None = None
        self._finished: Finished | None = None
        self._signal = asyncio.Event()

    def put_ready(self, ready: Ready) -> None:
        self._ready = ready
        self._signal.set()

    def put_tokens(self, batch: TokenBatch) -> None:
        self._tokens = batch if self._tokens is None else self._tokens.merged(batch)
        self._signal.set()

    def put_finished(self, finished: Finished) -> None:
        self._finished = finished
        self._signal.set()

    async def get(self) -> GenerateEvent:
        while True:
            event: GenerateEvent | None = None
            if self._ready is not None:
                event, self._ready = self._ready, None
            elif self._tokens is not None:
                event, self._tokens = self._tokens, None
            elif self._finished is not None:
                event, self._finished = self._finished, None
            if event is not None:
                return event
            self._signal.clear()
            await self._signal.wait()


@dataclass(eq=False)
class _SimRequest:
    request: GenerateRequest
    prompt_hashes: list[int]
    #: Planned output tokens and the reason to report once they are all emitted.
    output: tuple[int, ...]
    finish_reason: FinishReason
    transfer: _TransferPlan | None
    spec_rng: random.Random
    logprob_rng: random.Random | None
    arrival: float
    out: _OutputCollector = field(default_factory=_OutputCollector)
    phase: Phase = Phase.WAITING
    lease: Lease | None = None
    cached_tokens: int = 0
    #: Prompt tokens whose KV exists (cached or computed).
    computed: int = 0
    emitted: int = 0
    admitted_at: float = 0.0
    ready_at: float = 0.0
    timer: asyncio.TimerHandle | None = None

    @property
    def id(self) -> str:
        return self.request.request_id

    @property
    def prompt(self) -> tuple[int, ...]:
        return self.request.prompt_token_ids


@dataclass
class _Step:
    decode: list[_SimRequest]
    prefill: list[tuple[_SimRequest, int]]
    duration_us: float


def _us(seconds: float) -> int:
    return max(0, int(seconds * 1_000_000))


class SimEngine(Engine):
    def __init__(
        self,
        *,
        profile: SimProfile,
        tokenizer: V41Tokenizer,
        worker_id: str = "sim-0",
        role: Role = Role.UNIFIED,
        model: str = "DeepSeek-V4.1-Flash",
        max_waiting: int = DEFAULT_MAX_WAITING,
        event_queue_size: int = DEFAULT_QUEUE_SIZE,
    ) -> None:
        self.profile = profile
        self._tok = tokenizer
        self._worker_id = worker_id
        self._role = role
        self._model = model
        self._max_waiting = max_waiting
        self._block_size = profile.block_size
        self._context_limit = profile.effective_context_limit
        self.cache = TieredPrefixCache(profile.total_blocks, profile.cpu_blocks)
        self.hub = EventHub(self.load_counters, queue_size=event_queue_size)
        self._synth = OutputSynthesizer(tokenizer, mean_output_tokens=profile.mean_output_tokens)
        self._requests: dict[str, _SimRequest] = {}
        self._waiting: deque[_SimRequest] = deque()
        self._running: list[_SimRequest] = []
        self._wakeup = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._stopped = False
        self._prefill_tokens = 0
        self._decode_tokens = 0
        self._draft_tokens = 0
        self._accepted_tokens = 0

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.get_running_loop().create_task(self._run(), name=f"sim-scheduler:{self._worker_id}")
        log.info(
            "sim engine started",
            extra={"worker_id": self._worker_id, "role": self._role.name.lower(), "profile": self.profile.name,
                   "block_size": self._block_size, "total_blocks": self.profile.total_blocks,
                   "cpu_blocks": self.profile.cpu_blocks, "context_limit": self._context_limit},
        )

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        for r in list(self._requests.values()):
            self._finish(r, FinishReason.ABORT, error="worker stopped")
        self._flush_events()
        self.hub.close()

    def info(self) -> EngineInfo:
        return EngineInfo(
            worker_id=self._worker_id,
            backend="sim",
            model=self._model,
            role=self._role,
            block_size=self._block_size,
            context_limit=self._context_limit,
            total_blocks=self.profile.total_blocks,
            supports_kv_events=True,
            supports_constraints=True,
            supports_logprobs=True,
            deterministic=True,
            labels={"profile": self.profile.name},
        )

    def health(self) -> Health:
        if self._stopped:
            return Health(ready=False, status="stopped")
        if self._task is None:
            return Health(ready=False, status="not started")
        if self._task.done():
            return Health(ready=False, status="scheduler exited")
        return Health(ready=True, status="serving")

    def load_counters(self) -> LoadCounters:
        return LoadCounters(
            running=len(self._running),
            waiting=len(self._waiting),
            kv_usage=self.cache.usage,
            free_blocks=self.cache.free_blocks,
            prefill_tokens=self._prefill_tokens,
            decode_tokens=self._decode_tokens,
            spec_draft_tokens=self._draft_tokens,
            spec_accepted_tokens=self._accepted_tokens,
        )

    def subscribe(self, *, snapshot: bool, load_report_interval_ms: int) -> EventSubscription:
        if self._stopped:
            raise UnavailableError("engine is stopped")
        return self.hub.subscribe(snapshot=snapshot, load_report_interval_ms=load_report_interval_ms)

    # ------------------------------------------------------------------ requests

    async def generate(self, request: GenerateRequest) -> AsyncIterator[GenerateEvent]:
        r = self._submit(request)
        try:
            while True:
                event = await r.out.get()
                yield event
                if isinstance(event, Finished):
                    return
        finally:
            if r.phase is not Phase.FINISHED:  # the consumer went away
                self._finish(r, FinishReason.ABORT, error="cancelled")
                self._flush_events()

    async def abort(self, request_id: str) -> bool:
        r = self._requests.get(request_id)
        if r is None:
            return False
        self._finish(r, FinishReason.ABORT, error="aborted")
        self._flush_events()
        return True

    def _submit(self, req: GenerateRequest) -> _SimRequest:
        if self._stopped or self._task is None:
            raise UnavailableError("engine is not running")
        if not req.request_id:
            raise InvalidRequestError("request_id is required")
        if req.request_id in self._requests:
            raise DuplicateRequestError(f"request {req.request_id!r} is already running")
        prompt = req.prompt_token_ids
        if not prompt:
            raise InvalidRequestError("prompt_token_ids is empty")
        if len(prompt) >= self._context_limit:
            raise InvalidRequestError(
                f"prompt has {len(prompt)} tokens but the context limit is {self._context_limit}"
            )
        if min(prompt) < 0 or max(prompt) >= self._tok.vocab_size:
            bad = next(t for t in prompt if not 0 <= t < self._tok.vocab_size)
            raise InvalidRequestError(f"prompt token ID {bad} is outside the vocabulary")
        if len(self._waiting) >= self._max_waiting:
            raise OverloadedError(f"{len(self._waiting)} requests are already waiting")

        hashes = block_hashes(prompt, self._block_size)
        transfer = None
        if req.stage is Role.DECODE and req.kv_transfer is not None:
            transfer = self._parse_transfer(req.kv_transfer, hashes)
        seed = request_seed(prompt, req.sampling.seed, req.constraint, req.hints)
        output, reason = self._plan_output(req, seed)
        r = _SimRequest(
            request=req,
            prompt_hashes=hashes,
            output=output,
            finish_reason=reason,
            transfer=transfer,
            spec_rng=random.Random(derive_seed(seed, "spec")),
            logprob_rng=random.Random(derive_seed(seed, "logprobs")) if req.sampling.logprobs else None,
            arrival=time.monotonic(),
        )
        self._requests[r.id] = r
        self._enqueue(r)
        self._wakeup.set()
        log.debug(
            "request queued",
            extra={"request_id": r.id, "stage": req.stage.name.lower(), "prompt_tokens": len(prompt),
                   "planned_tokens": len(output)},
        )
        return r

    def _plan_output(self, req: GenerateRequest, seed: int) -> tuple[tuple[int, ...], FinishReason]:
        """The tokens this request will emit and the reason it will finish with."""
        if req.stage is Role.PREFILL:
            return (), FinishReason.STOP
        sampling = req.sampling
        room = self._context_limit - len(req.prompt_token_ids)
        limit = min(sampling.max_tokens, room) if sampling.max_tokens else room
        try:
            synthesized = self._synth.synthesize(
                req.prompt_token_ids,
                random.Random(derive_seed(seed, "content")),
                req.constraint,
                req.hints,
                min_tokens=limit if sampling.ignore_eos and sampling.max_tokens else 0,
            )
        except SynthesisError as exc:
            raise InvalidRequestError(str(exc)) from None
        ids = synthesized.token_ids
        stops = set(sampling.stop_token_ids)
        if not sampling.ignore_eos:
            stops.add(self._tok.eos_id)
        cut = next((i for i, t in enumerate(ids) if t in stops), None)
        if cut is not None:
            ids = ids[:cut]
        if len(ids) >= limit:
            return ids[:limit], FinishReason.LENGTH
        return ids, FinishReason.STOP

    def _parse_transfer(self, kv: KvTransfer, hashes: Sequence[int]) -> _TransferPlan:
        try:
            handle = json.loads(kv.handle)
        except json.JSONDecodeError:
            raise InvalidRequestError("kv_transfer.handle is not valid JSON") from None
        if not isinstance(handle, dict) or handle.get("format") != HANDLE_FORMAT:
            raise InvalidRequestError(f"kv_transfer.handle is not a {HANDLE_FORMAT} handle from a sim worker")
        if handle.get("block_size") != self._block_size:
            raise InvalidRequestError(
                f"kv_transfer block size {handle.get('block_size')} differs from this worker's {self._block_size}"
            )
        blocks, parents = handle.get("block_hashes"), handle.get("parents")
        if not isinstance(blocks, list) or not isinstance(parents, list) or len(parents) != len(blocks):
            raise InvalidRequestError("kv_transfer.handle needs block_hashes and parents of equal length")
        if blocks != list(hashes[: len(blocks)]) or len(blocks) > len(hashes):
            raise InvalidRequestError("kv_transfer.handle does not describe this prompt")
        if parents != [None, *blocks[:-1]][: len(blocks)]:
            raise InvalidRequestError("kv_transfer.handle parents do not chain the blocks")
        return _TransferPlan(source_worker_id=kv.source_worker_id, blocks=len(blocks))

    def _handoff(self, r: _SimRequest) -> KvTransfer:
        """The KV hand-off a PREFILL-stage request gives its decode worker."""
        hashes = r.prompt_hashes
        handle = {
            "format": HANDLE_FORMAT,
            "worker_id": self._worker_id,
            "block_size": self._block_size,
            "block_hashes": hashes,
            "parents": [None, *hashes[:-1]] if hashes else [],
        }
        return KvTransfer(
            source_worker_id=self._worker_id,
            handle=json.dumps(handle, separators=(",", ":")),
            num_tokens=len(hashes) * self._block_size,
        )

    # ------------------------------------------------------------------ scheduler

    async def _run(self) -> None:
        while True:
            step = self._guarded(self._plan_step)
            self._flush_events()
            if step is None:
                self._wakeup.clear()
                await self._wakeup.wait()
                continue
            await asyncio.sleep(step.duration_us / 1_000_000)
            self._guarded(lambda: self._complete_step(step))
            self._flush_events()

    def _guarded(self, fn: Callable[[], T]) -> T | None:
        """Run one scheduler phase; a bug fails the in-flight requests instead of killing the loop."""
        try:
            return fn()
        except Exception as exc:
            log.exception("sim scheduler failed; failing in-flight requests")
            for r in list(self._requests.values()):
                with contextlib.suppress(Exception):
                    self._finish(r, FinishReason.ERROR, error=f"internal error: {exc}")
            return None

    def _plan_step(self) -> _Step | None:
        p = self.profile
        budget = p.max_batched_tokens
        decode = [r for r in self._running if r.phase is Phase.DECODING]
        budget -= len(decode) * (1 + p.spec_num_draft)
        prefill: list[tuple[_SimRequest, int]] = []

        def schedule_chunk(r: _SimRequest) -> None:
            nonlocal budget
            if r.phase is Phase.PREFILLING and budget > 0:
                chunk = min(len(r.prompt) - r.computed, budget)
                prefill.append((r, chunk))
                budget -= chunk

        for r in self._running:
            schedule_chunk(r)
        now = time.monotonic()
        while self._waiting and len(self._running) < p.max_num_seqs:
            r = self._waiting[0]
            if r.transfer is None and budget <= 0:
                break
            if not self._admit(r, now):
                break  # strict FIFO: the head waits for blocks, later requests wait behind it
            self._waiting.popleft()
            self._running.append(r)
            schedule_chunk(r)
        if not decode and not prefill:
            return None
        prefill_tokens = sum(chunk for _, chunk in prefill)
        duration = 0.0
        if decode:
            duration += p.decode_step_base_us + p.decode_step_per_seq_us * len(decode)
        if prefill_tokens:
            duration += p.prefill_us_per_token * prefill_tokens + (0.0 if decode else p.prefill_base_us)
        return _Step(decode=decode, prefill=prefill, duration_us=duration)

    def _enqueue(self, r: _SimRequest) -> None:
        """FIFO within a priority; a lower ``priority`` value is served first
        (vLLM's ``--scheduling-policy priority``). Interactive requests use 0
        and background jobs a higher value, so jobs never delay them in the
        waiting queue. Requests already running are not preempted."""
        priority = r.request.priority
        if not self._waiting or self._waiting[-1].request.priority <= priority:
            self._waiting.append(r)
            return
        for index, queued in enumerate(self._waiting):
            if queued.request.priority > priority:
                self._waiting.insert(index, r)
                return
        self._waiting.append(r)

    def _admit(self, r: _SimRequest, now: float) -> bool:
        p, size = self.profile, self._block_size
        local = self.cache.match(r.prompt_hashes)
        hits = r.prompt_hashes[:local]
        budget = math.ceil((len(r.prompt) + len(r.output)) / size)
        if self.cache.slots_needed(hits, budget) > self.cache.free_blocks:
            return False
        r.lease = self.cache.acquire(hits, budget)
        r.admitted_at = now
        transferred = max(0, r.transfer.blocks - local) if r.transfer else 0
        # The last prompt token is always computed: its logits give the first output token.
        r.cached_tokens = min((local + transferred) * size, len(r.prompt) - 1)
        fetch_us = r.lease.loaded_from_cpu * p.cpu_load_us_per_block
        fetch_us += transferred * size * p.kv_bytes_per_token * 8 / (p.kv_transfer_gbps * 1_000)
        if fetch_us > 0:
            r.phase = Phase.FETCHING
            r.timer = asyncio.get_running_loop().call_later(fetch_us / 1_000_000, self._on_fetched, r)
        else:
            self._start_prefill(r)
        return True

    def _on_fetched(self, r: _SimRequest) -> None:
        r.timer = None
        if r.phase is not Phase.FETCHING:
            return
        self._guarded(lambda: self._start_prefill(r))
        self._flush_events()
        self._wakeup.set()

    def _start_prefill(self, r: _SimRequest) -> None:
        assert r.lease is not None
        if r.transfer is not None and r.transfer.blocks > len(r.lease.hashes):
            arrived = r.prompt_hashes[len(r.lease.hashes) : r.transfer.blocks]
            self.cache.commit(r.lease, arrived, r.lease.last_hash)
        r.computed = r.cached_tokens
        r.phase = Phase.PREFILLING

    def _complete_step(self, step: _Step) -> None:
        now = time.monotonic()
        for r in step.decode:
            if r.phase is Phase.DECODING:
                self._decode(r, now)
        for r, chunk in step.prefill:
            if r.phase is Phase.PREFILLING:
                self._prefill(r, chunk, now)

    def _prefill(self, r: _SimRequest, chunk: int, now: float) -> None:
        assert r.lease is not None
        size = self._block_size
        r.computed += chunk
        self._prefill_tokens += chunk
        full = r.computed // size
        if full > len(r.lease.hashes):
            self.cache.commit(r.lease, r.prompt_hashes[len(r.lease.hashes) : full], r.lease.last_hash)
        self.cache.allocate(r.lease, math.ceil(r.computed / size))
        if r.computed < len(r.prompt):
            return
        r.ready_at = now
        prefill_only = r.request.stage is Role.PREFILL
        r.out.put_ready(
            Ready(
                prompt_tokens=len(r.prompt),
                cached_tokens=r.cached_tokens,
                queue_time_us=_us(r.admitted_at - r.arrival),
                prefill_time_us=_us(now - r.admitted_at),
                kv_transfer=self._handoff(r) if prefill_only else None,
            )
        )
        r.phase = Phase.DECODING
        if prefill_only or not r.output:
            self._finish(r, r.finish_reason, now)
        else:
            self._emit(r, 1, drafted=0, accepted=0, now=now)  # the prefill step samples the first token

    def _decode(self, r: _SimRequest, now: float) -> None:
        p = self.profile
        accepted = 0
        while accepted < p.spec_num_draft and r.spec_rng.random() < p.spec_accept_prob:
            accepted += 1
        n = min(1 + accepted, len(r.output) - r.emitted)
        self._emit(r, n, drafted=p.spec_num_draft, accepted=min(accepted, n - 1), now=now)

    def _emit(self, r: _SimRequest, n: int, *, drafted: int, accepted: int, now: float) -> None:
        assert r.lease is not None
        ids = r.output[r.emitted : r.emitted + n]
        logprobs = ()
        if r.logprob_rng is not None:
            logprobs = fake_logprobs(
                ids, r.logprob_rng, top_n=r.request.sampling.top_logprobs, plain_vocab_size=self._tok.plain_vocab_size
            )
        r.emitted += n
        self._decode_tokens += n
        self._draft_tokens += drafted
        self._accepted_tokens += accepted
        r.out.put_tokens(TokenBatch(token_ids=ids, logprobs=logprobs, draft_tokens=drafted, accepted_tokens=accepted))
        if r.emitted >= len(r.output):
            self._finish(r, r.finish_reason, now)
        else:
            self.cache.allocate(r.lease, math.ceil((len(r.prompt) + r.emitted) / self._block_size))

    def _finish(self, r: _SimRequest, reason: FinishReason, now: float | None = None, error: str = "") -> None:
        if r.phase is Phase.FINISHED:
            return
        now = time.monotonic() if now is None else now
        if r.timer is not None:
            r.timer.cancel()
            r.timer = None
        if r.lease is not None:
            if reason in (FinishReason.STOP, FinishReason.LENGTH) and r.emitted:
                self._store_output_blocks(r)
            self.cache.release(r.lease)
        if r.phase is Phase.WAITING:
            self._waiting.remove(r)
        elif r in self._running:
            self._running.remove(r)
        r.phase = Phase.FINISHED
        self._requests.pop(r.id, None)
        decode_us = _us(now - r.ready_at) if r.ready_at else 0
        r.out.put_finished(Finished(reason=reason, completion_tokens=r.emitted, error=error, decode_time_us=decode_us))
        self._wakeup.set()
        log.debug(
            "request finished",
            extra={"request_id": r.id, "reason": reason.name.lower(), "prompt_tokens": len(r.prompt),
                   "cached_tokens": r.cached_tokens, "completion_tokens": r.emitted,
                   "queue_us": _us(r.admitted_at - r.arrival) if r.admitted_at else _us(now - r.arrival),
                   "decode_us": decode_us, "error": error or None},
        )

    def _store_output_blocks(self, r: _SimRequest) -> None:
        """Cache the full blocks of prompt + output, as vLLM does when a request finishes."""
        assert r.lease is not None
        size = self._block_size
        held = len(r.lease.hashes)
        full = (len(r.prompt) + r.emitted) // size
        if full <= held:
            return
        start, end = held * size, full * size
        n = len(r.prompt)
        tokens = list(r.prompt[min(start, n) : min(end, n)]) + list(r.output[max(0, start - n) : max(0, end - n)])
        parent = r.lease.last_hash
        new = block_hashes(tokens, size, parent=ROOT_SEED if parent is None else parent)
        self.cache.commit(r.lease, new, parent)

    def _flush_events(self) -> None:
        events = self.cache.drain_events()
        if events:
            self.hub.publish(events)

    # ------------------------------------------------------------------ introspection (tests, debugging)

    def snapshot_state(self) -> dict[str, Any]:
        return {
            "running": len(self._running),
            "waiting": len(self._waiting),
            "requests": len(self._requests),
            "used_blocks": self.cache.used_blocks,
            "free_blocks": self.cache.free_blocks,
        }
