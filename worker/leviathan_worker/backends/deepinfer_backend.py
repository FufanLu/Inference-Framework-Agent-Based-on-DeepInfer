"""DeepInfer backend: DeepSeek's production engine, driven online.

**Status: written against the DeepInfer source and tested with a fake engine of
the same interface; not run on real hardware.** DeepInfer's open-source
release only supports NVIDIA SM100 and Ascend 950, and this project's GPUs are
Hopper (see docs/deepinfer-status.md for the evidence).

DeepInfer ships offline inference only (``LLM.generate`` over a fixed batch).
Online serving needs the engine's step loop to stay alive and accept requests
as they arrive. This adapter does that on a dedicated thread:

```
asyncio (gRPC)                         engine thread
--------------                         -------------
generate() ── submit queue ──────────► drain: llm_engine.add_request(...)
                                       unfinished = has_unfinished_requests()
                                       outs = step() / flush_inflight_reqs()
per-request asyncio.Queue ◄── call_soon_threadsafe ── RequestOutput.clone(raw)
```

Interface used (DeepInfer ``main``, 2026-11):

- ``llm.llm_engine.add_request(user_id, request_id, sampling_params,
  prompt_token_ids: np.int32 array, multi_modal_data)``
- ``llm_engine.step()`` / ``flush_inflight_reqs()`` return raw output handles
  that must be copied with ``RequestOutput.clone`` before the next step;
- one ``RequestOutput`` per generated token: ``request_id``, ``token_id``,
  ``token_logprobs``, ``draft_tok`` / ``accept_tok`` (speculative decoding:
  a verify step's target token carries ``1 + drafts`` and ``1 + accepted``),
  ``finished``, ``finish_reason``, ``prompt_len``;
- ``llm_engine.abort_request(request_id)``; ``has_unfinished_requests()`` is a
  collective over the data-parallel group, so every rank keeps looping;
- ``SamplingParams(temperature, top_p, max_tokens, ignore_eos, top_logprobs,
  seq_seed, eos_id, so_args={"schema_type": "json"|"regex"|"ebnf", "schema"})``;
  a fixed ``seq_seed`` with ``DEEPINFER_BATCH_INVARIANT=1`` gives
  deterministic sampling.

DeepInfer's open release has no KV cache system and no published cache
events, so the worker reports ``supports_kv_events=false`` and the router
falls back to its shadow index for it. Prefill/decode disaggregation is not
wired in the open release either; only unified requests are accepted.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from ..engine import (
    ConstraintKind,
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
    Ready,
    Role,
    TokenBatch,
    TokenLogprob,
    UnavailableError,
)
from ..events import EventHub

log = logging.getLogger(__name__)


def so_args(request: GenerateRequest) -> dict[str, str] | None:
    """DeepInfer structured-output arguments for a constraint.

    Constraints apply from the first output token, so they are only used
    outside thinking mode (where reasoning would precede the constrained part).
    Tool-call blocks are left to the model, as with vLLM: their DSML tags must
    be special tokens.
    """
    constraint = request.constraint
    if constraint is None or request.hints.get("thinking") == "true":
        return None
    if constraint.kind == ConstraintKind.JSON:
        return {"schema_type": "json", "schema": constraint.schema or json.dumps({"type": "object"})}
    if constraint.kind == ConstraintKind.REGEX:
        return {"schema_type": "regex", "schema": constraint.schema}
    return None


@dataclass
class _Submission:
    request: GenerateRequest
    sink: asyncio.Queue
    loop: asyncio.AbstractEventLoop
    submitted: float = field(default_factory=time.monotonic)


@dataclass
class _Live:
    sink: asyncio.Queue
    loop: asyncio.AbstractEventLoop


@dataclass(frozen=True)
class DeepInferConfig:
    worker_id: str
    model_name: str
    #: ``LLM`` keyword arguments, e.g. expert/context parallel sizes.
    llm_kwargs: dict[str, Any] = field(default_factory=dict)
    model_path: str = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
    context_limit: int = 131_072


#: (llm_engine, clone(raw) -> RequestOutput, SamplingParams class)
EngineParts = tuple[Any, Callable[[Any], Any], type]


def _load_deepinfer(config: DeepInferConfig) -> EngineParts:
    from deepinfer._C.engine import RequestOutput  # noqa: PLC0415 - optional dependency
    from deepinfer.entrypoints.llm import LLM  # noqa: PLC0415
    from deepinfer.sampling_params import SamplingParams  # noqa: PLC0415

    llm = LLM(model=config.model_path, trust_remote_code=True, decode_sequence=False, **config.llm_kwargs)
    return llm.llm_engine, RequestOutput.clone, SamplingParams


class DeepInferEngine(Engine):
    def __init__(self, config: DeepInferConfig, load: Callable[[DeepInferConfig], EngineParts] = _load_deepinfer) -> None:
        self._config = config
        self._load_parts = load
        self._engine: Any = None
        self._clone: Callable[[Any], Any] | None = None
        self._sampling_cls: type | None = None
        self._submit: queue.SimpleQueue[_Submission | tuple[str, str]] = queue.SimpleQueue()
        self._live: dict[str, _Live] = {}
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._prefill_tokens = 0
        self._decode_tokens = 0
        self._drafted = 0
        self._accepted = 0
        self._hub = EventHub(self._counters)
        self._error: str | None = None

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        self._engine, self._clone, self._sampling_cls = await asyncio.to_thread(self._load_parts, self._config)
        self._running.set()
        self._thread = threading.Thread(target=self._step_loop, name="deepinfer-step", daemon=True)
        self._thread.start()

    async def stop(self) -> None:
        self._running.clear()
        if self._thread is not None:
            await asyncio.to_thread(self._thread.join, 10)
        self._hub.close()
        close = getattr(self._engine, "close", None)
        if callable(close):
            close()

    def _step_loop(self) -> None:
        engine, clone = self._engine, self._clone
        assert clone is not None
        while self._running.is_set():
            self._drain_submissions()
            try:
                unfinished = engine.has_unfinished_requests()
                raw_outputs = engine.step() if unfinished else engine.flush_inflight_reqs()
                for raw in raw_outputs:
                    self._deliver(clone(raw))
            except Exception as error:  # noqa: BLE001 - surfaced through health and requests
                log.exception("DeepInfer step failed")
                self._error = str(error)
                self._fail_all(str(error))
                return
            if not unfinished:
                time.sleep(0.001)

    def _drain_submissions(self) -> None:
        assert self._sampling_cls is not None
        while True:
            try:
                item = self._submit.get_nowait()
            except queue.Empty:
                return
            if isinstance(item, tuple):  # ("abort", request_id)
                _, request_id = item
                if self._live.pop(request_id, None) is not None:
                    self._engine.abort_request(request_id)
                continue
            request = item.request
            s = request.sampling
            params = self._sampling_cls(
                temperature=s.temperature,
                top_p=s.top_p,
                max_tokens=s.max_tokens or self._config.context_limit - len(request.prompt_token_ids),
                ignore_eos=s.ignore_eos,
                top_logprobs=s.top_logprobs if s.logprobs else 0,
                seq_seed=s.seed,
                eos_id=getattr(self._engine, "eos_id", None),
                so_args=so_args(request),
            )
            import numpy as np  # noqa: PLC0415 - DeepInfer depends on numpy

            self._live[request.request_id] = _Live(item.sink, item.loop)
            self._engine.add_request(
                user_id=request.tenant_id or "0",
                request_id=request.request_id,
                sampling_params=params,
                prompt_token_ids=np.asarray(request.prompt_token_ids, dtype=np.int32),
                multi_modal_data=None,
            )
            self._prefill_tokens += len(request.prompt_token_ids)

    def _deliver(self, output: Any) -> None:
        live = self._live.get(output.request_id)
        if live is None:
            return
        if getattr(output, "finished", False):
            self._live.pop(output.request_id, None)
        live.loop.call_soon_threadsafe(live.sink.put_nowait, output)

    def _fail_all(self, error: str) -> None:
        for request_id, live in list(self._live.items()):
            live.loop.call_soon_threadsafe(live.sink.put_nowait, RuntimeError(error))
            self._live.pop(request_id, None)

    # ------------------------------------------------------------------ info

    def info(self) -> EngineInfo:
        return EngineInfo(
            worker_id=self._config.worker_id,
            backend="deepinfer",
            model=self._config.model_name,
            role=Role.UNIFIED,
            block_size=64,
            context_limit=self._config.context_limit,
            total_blocks=0,
            supports_kv_events=False,
            supports_constraints=True,
            supports_logprobs=True,
            deterministic=True,
            labels={"status": "untested-on-hardware"},
        )

    def health(self) -> Health:
        if self._error:
            return Health(ready=False, status=f"engine failed: {self._error}")
        if not self._running.is_set():
            return Health(ready=False, status="starting")
        return Health(ready=True, status="ok")

    def _counters(self) -> LoadCounters:
        running = len(self._live)
        return LoadCounters(
            running=running,
            waiting=0,
            kv_usage=0.0,
            free_blocks=0,
            prefill_tokens=self._prefill_tokens,
            decode_tokens=self._decode_tokens,
            spec_draft_tokens=self._drafted,
            spec_accepted_tokens=self._accepted,
        )

    def subscribe(self, *, snapshot: bool, load_report_interval_ms: int) -> EventSubscription:
        return self._hub.subscribe(snapshot=snapshot, load_report_interval_ms=load_report_interval_ms)

    async def abort(self, request_id: str) -> bool:
        found = request_id in self._live
        self._submit.put(("abort", request_id))
        return found

    # ------------------------------------------------------------------ generate

    async def generate(self, request: GenerateRequest) -> AsyncIterator[GenerateEvent]:
        if not self._running.is_set():
            raise UnavailableError("DeepInfer engine is not running")
        if request.stage != Role.UNIFIED:
            raise InvalidRequestError("the DeepInfer adapter runs unified requests only")
        if not request.prompt_token_ids:
            raise InvalidRequestError("empty prompt")
        sink: asyncio.Queue = asyncio.Queue()
        submission = _Submission(request, sink, asyncio.get_running_loop())
        self._submit.put(submission)
        eos = getattr(self._engine, "eos_id", None)
        stops = set(request.sampling.stop_token_ids) | ({eos} if eos is not None else set())
        completion = 0
        ready = False
        try:
            while True:
                item = await sink.get()
                if isinstance(item, Exception):
                    yield Finished(reason=FinishReason.ERROR, completion_tokens=completion, error=str(item))
                    return
                if not ready:
                    ready = True
                    yield Ready(
                        prompt_tokens=int(getattr(item, "prompt_len", len(request.prompt_token_ids))),
                        cached_tokens=0,
                        queue_time_us=0,
                        prefill_time_us=int((time.monotonic() - submission.submitted) * 1e6),
                    )
                drafted = accepted = 0
                if getattr(item, "draft_tok", 0) > 0:
                    drafted = int(item.draft_tok) - 1
                    accepted = int(item.accept_tok) - 1
                    self._drafted += drafted
                    self._accepted += accepted
                finished = bool(getattr(item, "finished", False))
                reason = (getattr(item, "finish_reason", None) or "stop") if finished else None
                token = int(item.token_id)
                is_stop = finished and reason == "stop" and token in stops
                if not is_stop:
                    completion += 1
                    self._decode_tokens += 1
                    logprob = getattr(item, "token_logprobs", None)
                    yield TokenBatch(
                        token_ids=(token,),
                        logprobs=(TokenLogprob(logprob=float(logprob)),)
                        if request.sampling.logprobs and logprob is not None
                        else (),
                        draft_tokens=max(drafted, 0),
                        accepted_tokens=max(accepted, 0),
                    )
                elif drafted > 0:
                    # The stop token is dropped, but its verify step still counts.
                    yield TokenBatch(token_ids=(), draft_tokens=drafted, accepted_tokens=max(accepted, 0))
                if finished:
                    yield Finished(
                        reason=FinishReason.LENGTH if reason == "length" else FinishReason.STOP,
                        completion_tokens=completion,
                    )
                    return
        finally:
            if request.request_id in self._live:
                self._submit.put(("abort", request.request_id))
