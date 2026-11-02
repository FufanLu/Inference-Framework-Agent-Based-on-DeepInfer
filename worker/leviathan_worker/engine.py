"""The backend-neutral engine interface that every worker backend implements.

The gRPC layer (:mod:`leviathan_worker.server`) translates protobuf messages to
and from the plain dataclasses defined here, so backends (sim today, vLLM and
DeepInfer later) never touch generated code. Enum values equal their protobuf
counterparts in ``proto/leviathan/worker/v1/worker.proto``.
"""

from __future__ import annotations

import abc
import enum
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Union


class Role(enum.IntEnum):
    """The part a worker, or one stage of a request, plays in PD disaggregation."""

    UNSPECIFIED = 0
    UNIFIED = 1
    PREFILL = 2
    DECODE = 3


class FinishReason(enum.IntEnum):
    UNSPECIFIED = 0
    STOP = 1
    LENGTH = 2
    ABORT = 3
    ERROR = 4
    CONTENT_FILTER = 5


class Medium(enum.IntEnum):
    """KV cache tier."""

    UNSPECIFIED = 0
    GPU = 1
    CPU = 2


class ConstraintKind(enum.IntEnum):
    """Mirrors deepseek-recipe ``OutputConstraint::kind()``."""

    UNSPECIFIED = 0
    JSON = 1
    REGEX = 2
    TOOL_CALL = 3
    TOOL_CALL_REQUIRED = 4


# --------------------------------------------------------------------------- requests


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    #: 0 means "limited only by the context window".
    max_tokens: int = 0
    seed: int = 0
    logprobs: bool = False
    top_logprobs: int = 0
    stop_token_ids: tuple[int, ...] = ()
    ignore_eos: bool = False


@dataclass(frozen=True)
class OutputConstraint:
    kind: ConstraintKind
    #: JSON schema, regex, or JSON array of tools, depending on ``kind``.
    schema: str = ""


@dataclass(frozen=True)
class KvTransfer:
    """KV cache hand-off between a prefill and a decode worker."""

    source_worker_id: str
    #: Backend-specific opaque payload.
    handle: str
    num_tokens: int


@dataclass(frozen=True)
class GenerateRequest:
    request_id: str
    prompt_token_ids: tuple[int, ...]
    sampling: SamplingParams = SamplingParams()
    constraint: OutputConstraint | None = None
    stage: Role = Role.UNIFIED
    kv_transfer: KvTransfer | None = None
    tenant_id: str = ""
    priority: int = 0
    #: Facts the gateway derived during conversion (``thinking``, ``last_role``,
    #: ``effort``). Real engines ignore them; the sim shapes its output with them.
    hints: Mapping[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------- generate events


@dataclass(frozen=True)
class Ready:
    """Prefill is done; sent once, before any output token."""

    prompt_tokens: int
    #: Prompt tokens served from the prefix cache or received via KV transfer.
    cached_tokens: int
    queue_time_us: int
    prefill_time_us: int
    #: For the PREFILL stage: the hand-off for the decode worker.
    kv_transfer: KvTransfer | None = None


@dataclass(frozen=True)
class TokenLogprob:
    logprob: float
    #: The most likely tokens at this position, most likely first.
    top_token_ids: tuple[int, ...] = ()
    top_logprobs: tuple[float, ...] = ()


@dataclass(frozen=True)
class TokenBatch:
    """Output tokens of one engine step (EOS and stop tokens are never included)."""

    token_ids: tuple[int, ...]
    #: Empty unless logprobs were requested; otherwise one entry per token.
    logprobs: tuple[TokenLogprob, ...] = ()
    draft_tokens: int = 0
    accepted_tokens: int = 0

    def merged(self, later: TokenBatch) -> TokenBatch:
        """This batch followed by ``later``, as one batch."""
        return TokenBatch(
            token_ids=self.token_ids + later.token_ids,
            logprobs=self.logprobs + later.logprobs,
            draft_tokens=self.draft_tokens + later.draft_tokens,
            accepted_tokens=self.accepted_tokens + later.accepted_tokens,
        )


@dataclass(frozen=True)
class Finished:
    reason: FinishReason
    completion_tokens: int
    error: str = ""
    decode_time_us: int = 0


GenerateEvent = Union[Ready, TokenBatch, Finished]


# --------------------------------------------------------------------------- worker events


@dataclass(frozen=True)
class BlockStored:
    """A chain of consecutive blocks; the first one follows ``parent_hash``."""

    block_hashes: tuple[int, ...]
    parent_hash: int | None
    medium: Medium


@dataclass(frozen=True)
class BlockRemoved:
    block_hashes: tuple[int, ...]
    medium: Medium


@dataclass(frozen=True)
class AllCleared:
    pass


KvEvent = Union[BlockStored, BlockRemoved, AllCleared]


@dataclass(frozen=True)
class KvEventBatch:
    #: Strictly increasing (by one) per subscription.
    seq: int
    events: tuple[KvEvent, ...]


@dataclass(frozen=True)
class LoadReport:
    running: int
    waiting: int
    kv_usage: float
    free_blocks: int
    prefill_tokens_per_s: float
    decode_tokens_per_s: float
    spec_accept_rate: float
    timestamp_ms: int


WorkerEvent = Union[KvEventBatch, LoadReport]


@dataclass(frozen=True)
class LoadCounters:
    """A backend's load at one instant: gauges plus monotonic counters.

    Subscriptions turn the counters into per-window rates, so a backend only
    has to count (as vLLM's Prometheus counters do).
    """

    running: int
    waiting: int
    kv_usage: float
    free_blocks: int
    #: Prompt tokens computed so far (cache hits excluded).
    prefill_tokens: int
    #: Output tokens produced so far.
    decode_tokens: int
    spec_draft_tokens: int
    spec_accepted_tokens: int


# --------------------------------------------------------------------------- info


@dataclass(frozen=True)
class EngineInfo:
    worker_id: str
    backend: str
    model: str
    role: Role
    block_size: int
    context_limit: int
    total_blocks: int
    supports_kv_events: bool
    supports_constraints: bool
    supports_logprobs: bool
    deterministic: bool
    labels: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Health:
    ready: bool
    status: str


# --------------------------------------------------------------------------- errors


class EngineError(Exception):
    """A request the engine refuses before it starts; the server maps it to a gRPC status."""


class InvalidRequestError(EngineError):
    """The request is malformed or can never run on this worker (INVALID_ARGUMENT)."""


class DuplicateRequestError(EngineError):
    """A request with the same ID is already in flight (ALREADY_EXISTS)."""


class OverloadedError(EngineError):
    """The engine's queue is full; retry later or elsewhere (RESOURCE_EXHAUSTED)."""


class UnavailableError(EngineError):
    """The engine is not serving, e.g. before start or after stop (UNAVAILABLE)."""


class SubscriberOverflowError(Exception):
    """An event subscriber fell too far behind and was closed; resubscribe with a snapshot."""


# --------------------------------------------------------------------------- interface


class EventSubscription(abc.ABC):
    """A live stream of :data:`WorkerEvent`; iterate it with ``async for``.

    Iteration ends when the engine stops and raises
    :class:`SubscriberOverflowError` when the subscriber falls too far behind.
    Call :meth:`close` when done; it is idempotent.
    """

    def __aiter__(self) -> EventSubscription:
        return self

    @abc.abstractmethod
    async def __anext__(self) -> WorkerEvent: ...

    @abc.abstractmethod
    def close(self) -> None: ...


class Engine(abc.ABC):
    """One inference engine replica behind a worker."""

    async def start(self) -> None:
        """Start background work. Called once, before serving."""

    async def stop(self) -> None:
        """Abort in-flight requests, end subscriptions and release resources. Idempotent."""

    @abc.abstractmethod
    def info(self) -> EngineInfo: ...

    @abc.abstractmethod
    def health(self) -> Health: ...

    @abc.abstractmethod
    def generate(self, request: GenerateRequest) -> AsyncIterator[GenerateEvent]:
        """Run one request and stream its events.

        The stream carries at most one :class:`Ready` (when prefill is done),
        then :class:`TokenBatch` events, and ends with exactly one
        :class:`Finished`. A failure after the request was accepted is
        reported as ``Finished(ERROR)``, not raised.

        Raises :class:`EngineError` subclasses before the first event when the
        request is refused. Closing the iterator early (``aclose()``, or
        cancelling the task that consumes it) aborts the request.
        """

    @abc.abstractmethod
    async def abort(self, request_id: str) -> bool:
        """Abort an in-flight request; its stream ends with ``Finished(ABORT)``.

        Returns whether the request was found.
        """

    @abc.abstractmethod
    def subscribe(self, *, snapshot: bool, load_report_interval_ms: int) -> EventSubscription:
        """Open a stream of KV event batches and periodic load reports.

        With ``snapshot``, the stream starts with an :class:`AllCleared` followed
        by :class:`BlockStored` chains for every cached block, consistent with
        the live events that follow.
        """
