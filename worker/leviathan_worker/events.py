"""Fan-out of KV cache events and load reports to subscribers.

This module is backend-neutral: a backend publishes the KV events of its
cache, and the hub does the rest. It keeps a mirror of the cache contents those
events imply, which gives every backend a snapshot for free and makes the
snapshot consistent with the live stream by construction (the snapshot and the
subscriber's registration happen in one synchronous step on the event loop).
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from .engine import (
    AllCleared,
    BlockRemoved,
    BlockStored,
    EventSubscription,
    KvEvent,
    KvEventBatch,
    LoadCounters,
    LoadReport,
    Medium,
    SubscriberOverflowError,
    WorkerEvent,
)

DEFAULT_QUEUE_SIZE = 1024
#: Snapshot events are split so that no batch carries more block hashes than this.
SNAPSHOT_CHUNK_BLOCKS = 4096
#: Time constant of the throughput EMA in load reports.
EMA_TAU_S = 1.0


class CacheMirror:
    """The set of cached blocks per tier implied by a stream of KV events."""

    def __init__(self) -> None:
        # (hash, medium) -> parent hash, in insertion order.
        self._blocks: dict[tuple[int, Medium], int | None] = {}

    def __len__(self) -> int:
        return len(self._blocks)

    def blocks(self, medium: Medium) -> set[int]:
        return {h for (h, m) in self._blocks if m is medium}

    def apply(self, events: Iterable[KvEvent]) -> None:
        for event in events:
            if isinstance(event, BlockStored):
                parent = event.parent_hash
                for h in event.block_hashes:
                    self._blocks[(h, event.medium)] = parent
                    parent = h
            elif isinstance(event, BlockRemoved):
                media = (Medium.GPU, Medium.CPU) if event.medium is Medium.UNSPECIFIED else (event.medium,)
                for h in event.block_hashes:
                    for m in media:
                        self._blocks.pop((h, m), None)
            elif isinstance(event, AllCleared):
                self._blocks.clear()

    def snapshot(self, chunk: int = SNAPSHOT_CHUNK_BLOCKS) -> list[BlockStored]:
        """The cached blocks as ``BlockStored`` chains, parents before children.

        Blocks form a forest through their parent hashes. A depth-first walk
        emits each root-to-leaf path as chains, starting a new chain whenever
        the path branches or changes tier, so every chain's ``parent_hash`` was
        stored by an earlier chain (or is not cached at all).
        """
        media: dict[int, list[Medium]] = {}
        parent_of: dict[int, int | None] = {}
        for (h, m), parent in self._blocks.items():
            media.setdefault(h, []).append(m)
            parent_of.setdefault(h, parent)
        children: dict[int, list[int]] = {}
        roots: list[int] = []
        for h, parent in parent_of.items():
            if parent is not None and parent in parent_of:
                children.setdefault(parent, []).append(h)
            else:
                roots.append(h)

        chains: list[_Chain] = []
        open_chain: dict[Medium, _Chain] = {}
        visited: set[int] = set()

        def visit(start: int) -> None:
            stack = [start]
            while stack:
                h = stack.pop()
                if h in visited:
                    continue
                visited.add(h)
                parent = parent_of[h]
                for m in media[h]:
                    current = open_chain.get(m)
                    if current is not None and parent is not None and current.hashes[-1] == parent:
                        current.hashes.append(h)
                    else:
                        current = _Chain(m, parent, [h])
                        chains.append(current)
                        open_chain[m] = current
                stack.extend(reversed(children.get(h, ())))

        for root in roots:
            visit(root)
        for h in parent_of:  # only reachable through a hash collision cycle
            visit(h)

        events: list[BlockStored] = []
        for c in chains:
            parent = c.parent
            for i in range(0, len(c.hashes), chunk):
                part = tuple(c.hashes[i : i + chunk])
                events.append(BlockStored(block_hashes=part, parent_hash=parent, medium=c.medium))
                parent = part[-1]
        return events


@dataclass
class _Chain:
    medium: Medium
    parent: int | None
    hashes: list[int] = field(default_factory=list)


def snapshot_batches(mirror: CacheMirror, chunk: int = SNAPSHOT_CHUNK_BLOCKS) -> list[tuple[KvEvent, ...]]:
    """``AllCleared`` plus the mirror's chains, packed into batches of at most ``chunk`` hashes."""
    batches: list[tuple[KvEvent, ...]] = []
    current: list[KvEvent] = [AllCleared()]
    size = 0
    for event in mirror.snapshot(chunk):
        if size + len(event.block_hashes) > chunk and size > 0:
            batches.append(tuple(current))
            current, size = [], 0
        current.append(event)
        size += len(event.block_hashes)
    batches.append(tuple(current))
    return batches


class EventHub:
    """Publishes KV event batches to subscribers, each with its own sequence numbers."""

    def __init__(
        self,
        load: Callable[[], LoadCounters],
        *,
        queue_size: int = DEFAULT_QUEUE_SIZE,
        snapshot_chunk: int = SNAPSHOT_CHUNK_BLOCKS,
    ) -> None:
        if queue_size <= 0:
            raise ValueError("queue_size must be positive")
        self._load = load
        self._queue_size = queue_size
        self._snapshot_chunk = snapshot_chunk
        self._mirror = CacheMirror()
        self._subscribers: set[_HubSubscription] = set()
        self._closed = False

    @property
    def mirror(self) -> CacheMirror:
        return self._mirror

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def publish(self, events: Sequence[KvEvent]) -> None:
        if not events:
            return
        batch = tuple(events)
        self._mirror.apply(batch)
        for subscriber in list(self._subscribers):
            subscriber.offer(batch)

    def subscribe(self, *, snapshot: bool, load_report_interval_ms: int) -> EventSubscription:
        if self._closed:
            raise RuntimeError("event hub is closed")
        if load_report_interval_ms <= 0:
            raise ValueError("load_report_interval_ms must be positive")
        initial = snapshot_batches(self._mirror, self._snapshot_chunk) if snapshot else []
        subscriber = _HubSubscription(
            self, initial=initial, interval_s=load_report_interval_ms / 1000, queue_size=self._queue_size
        )
        self._subscribers.add(subscriber)
        return subscriber

    def load(self) -> LoadCounters:
        return self._load()

    def close(self) -> None:
        """End every subscription; later subscribe() calls fail."""
        self._closed = True
        for subscriber in list(self._subscribers):
            subscriber.close()

    def discard(self, subscriber: _HubSubscription) -> None:
        self._subscribers.discard(subscriber)


class _HubSubscription(EventSubscription):
    def __init__(
        self, hub: EventHub, *, initial: list[tuple[KvEvent, ...]], interval_s: float, queue_size: int
    ) -> None:
        self._hub = hub
        self._initial = deque(initial)
        # None is a wake-up sentinel put by close().
        self._queue: asyncio.Queue[tuple[KvEvent, ...] | None] = asyncio.Queue(queue_size)
        self._interval = interval_s
        self._next_report = time.monotonic()  # the first report follows the snapshot
        self._seq = 0
        self._window = _LoadWindow()
        self._closed = False
        self._overflowed = False

    def offer(self, events: tuple[KvEvent, ...]) -> None:
        if self._closed:
            return
        try:
            self._queue.put_nowait(events)
        except asyncio.QueueFull:
            self._overflowed = True
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._hub.discard(self)
        self._initial.clear()
        while not self._queue.empty():  # drop undelivered batches
            self._queue.get_nowait()
        self._queue.put_nowait(None)  # wake a consumer blocked in get()

    def _batch(self, events: tuple[KvEvent, ...]) -> KvEventBatch:
        self._seq += 1
        return KvEventBatch(seq=self._seq, events=events)

    async def __anext__(self) -> WorkerEvent:
        while True:
            if self._overflowed:
                raise SubscriberOverflowError(
                    "event subscriber fell behind and was dropped; resubscribe with snapshot=true"
                )
            if self._closed:
                raise StopAsyncIteration
            if self._initial:
                return self._batch(self._initial.popleft())
            now = time.monotonic()
            if now >= self._next_report:
                self._next_report = now + self._interval
                return self._window.report(self._hub.load(), now)
            try:
                events = await asyncio.wait_for(self._queue.get(), self._next_report - now)
            except asyncio.TimeoutError:
                continue
            if events is not None:
                return self._batch(events)


class _LoadWindow:
    """Turns an engine's monotonic counters into the rates of one subscriber's reports."""

    def __init__(self) -> None:
        self._prev: LoadCounters | None = None
        self._prev_t = 0.0
        self._prefill_rate = 0.0
        self._decode_rate = 0.0
        self._accept_rate = 0.0

    def report(self, c: LoadCounters, now: float) -> LoadReport:
        prev = self._prev
        if prev is not None and now > self._prev_t:
            dt = now - self._prev_t
            # A time-based smoothing factor keeps the EMA independent of the report interval.
            alpha = 1.0 - math.exp(-dt / EMA_TAU_S)
            self._prefill_rate += alpha * ((c.prefill_tokens - prev.prefill_tokens) / dt - self._prefill_rate)
            self._decode_rate += alpha * ((c.decode_tokens - prev.decode_tokens) / dt - self._decode_rate)
            drafted = c.spec_draft_tokens - prev.spec_draft_tokens
            if drafted > 0:  # an idle window keeps the last known rate
                self._accept_rate = (c.spec_accepted_tokens - prev.spec_accepted_tokens) / drafted
        self._prev, self._prev_t = c, now
        return LoadReport(
            running=c.running,
            waiting=c.waiting,
            kv_usage=c.kv_usage,
            free_blocks=c.free_blocks,
            prefill_tokens_per_s=self._prefill_rate,
            decode_tokens_per_s=self._decode_rate,
            spec_accept_rate=self._accept_rate,
            timestamp_ms=int(time.time() * 1000),
        )
