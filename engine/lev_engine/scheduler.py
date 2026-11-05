"""Continuous batching with chunked prefill, priorities and preemption.

One step has a token budget (`max_num_batched_tokens`). Running sequences
are served first, in priority order: a decoding sequence needs one token, a
sequence still in its prompt gets as much of it as the budget allows (chunked
prefill), so a long prompt never stalls the decodes sharing its step. Waiting
sequences are admitted with what is left, after a prefix-cache lookup that
skips the blocks already computed.

When the pool runs out of blocks, the lowest-priority, most recently arrived
running sequence is preempted by recomputation: its blocks are released (the
full ones stay in the prefix cache, so re-admission is usually cheap) and it
goes back to the waiting queue.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import EngineConfig
from .kv_cache import BlockPool
from .sequence import Sequence, Status


@dataclass
class ScheduledSeq:
    seq: Sequence
    num_new: int
    #: Fixed when scheduled; `commit` moves `seq.num_computed` on.
    start: int = -1
    #: After this step every known token is computed, so it samples one.
    completes_prompt: bool = False

    def __post_init__(self) -> None:
        self.start = self.seq.num_computed
        self.completes_prompt = self.seq.num_computed + self.num_new == self.seq.num_tokens


@dataclass
class SchedulerOutput:
    scheduled: list[ScheduledSeq] = field(default_factory=list)
    preempted: list[Sequence] = field(default_factory=list)

    @property
    def num_tokens(self) -> int:
        return sum(s.num_new for s in self.scheduled)


class Scheduler:
    def __init__(self, config: EngineConfig, pool: BlockPool):
        self.config = config
        self.pool = pool
        self.waiting: list[Sequence] = []
        self.running: list[Sequence] = []

    # ------------------------------------------------------------------ queue

    def add(self, seq: Sequence) -> None:
        if seq.num_tokens + max(seq.params.max_tokens, 0) > self.config.max_model_len and seq.params.max_tokens:
            raise ValueError(f"request {seq.request_id} exceeds max_model_len {self.config.max_model_len}")
        if seq.num_tokens >= self.config.max_model_len:
            raise ValueError(f"prompt of {seq.num_tokens} tokens exceeds max_model_len {self.config.max_model_len}")
        self.waiting.append(seq)

    def finish(self, seq: Sequence) -> None:
        self.pool.free(seq.block_ids)
        seq.block_ids = []
        seq.status = Status.FINISHED
        if seq in self.running:
            self.running.remove(seq)
        elif seq in self.waiting:
            self.waiting.remove(seq)

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    # ------------------------------------------------------------------ step

    def _blocks_needed(self, seq: Sequence, upto_tokens: int) -> int:
        bs = self.pool.block_size
        return max(0, (upto_tokens + bs - 1) // bs - len(seq.block_ids))

    def _preempt(self, seq: Sequence, out: SchedulerOutput) -> None:
        self.pool.free(seq.block_ids)
        seq.block_ids = []
        seq.num_computed = 0
        seq.num_registered = 0
        seq.num_preemptions += 1
        seq.status = Status.WAITING
        self.running.remove(seq)
        self.waiting.append(seq)
        out.preempted.append(seq)

    def schedule(self) -> SchedulerOutput:
        out = SchedulerOutput()
        budget = self.config.max_num_batched_tokens
        bs = self.pool.block_size

        self.running.sort(key=lambda s: s.sort_key)
        for seq in list(self.running):
            if budget == 0:
                break
            if seq.status is not Status.RUNNING:
                continue  # preempted earlier in this loop
            n = min(seq.num_tokens - seq.num_computed, budget)
            if n <= 0:
                continue
            need = self._blocks_needed(seq, seq.num_computed + n)
            while need > self.pool.num_free:
                victim = max(
                    (s for s in self.running if s.status is Status.RUNNING and s not in (x.seq for x in out.scheduled)),
                    key=lambda s: s.sort_key,
                )
                self._preempt(victim, out)
                if victim is seq:
                    break
            if seq.status is not Status.RUNNING:
                continue
            seq.block_ids += self.pool.allocate(need)
            out.scheduled.append(ScheduledSeq(seq, n))
            budget -= n

        if out.preempted:
            return out  # no admissions while memory is short

        self.waiting.sort(key=lambda s: s.sort_key)
        for seq in list(self.waiting):
            if budget == 0 or len(self.running) >= self.config.max_num_seqs:
                break
            if seq.num_computed == 0 and not seq.block_ids:
                # Keep at least one token to compute: its logits are needed.
                hit = self.pool.match(seq.block_hashes(bs, seq.num_tokens - 1))
                seq.block_ids = hit
                seq.num_computed = seq.num_registered = len(hit) * bs
            n = min(seq.num_tokens - seq.num_computed, budget)
            need = self._blocks_needed(seq, seq.num_computed + n)
            if need > self.pool.num_free:
                self.pool.free(seq.block_ids)  # give the prefix hit back
                seq.block_ids, seq.num_computed, seq.num_registered = [], 0, 0
                break
            if seq.num_cached_prompt is None:
                seq.num_cached_prompt = seq.num_computed
            seq.block_ids += self.pool.allocate(need)
            seq.status = Status.RUNNING
            self.waiting.remove(seq)
            self.running.append(seq)
            out.scheduled.append(ScheduledSeq(seq, n))
            budget -= n
        return out

    def commit(self, scheduled: ScheduledSeq, register: bool = True) -> None:
        """After the forward pass: the new tokens' KV is in the cache.
        `register=False` defers prefix registration (overlapped scheduling:
        the last input token may still be a placeholder)."""
        scheduled.seq.num_computed += scheduled.num_new
        if register:
            self.register(scheduled.seq)

    def register(self, seq: Sequence) -> None:
        """Put the sequence's full, computed blocks into the prefix cache."""
        full = seq.num_computed // self.pool.block_size
        if full > seq.num_registered:
            hashes = seq.block_hashes(self.pool.block_size, full * self.pool.block_size)
            for i in range(seq.num_registered, full):
                parent = hashes[i - 1] if i else None
                self.pool.register(seq.block_ids[i], hashes[i], parent)
            seq.num_registered = full
