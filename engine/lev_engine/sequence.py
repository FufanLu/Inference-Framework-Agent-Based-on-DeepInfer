"""Per-request state."""

from __future__ import annotations

import enum
import itertools
from dataclasses import dataclass, field

from .hashing import ROOT_SEED, block_hash

_arrivals = itertools.count()


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    #: 0: until EOS or the context limit.
    max_tokens: int = 16
    seed: int = 0
    ignore_eos: bool = False
    stop_token_ids: tuple[int, ...] = ()
    logprobs: bool = False


class Status(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


class FinishReason(enum.Enum):
    STOP = "stop"
    LENGTH = "length"
    ABORT = "abort"


@dataclass(eq=False)  # identity: two requests are never "equal"
class Sequence:
    request_id: str
    prompt: list[int]
    params: SamplingParams
    #: Lower runs first (the gateway sends interactive 0, batch jobs 1).
    priority: int = 0
    arrival: int = field(default_factory=lambda: next(_arrivals))
    output: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    status: Status = Status.WAITING
    finish_reason: FinishReason | None = None
    #: Tokens whose KV is in `block_ids`.
    num_computed: int = 0
    #: Prompt tokens served from the prefix cache at first admission.
    num_cached_prompt: int | None = None
    block_ids: list[int] = field(default_factory=list)
    #: Leading blocks already registered in the prefix cache.
    num_registered: int = 0
    #: Chain hashes of the full blocks of `tokens`, computed lazily.
    _hashes: list[int] = field(default_factory=list)
    num_preemptions: int = 0

    @property
    def tokens(self) -> list[int]:
        return self.prompt + self.output

    @property
    def num_tokens(self) -> int:
        return len(self.prompt) + len(self.output)

    @property
    def sort_key(self) -> tuple[int, int]:
        return (self.priority, self.arrival)

    def block_hashes(self, block_size: int, upto_tokens: int) -> list[int]:
        """Chain hashes of the full blocks among the first `upto_tokens` tokens."""
        want = upto_tokens // block_size
        if len(self._hashes) < want:
            tokens = self.tokens
            parent = self._hashes[-1] if self._hashes else ROOT_SEED
            for i in range(len(self._hashes), want):
                parent = block_hash(tokens[i * block_size : (i + 1) * block_size], parent)
                self._hashes.append(parent)
        return self._hashes[:want]
