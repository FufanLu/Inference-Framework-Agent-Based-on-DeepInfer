"""The engine loop: schedule -> build batch -> forward -> sample -> commit."""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .batch import ForwardBatch
from .config import EngineConfig, ModelSpec
from .kv_cache import BlockPool, BlockRemoved, BlockStored
from .models.deepseek_v2 import DeepseekV2
from .ops import get_ops
from .sampler import sample
from .scheduler import Scheduler
from .sequence import FinishReason, SamplingParams, Sequence, Status
from .weights import load_state_dict

DTYPES = {"float32": torch.float32, "float64": torch.float64, "bfloat16": torch.bfloat16, "float16": torch.float16}


@dataclass
class StepOutput:
    request_id: str
    new_tokens: list[int]
    logprobs: list[float] | None
    finished: bool
    finish_reason: FinishReason | None
    #: Set on the first output of a request.
    num_cached_prompt: int | None = None


@dataclass
class StepStats:
    num_seqs: int = 0
    num_tokens: int = 0
    num_preempted: int = 0
    schedule_s: float = 0.0
    forward_s: float = 0.0


@dataclass
class EngineStats:
    steps: int = 0
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    generated_tokens: int = 0
    preemptions: int = 0
    last: StepStats = field(default_factory=StepStats)


class LLMEngine:
    def __init__(self, model_path: str | Path, config: EngineConfig | None = None):
        self.config = config = config or EngineConfig()
        self.spec = ModelSpec.from_pretrained(model_path)
        self.device = torch.device(config.device)
        dtype = DTYPES[config.dtype]
        weights = load_state_dict(model_path, dtype, self.device)
        self.model = DeepseekV2(self.spec, weights, get_ops(config.kernel_backend, config.batch_invariant), config.max_model_len, self.device)
        num_blocks = config.num_blocks or self._blocks_for_memory(dtype)
        self.pool = BlockPool(num_blocks, config.block_size, config.enable_prefix_caching)
        shape = self.model.kv_cache_shape(num_blocks, config.block_size)
        self.kv_caches = [torch.zeros(shape, dtype=dtype, device=self.device) for _ in range(self.spec.num_layers)]
        self.scheduler = Scheduler(config, self.pool)
        self.requests: dict[str, Sequence] = {}
        self.stats = EngineStats()

    def _blocks_for_memory(self, dtype: torch.dtype) -> int:
        if not self.config.kv_cache_memory_bytes:
            raise ValueError("set EngineConfig.num_blocks or kv_cache_memory_bytes")
        per_block = self.spec.num_layers * self.config.block_size * self.spec.kv_cache_dim * dtype.itemsize
        return self.config.kv_cache_memory_bytes // per_block

    # ------------------------------------------------------------------ requests

    def add_request(
        self, request_id: str, prompt: list[int], params: SamplingParams | None = None, priority: int = 0
    ) -> None:
        if request_id in self.requests:
            raise ValueError(f"duplicate request id {request_id}")
        if not prompt:
            raise ValueError("empty prompt")
        seq = Sequence(request_id, list(prompt), params or SamplingParams(), priority)
        self.scheduler.add(seq)
        self.requests[request_id] = seq

    def abort(self, request_id: str) -> None:
        seq = self.requests.pop(request_id, None)
        if seq is not None and seq.status is not Status.FINISHED:
            seq.finish_reason = FinishReason.ABORT
            self.scheduler.finish(seq)

    @property
    def has_unfinished(self) -> bool:
        return self.scheduler.has_work

    def drain_events(self) -> list[BlockStored | BlockRemoved]:
        return self.pool.drain_events()

    # ------------------------------------------------------------------ step

    def _stop(self, seq: Sequence, token: int) -> FinishReason | None:
        p = seq.params
        if not p.ignore_eos and token == self.spec.eos_token_id:
            return FinishReason.STOP
        if token in p.stop_token_ids:
            return FinishReason.STOP
        if p.max_tokens and len(seq.output) >= p.max_tokens:
            return FinishReason.LENGTH
        if seq.num_tokens >= self.config.max_model_len:
            return FinishReason.LENGTH
        return None

    def step(self) -> list[StepOutput]:
        t0 = time.perf_counter()
        plan = self.scheduler.schedule()
        stats = StepStats(num_preempted=len(plan.preempted))
        self.stats.preemptions += len(plan.preempted)
        if not plan.scheduled:
            self.stats.last = stats
            if self.scheduler.waiting and not self.scheduler.running:
                seq = self.scheduler.waiting[0]
                raise RuntimeError(
                    f"request {seq.request_id} ({seq.num_tokens} tokens) does not fit in "
                    f"{self.pool.num_blocks} blocks of {self.config.block_size}"
                )
            return []
        bs = self.config.block_size
        batch = ForwardBatch.build(
            [s.seq.tokens[s.start : s.start + s.num_new] for s in plan.scheduled],
            [s.start for s in plan.scheduled],
            [s.seq.block_ids for s in plan.scheduled],
            bs,
            [s.completes_prompt for s in plan.scheduled],
            self.device,
        )
        t1 = time.perf_counter()
        logits = self.model.forward(batch, self.kv_caches)
        samplers = [s.seq for s in plan.scheduled if s.completes_prompt]
        result = sample(logits, [q.params for q in samplers], [len(q.output) for q in samplers]) if samplers else None
        t2 = time.perf_counter()

        outputs: list[StepOutput] = []
        sampled = iter(range(len(samplers)))
        for s in plan.scheduled:
            seq = s.seq
            self.scheduler.commit(s)
            if not s.completes_prompt:
                continue
            i = next(sampled)
            token = result.tokens[i]
            seq.output.append(token)
            lp = None
            if result.logprobs is not None:
                lp = [result.logprobs[i]]
                seq.logprobs.append(result.logprobs[i])
            self.stats.generated_tokens += 1
            if len(seq.output) == 1:
                self.stats.prompt_tokens += len(seq.prompt)
                self.stats.cached_prompt_tokens += seq.num_cached_prompt or 0
            reason = self._stop(seq, token)
            if reason is not None:
                seq.finish_reason = reason
                self.scheduler.finish(seq)
            outputs.append(
                StepOutput(
                    seq.request_id,
                    [token],
                    lp,
                    reason is not None,
                    reason,
                    seq.num_cached_prompt if len(seq.output) == 1 else None,
                )
            )
        stats.num_seqs = len(plan.scheduled)
        stats.num_tokens = plan.num_tokens
        stats.schedule_s, stats.forward_s = t1 - t0, t2 - t1
        self.stats.steps += 1
        self.stats.last = stats
        return outputs

    # ------------------------------------------------------------------ helpers

    def generate(
        self, prompts: Iterable[list[int]], params: SamplingParams | list[SamplingParams] | None = None
    ) -> list[list[int]]:
        """Run prompts to completion; returns output tokens in prompt order."""
        prompts = list(prompts)
        plist = params if isinstance(params, list) else [params or SamplingParams()] * len(prompts)
        ids = [f"gen-{id(self)}-{i}-{time.monotonic_ns()}" for i in range(len(prompts))]
        for rid, prompt, p in zip(ids, prompts, plist, strict=True):
            self.add_request(rid, prompt, p)
        while self.has_unfinished:
            self.step()
        outs = [self.requests[rid].output for rid in ids]
        for rid in ids:
            self.requests.pop(rid, None)
        return outs
