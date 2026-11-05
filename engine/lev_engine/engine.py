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
from .models import model_class
from .ops import get_ops
from .sampler import sample_tensors
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
class _Inflight:
    """A launched step whose samples have not been read back yet."""

    seqs: list[Sequence]
    tokens: torch.Tensor  # [S] on the device
    logprobs: torch.Tensor | None
    positions: list[int] = field(default_factory=list)  # index of each placeholder in seq.output

    def place_holders(self) -> None:
        """Append placeholder -(i+1) for sample i: the token is unknown until
        `LLMEngine._resolve`, but the scheduler can already count it."""
        for i, seq in enumerate(self.seqs):
            self.positions.append(len(seq.output))
            seq.output.append(-(i + 1))
            if seq.params.logprobs:
                seq.logprobs.append(0.0)


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
        self.model = model_class(self.spec)(self.spec, weights, get_ops(config.kernel_backend, config.batch_invariant), config.max_model_len, self.device)
        num_blocks = config.num_blocks or self._blocks_for_memory(dtype)
        graphs = config.cuda_graph and self.device.type == "cuda" and config.kernel_backend == "triton"
        if config.cuda_graph and not graphs:
            raise ValueError("cuda_graph needs device=cuda and kernel_backend=triton")
        # With graphs, the last block is scratch space for padding rows.
        self.pool = BlockPool(num_blocks - int(graphs), config.block_size, config.enable_prefix_caching)
        shape = self.model.kv_cache_shape(num_blocks, config.block_size)
        self.kv_caches = [torch.zeros(shape, dtype=dtype, device=self.device) for _ in range(self.spec.num_layers)]
        self.graphs = None
        if graphs:
            from .cuda_graph import DecodeGraphs

            max_blocks = (config.max_model_len + config.block_size - 1) // config.block_size
            self.graphs = DecodeGraphs(
                self.model, self.kv_caches, config.block_size, max_blocks, num_blocks - 1, config.cuda_graph_max_batch
            )
        self.scheduler = Scheduler(config, self.pool)
        self.requests: dict[str, Sequence] = {}
        self.stats = EngineStats()
        self._inflight: _Inflight | None = None

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
            while seq.output and seq.output[-1] < 0:  # unresolved placeholders
                seq.output.pop()
                if seq.params.logprobs:
                    seq.logprobs.pop()
            self.scheduler.finish(seq)

    def update_weights(self, weights: dict[str, torch.Tensor]) -> int:
        """Load new weights (RL weight sync). Every cached block was computed
        with the old weights, so the prefix cache is dropped; requests still
        running would continue on stale KV, so none may be in flight."""
        if self.scheduler.running or self.scheduler.waiting:
            raise RuntimeError("update_weights needs an idle engine (finish or abort requests first)")
        if not hasattr(self.model, "update_weights"):
            raise NotImplementedError(f"{type(self.model).__name__} does not support weight updates")
        applied = self.model.update_weights(weights)
        self.pool.reset_prefix_cache()
        return applied

    @property
    def has_unfinished(self) -> bool:
        return self.scheduler.has_work or self._inflight is not None

    def drain_events(self) -> list[BlockStored | BlockRemoved]:
        return self.pool.drain_events()

    # ------------------------------------------------------------------ step

    def _stop_at(self, seq: Sequence, token: int, produced: int) -> FinishReason | None:
        """Stop after the `produced`-th output token (`token`)?"""
        p = seq.params
        if not p.ignore_eos and token == self.spec.eos_token_id:
            return FinishReason.STOP
        if token in p.stop_token_ids:
            return FinishReason.STOP
        if p.max_tokens and produced >= p.max_tokens:
            return FinishReason.LENGTH
        if len(seq.prompt) + produced >= self.config.max_model_len:
            return FinishReason.LENGTH
        return None

    def step(self) -> list[StepOutput]:
        """Run one step. With `overlap_schedule`, returns the outputs of the
        previous step (its tokens are read back after this one is launched)."""
        t0 = time.perf_counter()
        plan = self.scheduler.schedule()
        stats = StepStats(num_preempted=len(plan.preempted))
        self.stats.preemptions += len(plan.preempted)
        if not plan.scheduled:
            self.stats.last = stats
            if self._inflight is not None:
                outputs, self._inflight = self._resolve(self._inflight), None
                return outputs
            if self.scheduler.waiting and not self.scheduler.running:
                seq = self.scheduler.waiting[0]
                raise RuntimeError(
                    f"request {seq.request_id} ({seq.num_tokens} tokens) does not fit in "
                    f"{self.pool.num_blocks} blocks of {self.config.block_size}"
                )
            return []
        batch = ForwardBatch.build(
            [s.seq.tokens[s.start : s.start + s.num_new] for s in plan.scheduled],
            [s.start for s in plan.scheduled],
            [s.seq.block_ids for s in plan.scheduled],
            self.config.block_size,
            [s.completes_prompt for s in plan.scheduled],
            self.device,
        )
        if self._inflight is not None:
            # Inputs that are the previous step's samples: placeholder -(i+1)
            # is row i of its token tensor, still on the device.
            ids = batch.input_ids
            batch.input_ids = torch.where(ids < 0, self._inflight.tokens[(-ids - 1).clamp(min=0)], ids)
        t1 = time.perf_counter()
        if self.graphs is not None and self.graphs.can_run(batch):
            logits = self.graphs.run(batch)
        else:
            logits = self.model.forward(batch, self.kv_caches)
        samplers = [s.seq for s in plan.scheduled if s.completes_prompt]
        launched = None
        if samplers:
            tokens, logprobs = sample_tensors(logits, [q.params for q in samplers], [len(q.output) for q in samplers])
            launched = _Inflight(samplers, tokens, logprobs)
        stats.num_seqs = len(plan.scheduled)
        stats.num_tokens = plan.num_tokens
        self.stats.steps += 1
        self.stats.last = stats

        if not self.config.overlap_schedule:
            for s in plan.scheduled:
                self.scheduler.commit(s)
            if launched is not None:
                launched.place_holders()
            outputs = self._resolve(launched) if launched is not None else []
            stats.schedule_s, stats.forward_s = t1 - t0, time.perf_counter() - t1
            return outputs

        # Overlapped: the KV of this step is "computed" as far as scheduling
        # goes; its samples are placeholders until the next call reads them.
        for s in plan.scheduled:
            self.scheduler.commit(s, register=False)
        if launched is not None:
            launched.place_holders()
        outputs = self._resolve(self._inflight) if self._inflight is not None else []
        self._inflight = launched
        stats.schedule_s, stats.forward_s = t1 - t0, time.perf_counter() - t1
        return outputs

    def _resolve(self, inflight: "_Inflight") -> list[StepOutput]:
        """Read a step's samples back (this is the synchronisation point),
        put them in place of the placeholders, apply stop conditions."""
        tokens = inflight.tokens.tolist()
        logprobs = inflight.logprobs.tolist() if inflight.logprobs is not None else None
        outputs: list[StepOutput] = []
        for i, seq in enumerate(inflight.seqs):
            if seq.status is Status.FINISHED:
                continue  # stopped one step earlier (overlap) or aborted
            pos = inflight.positions[i]
            token = tokens[i]
            seq.output[pos] = token
            lp = None
            if logprobs is not None and seq.params.logprobs:
                seq.logprobs[pos] = logprobs[i]
                lp = [logprobs[i]]
            self.stats.generated_tokens += 1
            if pos == 0:
                self.stats.prompt_tokens += len(seq.prompt)
                self.stats.cached_prompt_tokens += seq.num_cached_prompt or 0
            # Stop conditions see only the resolved tokens.
            reason = self._stop_at(seq, token, pos + 1)
            if reason is not None:
                del seq.output[pos + 1 :]  # an overlapped extra step's placeholder
                del seq.logprobs[pos + 1 :]
                seq.num_computed = min(seq.num_computed, seq.num_tokens)
                seq.finish_reason = reason
                self.scheduler.register(seq)
                self.scheduler.finish(seq)
            else:
                self.scheduler.register(seq)
            outputs.append(
                StepOutput(seq.request_id, [token], lp, reason is not None, reason,
                           seq.num_cached_prompt if pos == 0 else None)  # fmt: skip
            )
        return outputs

    # ------------------------------------------------------------------ helpers

    @torch.inference_mode()
    def score(self, sequences: list[list[int]], prompt_lens: list[int]) -> list[list[float]]:
        """Log-probabilities (temperature 1) of every token after the prompt,
        recomputed with the engine's kernels: what a trainer compares rollout
        logprobs against. Runs one sequence at a time through its own blocks."""
        out: list[list[float]] = []
        bs = self.config.block_size
        for tokens, plen in zip(sequences, prompt_lens, strict=True):
            if not 0 < plen < len(tokens):
                raise ValueError("each sequence needs a prompt and at least one scored token")
            blocks = self.pool.allocate((len(tokens) + bs - 1) // bs)
            try:
                batch = ForwardBatch.build([tokens], [0], [blocks], bs, [True], self.device)
                batch.logits_indices = torch.arange(plen - 1, len(tokens) - 1, device=self.device)
                logits = self.model.forward(batch, self.kv_caches).to(torch.float64)
                targets = torch.tensor(tokens[plen:], device=self.device)
                out.append(logits.log_softmax(-1).gather(1, targets[:, None])[:, 0].tolist())
            finally:
                self.pool.free(blocks)
        return out

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
