"""vLLM backend: DeepSeek-V4.1-Flash on Hopper GPUs (Explorer H200).

The worker runs vLLM's ``AsyncLLM`` in-process and speaks token IDs on both
sides, exactly like the sim backend:

- prompts arrive as token IDs rendered by deepseek-recipe in the gateway and
  go to vLLM as ``TokensPrompt`` (no chat template, no detokenization);
- output token IDs stream back as ``TokenBatch`` deltas, with DSpark
  speculative decoding statistics taken from the per-request metrics;
- vLLM's KV cache events (ZMQ PUB, msgpack ``KVEventBatch``) are translated
  into Leviathan's block hash convention, because vLLM's own block hashes are
  not reproducible by the router. Each vLLM ``BlockStored`` carries the token
  IDs of its blocks, so the translation re-hashes them as a continuation of the
  parent block's chain;
- for prefill/decode disaggregation the NIXL ``kv_transfer_params`` returned by
  the prefill stage are the opaque hand-off passed to the decode stage.

vLLM is imported lazily so that this module (and its translator tests) load on
machines without vLLM.

Engine arguments are given exactly as to ``vllm serve`` (``--vllm-args``), so
the H200 recipe's flags apply unchanged, for example::

    --vllm-args "deepseek-ai/DeepSeek-V4.1-Flash-1017 --tokenizer-mode deepseek_v41
                 --tensor-parallel-size 4 --kv-cache-dtype fp8
                 --engram-config '{\\"cpu_offload\\":true}' --reasoning-parser deepseek_v41"
"""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
import time
from collections.abc import AsyncIterator, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..engine import (
    AllCleared,
    BlockRemoved,
    BlockStored,
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
    KvEvent,
    KvTransfer,
    LoadCounters,
    Medium,
    Ready,
    Role,
    TokenBatch,
    TokenLogprob,
    UnavailableError,
)
from ..events import EventHub
from ..hashing import ROOT_SEED, block_hashes

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- KV event translation


def _medium(value: str | None) -> Medium:
    return Medium.CPU if (value or "GPU").upper() in ("CPU", "STORAGE") else Medium.GPU


@dataclass
class KvEventTranslator:
    """Translate vLLM KV cache events into Leviathan block hashes.

    vLLM identifies blocks by its own (salted) hashes; ``BlockStored`` also
    carries the blocks' token IDs and the vLLM hash of the preceding block. The
    translator keeps ``vllm hash -> leviathan hash`` for every live block so
    that a chain can be continued from its parent and removals can be mapped.

    Hybrid-attention models (V4.1 has sliding-window and compressed layers)
    publish one event stream per KV cache group. Prefix reuse is decided by
    the full-context group, so events of sliding-window groups are ignored:
    ``group`` pins a group index, otherwise the first group whose kind is not
    sliding-window is used.
    """

    group: int | None = None
    _map: dict[Any, int] = field(default_factory=dict)
    #: Blocks whose parent was unknown, i.e. events missed: a resync is due.
    orphans: int = 0

    def _accept_group(self, event: Any) -> bool:
        group = getattr(event, "group_idx", None)
        if group is None:
            return True
        if self.group is None:
            kind = (getattr(event, "kv_cache_spec_kind", None) or "").lower()
            window = getattr(event, "kv_cache_spec_sliding_window", None)
            if "sliding" in kind or window:
                return False
            self.group = group
            log.info("following KV events of cache group %d (%s)", group, kind or "unknown kind")
        return group == self.group

    def translate(self, events: Iterable[Any]) -> list[KvEvent]:
        out: list[KvEvent] = []
        for event in events:
            name = type(event).__name__
            if name == "AllBlocksCleared":
                self._map.clear()
                out.append(AllCleared())
            elif name == "BlockStored":
                if not self._accept_group(event):
                    continue
                stored = self._stored(event)
                if stored is not None:
                    out.append(stored)
            elif name == "BlockRemoved":
                if not self._accept_group(event):
                    continue
                removed = [self._map.pop(h) for h in event.block_hashes if h in self._map]
                if removed:
                    out.append(BlockRemoved(block_hashes=tuple(removed), medium=_medium(event.medium)))
        return out

    def _stored(self, event: Any) -> BlockStored | None:
        size = int(event.block_size)
        tokens = list(event.token_ids)
        vllm_hashes = list(event.block_hashes)
        if size <= 0 or len(tokens) < size * len(vllm_hashes):
            return None
        parent_vllm = event.parent_block_hash
        if parent_vllm is None:
            parent = ROOT_SEED
            parent_out: int | None = None
        elif parent_vllm in self._map:
            parent = self._map[parent_vllm]
            parent_out = parent
        else:
            self.orphans += 1
            return None
        ours = block_hashes(tokens[: size * len(vllm_hashes)], size, parent=parent)
        for vllm_hash, our_hash in zip(vllm_hashes, ours, strict=True):
            self._map[vllm_hash] = our_hash
        return BlockStored(block_hashes=tuple(ours), parent_hash=parent_out, medium=_medium(event.medium))


# --------------------------------------------------------------------------- request mapping


def stop_set(request: GenerateRequest, eos_token_id: int | None) -> set[int]:
    stops = set(request.sampling.stop_token_ids)
    if eos_token_id is not None and not request.sampling.ignore_eos:
        stops.add(eos_token_id)
    return stops


def structured_outputs_kwargs(request: GenerateRequest) -> dict[str, Any] | None:
    """vLLM ``StructuredOutputsParams`` keyword arguments for a constraint.

    Tool-call blocks are not constrained: the DSML tags must be produced as
    special tokens for the output parser to recognise them, which a character
    grammar cannot guarantee. The prompt already instructs the DSML format and,
    for ``required``, ends inside the opened call block.
    """
    constraint = request.constraint
    if constraint is None:
        return None
    if constraint.kind == ConstraintKind.JSON:
        return {"json": json.loads(constraint.schema)} if constraint.schema else {"json_object": True}
    if constraint.kind == ConstraintKind.REGEX:
        return {"regex": constraint.schema}
    return None


def logprob_entries(step_logprobs: Sequence[dict[int, Any]] | None, token_ids: Sequence[int]) -> tuple[TokenLogprob, ...]:
    if not step_logprobs:
        return ()
    entries = []
    for token_id, candidates in zip(token_ids, step_logprobs, strict=False):
        chosen = candidates.get(token_id)
        ranked = sorted(candidates.items(), key=lambda item: getattr(item[1], "rank", 0) or 0)
        entries.append(
            TokenLogprob(
                logprob=float(chosen.logprob) if chosen is not None else float("-inf"),
                top_token_ids=tuple(int(tid) for tid, _ in ranked),
                top_logprobs=tuple(float(lp.logprob) for _, lp in ranked),
            )
        )
    return tuple(entries)


@dataclass
class SpecTracker:
    """Turns vLLM's cumulative per-request speculative metrics into per-step deltas."""

    drafted: int = 0
    accepted: int = 0

    def step(self, metrics: Any) -> tuple[int, int]:
        if metrics is None:
            return 0, 0
        drafted = int(getattr(metrics, "num_draft_tokens", 0) or 0)
        histogram = list(getattr(metrics, "histogram", []) or [])
        accepted = sum(j * count for j, count in enumerate(histogram))
        delta = (max(0, drafted - self.drafted), max(0, accepted - self.accepted))
        self.drafted, self.accepted = drafted, accepted
        return delta


def finish_reason(value: str | None) -> FinishReason:
    return {
        "stop": FinishReason.STOP,
        "length": FinishReason.LENGTH,
        "abort": FinishReason.ABORT,
        "error": FinishReason.ERROR,
    }.get(value or "", FinishReason.STOP)


# --------------------------------------------------------------------------- engine


@dataclass(frozen=True)
class VllmConfig:
    worker_id: str
    model_name: str
    role: Role
    #: Arguments exactly as for ``vllm serve`` (the model path first).
    vllm_args: str
    #: Port of vLLM's KV event ZMQ publisher; unique per worker on a node.
    kv_events_port: int = 5557
    kv_event_group: int | None = None


class _Counters:
    """Monotonic counters fed by the stat logger, read by load reports."""

    def __init__(self) -> None:
        self.running = 0
        self.waiting = 0
        self.kv_usage = 0.0
        self.prefill_tokens = 0
        self.decode_tokens = 0
        self.spec_draft_tokens = 0
        self.spec_accepted_tokens = 0

    def record(self, scheduler_stats: Any, iteration_stats: Any) -> None:
        if scheduler_stats is not None:
            self.running = int(scheduler_stats.num_running_reqs)
            self.waiting = int(scheduler_stats.num_waiting_reqs)
            self.kv_usage = float(scheduler_stats.kv_cache_usage)
            spec = getattr(scheduler_stats, "spec_decoding_stats", None)
            if spec is not None:
                self.spec_draft_tokens += int(spec.num_draft_tokens)
                self.spec_accepted_tokens += int(spec.num_accepted_tokens)
        if iteration_stats is not None:
            self.decode_tokens += int(iteration_stats.num_generation_tokens)
            prompt_stats = getattr(iteration_stats, "prompt_token_stats", None)
            self.prefill_tokens += int(getattr(prompt_stats, "computed", 0) or 0)


class VllmEngine(Engine):
    def __init__(self, config: VllmConfig) -> None:
        self._config = config
        self._counters = _Counters()
        self._hub = EventHub(self._load)
        self._translator = KvEventTranslator(group=config.kv_event_group)
        self._engine: Any = None
        self._vllm: Any = None
        self._block_size = 0
        self._total_blocks = 0
        self._context_limit = 0
        self._eos: int | None = None
        self._kv_task: asyncio.Task | None = None
        self._ready = False

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        import vllm  # noqa: PLC0415 - optional dependency
        from vllm.config.kv_events import KVEventsConfig  # noqa: PLC0415
        from vllm.engine.arg_utils import AsyncEngineArgs  # noqa: PLC0415
        from vllm.utils.argparse_utils import FlexibleArgumentParser  # noqa: PLC0415
        from vllm.v1.engine.async_llm import AsyncLLM  # noqa: PLC0415
        from vllm.v1.metrics.loggers import StatLoggerBase  # noqa: PLC0415

        self._vllm = vllm
        parser = AsyncEngineArgs.add_cli_args(FlexibleArgumentParser())
        argv = shlex.split(self._config.vllm_args)
        if argv and not argv[0].startswith("-"):
            argv = ["--model", argv[0], *argv[1:]]
        args = AsyncEngineArgs.from_cli_args(parser.parse_args(argv))
        args.kv_events_config = KVEventsConfig(
            enable_kv_cache_events=True,
            publisher="zmq",
            endpoint=f"tcp://*:{self._config.kv_events_port}",
            topic="leviathan",
        )
        args.enable_prefix_caching = True
        args.disable_log_stats = False

        counters = self._counters

        class LeviathanStats(StatLoggerBase):
            def __init__(self, vllm_config: Any, engine_index: int = 0) -> None:
                self.engine_index = engine_index

            def record(self, scheduler_stats, iteration_stats, mm_cache_stats=None, engine_idx=0):  # noqa: ANN001
                counters.record(scheduler_stats, iteration_stats)

            def log_engine_initialized(self) -> None:
                pass

        self._engine = AsyncLLM.from_engine_args(args, stat_loggers=[LeviathanStats])
        cache = self._engine.vllm_config.cache_config
        self._block_size = int(cache.block_size)
        self._total_blocks = int(cache.num_gpu_blocks or 0)
        self._context_limit = int(self._engine.vllm_config.model_config.max_model_len)
        tokenizer = self._engine.get_tokenizer()
        self._eos = getattr(tokenizer, "eos_token_id", None)
        self._kv_task = asyncio.create_task(self._follow_kv_events(), name="vllm-kv-events")
        self._ready = True
        log.info(
            "vLLM engine ready: block_size=%d gpu_blocks=%d max_model_len=%d",
            self._block_size,
            self._total_blocks,
            self._context_limit,
        )

    async def stop(self) -> None:
        self._ready = False
        if self._kv_task is not None:
            self._kv_task.cancel()
        self._hub.close()
        if self._engine is not None:
            self._engine.shutdown()

    async def _follow_kv_events(self) -> None:
        import msgspec  # noqa: PLC0415 - shipped with vLLM
        import zmq  # noqa: PLC0415
        import zmq.asyncio  # noqa: PLC0415
        from vllm.distributed.kv_events import KVEventBatch  # noqa: PLC0415

        decoder = msgspec.msgpack.Decoder(type=KVEventBatch)
        context = zmq.asyncio.Context.instance()
        socket = context.socket(zmq.SUB)
        socket.connect(f"tcp://127.0.0.1:{self._config.kv_events_port}")
        socket.setsockopt(zmq.SUBSCRIBE, b"leviathan")
        last_seq = None
        try:
            while True:
                _topic, seq_bytes, payload = await socket.recv_multipart()
                seq = int.from_bytes(seq_bytes, "big")
                if last_seq is not None and seq != last_seq + 1:
                    log.warning("missed vLLM KV event batches %d..%d", last_seq + 1, seq - 1)
                last_seq = seq
                events = self._translator.translate(decoder.decode(payload).events)
                self._hub.publish(events)
        finally:
            socket.close(linger=0)

    # ------------------------------------------------------------------ info

    def info(self) -> EngineInfo:
        return EngineInfo(
            worker_id=self._config.worker_id,
            backend="vllm",
            model=self._config.model_name,
            role=self._config.role,
            block_size=self._block_size,
            context_limit=self._context_limit,
            total_blocks=self._total_blocks,
            supports_kv_events=True,
            supports_constraints=True,
            supports_logprobs=True,
            deterministic=False,
            labels={"vllm": getattr(self._vllm, "__version__", "unknown")},
        )

    def health(self) -> Health:
        if not self._ready or self._engine is None:
            return Health(ready=False, status="starting")
        if self._engine.errored:
            return Health(ready=False, status=f"engine dead: {self._engine.dead_error}")
        return Health(ready=True, status="ok")

    def _load(self) -> LoadCounters:
        c = self._counters
        free = int(self._total_blocks * (1.0 - c.kv_usage))
        return LoadCounters(
            running=c.running,
            waiting=c.waiting,
            kv_usage=c.kv_usage,
            free_blocks=free,
            prefill_tokens=c.prefill_tokens,
            decode_tokens=c.decode_tokens,
            spec_draft_tokens=c.spec_draft_tokens,
            spec_accepted_tokens=c.spec_accepted_tokens,
        )

    def subscribe(self, *, snapshot: bool, load_report_interval_ms: int) -> EventSubscription:
        return self._hub.subscribe(snapshot=snapshot, load_report_interval_ms=load_report_interval_ms)

    async def abort(self, request_id: str) -> bool:
        if self._engine is None:
            return False
        await self._engine.abort(request_id)
        return True

    # ------------------------------------------------------------------ generate

    def _sampling(self, request: GenerateRequest) -> Any:
        from vllm.sampling_params import (  # noqa: PLC0415
            RequestOutputKind,
            SamplingParams,
            StructuredOutputsParams,
        )

        s = request.sampling
        structured = structured_outputs_kwargs(request)
        extra: dict[str, Any] = {}
        max_tokens = s.max_tokens or None
        if request.stage == Role.PREFILL:
            # Prefill only: compute the KV, keep it for the decode worker.
            max_tokens = 1
            extra["kv_transfer_params"] = {"do_remote_decode": True}
        elif request.stage == Role.DECODE and request.kv_transfer is not None:
            extra["kv_transfer_params"] = json.loads(request.kv_transfer.handle)
        return SamplingParams(
            temperature=s.temperature,
            top_p=s.top_p,
            max_tokens=max_tokens,
            seed=s.seed,
            logprobs=(s.top_logprobs or 0) if s.logprobs else None,
            stop_token_ids=list(s.stop_token_ids),
            ignore_eos=s.ignore_eos,
            output_kind=RequestOutputKind.DELTA,
            detokenize=False,
            skip_special_tokens=False,
            structured_outputs=StructuredOutputsParams(**structured) if structured else None,
            extra_args=extra or None,
        )

    async def generate(self, request: GenerateRequest) -> AsyncIterator[GenerateEvent]:
        if self._engine is None or not self._ready:
            raise UnavailableError("vLLM engine is not ready")
        if not request.prompt_token_ids:
            raise InvalidRequestError("empty prompt")
        if self._context_limit and len(request.prompt_token_ids) >= self._context_limit:
            raise InvalidRequestError(
                f"prompt of {len(request.prompt_token_ids)} tokens exceeds max_model_len {self._context_limit}"
            )
        from vllm import TokensPrompt  # noqa: PLC0415

        sampling = self._sampling(request)
        stops = stop_set(request, self._eos)
        thinking = request.hints.get("thinking") == "true"
        submitted = time.monotonic()
        ready_sent = False
        spec = SpecTracker()
        completion = 0
        try:
            async for out in self._engine.generate(
                TokensPrompt(prompt_token_ids=list(request.prompt_token_ids)),
                sampling,
                request.request_id,
                priority=request.priority,
                # The prompt ends with </think> outside thinking mode, so any
                # grammar applies from the first output token.
                reasoning_ended=not thinking,
            ):
                if not ready_sent:
                    ready_sent = True
                    elapsed_us = int((time.monotonic() - submitted) * 1e6)
                    transfer = None
                    if request.stage == Role.PREFILL and out.kv_transfer_params:
                        transfer = KvTransfer(
                            source_worker_id=self._config.worker_id,
                            handle=json.dumps(out.kv_transfer_params),
                            num_tokens=len(request.prompt_token_ids),
                        )
                    cached = int(out.num_cached_tokens or 0)
                    if request.stage == Role.DECODE and request.kv_transfer is not None:
                        cached = max(cached, request.kv_transfer.num_tokens)
                    yield Ready(
                        prompt_tokens=len(request.prompt_token_ids),
                        cached_tokens=cached,
                        queue_time_us=0,
                        prefill_time_us=elapsed_us,
                        kv_transfer=transfer,
                    )
                if request.stage == Role.PREFILL:
                    # The decode worker produces the output.
                    if out.finished:
                        break
                    continue
                choice = out.outputs[0]
                token_ids = list(choice.token_ids)
                logprobs = logprob_entries(choice.logprobs, token_ids)
                if out.finished and choice.finish_reason == "stop" and token_ids and token_ids[-1] in stops:
                    token_ids.pop()
                    logprobs = logprobs[: len(token_ids)]
                drafted, accepted = spec.step(getattr(choice, "spec_decode_metrics", None))
                if token_ids or drafted:
                    completion += len(token_ids)
                    yield TokenBatch(
                        token_ids=tuple(token_ids),
                        logprobs=logprobs,
                        draft_tokens=drafted,
                        accepted_tokens=accepted,
                    )
                if out.finished:
                    yield Finished(
                        reason=finish_reason(choice.finish_reason),
                        completion_tokens=completion,
                        decode_time_us=int((time.monotonic() - submitted) * 1e6),
                    )
                    return
            yield Finished(reason=FinishReason.STOP, completion_tokens=completion)
        except asyncio.CancelledError:
            await self._engine.abort(request.request_id)
            raise
        except Exception as error:  # noqa: BLE001 - reported to the gateway
            log.exception("vLLM request %s failed", request.request_id)
            yield Finished(reason=FinishReason.ERROR, completion_tokens=completion, error=str(error))
