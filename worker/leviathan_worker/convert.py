"""Translation between protobuf messages and the engine dataclasses."""

from __future__ import annotations

from typing import Any

from .engine import (
    AllCleared,
    BlockRemoved,
    BlockStored,
    ConstraintKind,
    EngineInfo,
    Finished,
    GenerateEvent,
    GenerateRequest,
    Health,
    KvEvent,
    KvEventBatch,
    KvTransfer,
    LoadReport,
    OutputConstraint,
    Ready,
    Role,
    SamplingParams,
    TokenBatch,
    WorkerEvent,
)
from .pb import worker_pb2 as pb


def generate_request(msg: Any) -> GenerateRequest:
    """Convert a ``GenerateRequest`` message; raises ValueError for unknown enum values."""
    s = msg.sampling
    sampling = SamplingParams(
        temperature=s.temperature,
        top_p=s.top_p,
        max_tokens=s.max_tokens,
        seed=s.seed,
        logprobs=s.logprobs,
        top_logprobs=s.top_logprobs,
        stop_token_ids=tuple(s.stop_token_ids),
        ignore_eos=s.ignore_eos,
    )
    constraint = None
    if msg.HasField("constraint"):
        kind = _enum(ConstraintKind, msg.constraint.kind, "constraint.kind")
        if kind is not ConstraintKind.UNSPECIFIED:
            constraint = OutputConstraint(kind=kind, schema=msg.constraint.schema)
    kv_transfer = None
    if msg.HasField("kv_transfer"):
        t = msg.kv_transfer
        kv_transfer = KvTransfer(source_worker_id=t.source_worker_id, handle=t.handle, num_tokens=t.num_tokens)
    stage = _enum(Role, msg.stage, "stage")
    return GenerateRequest(
        request_id=msg.request_id,
        prompt_token_ids=tuple(msg.prompt_token_ids),
        sampling=sampling,
        constraint=constraint,
        stage=Role.UNIFIED if stage is Role.UNSPECIFIED else stage,
        kv_transfer=kv_transfer,
        tenant_id=msg.tenant_id,
        priority=msg.priority,
        hints=dict(msg.hints),
    )


def _enum(enum_type: Any, value: int, name: str) -> Any:
    try:
        return enum_type(value)
    except ValueError:
        raise ValueError(f"unknown {name} value {value}") from None


def _kv_transfer(t: KvTransfer) -> Any:
    return pb.KvTransfer(source_worker_id=t.source_worker_id, handle=t.handle, num_tokens=t.num_tokens)


def generate_response(event: GenerateEvent) -> Any:
    if isinstance(event, Ready):
        ready = pb.Ready(
            prompt_tokens=event.prompt_tokens,
            cached_tokens=event.cached_tokens,
            queue_time_us=event.queue_time_us,
            prefill_time_us=event.prefill_time_us,
        )
        if event.kv_transfer is not None:
            ready.kv_transfer.CopyFrom(_kv_transfer(event.kv_transfer))
        return pb.GenerateResponse(ready=ready)
    if isinstance(event, TokenBatch):
        return pb.GenerateResponse(
            tokens=pb.TokenBatch(
                token_ids=event.token_ids,
                logprobs=[
                    pb.TokenLogprob(
                        logprob=lp.logprob, top_token_ids=lp.top_token_ids, top_logprobs=lp.top_logprobs
                    )
                    for lp in event.logprobs
                ],
                draft_tokens=event.draft_tokens,
                accepted_tokens=event.accepted_tokens,
            )
        )
    if isinstance(event, Finished):
        return pb.GenerateResponse(
            finished=pb.Finished(
                reason=int(event.reason),
                completion_tokens=event.completion_tokens,
                error=event.error,
                decode_time_us=event.decode_time_us,
            )
        )
    raise TypeError(f"not a generate event: {event!r}")


def _kv_event(event: KvEvent) -> Any:
    if isinstance(event, BlockStored):
        stored = pb.BlockStored(block_hashes=event.block_hashes, medium=int(event.medium))
        if event.parent_hash is not None:
            stored.parent_hash = event.parent_hash
        return pb.KvEvent(stored=stored)
    if isinstance(event, BlockRemoved):
        return pb.KvEvent(removed=pb.BlockRemoved(block_hashes=event.block_hashes, medium=int(event.medium)))
    if isinstance(event, AllCleared):
        return pb.KvEvent(cleared=pb.AllCleared())
    raise TypeError(f"not a KV event: {event!r}")


def worker_event(event: WorkerEvent) -> Any:
    if isinstance(event, KvEventBatch):
        return pb.WorkerEvent(kv=pb.KvEventBatch(seq=event.seq, events=[_kv_event(e) for e in event.events]))
    if isinstance(event, LoadReport):
        return pb.WorkerEvent(
            load=pb.LoadReport(
                running=event.running,
                waiting=event.waiting,
                kv_usage=event.kv_usage,
                free_blocks=event.free_blocks,
                prefill_tokens_per_s=event.prefill_tokens_per_s,
                decode_tokens_per_s=event.decode_tokens_per_s,
                spec_accept_rate=event.spec_accept_rate,
                timestamp_ms=event.timestamp_ms,
            )
        )
    raise TypeError(f"not a worker event: {event!r}")


def worker_info(info: EngineInfo) -> Any:
    return pb.WorkerInfo(
        worker_id=info.worker_id,
        backend=info.backend,
        model=info.model,
        role=int(info.role),
        block_size=info.block_size,
        context_limit=info.context_limit,
        total_blocks=info.total_blocks,
        supports_kv_events=info.supports_kv_events,
        supports_constraints=info.supports_constraints,
        supports_logprobs=info.supports_logprobs,
        deterministic=info.deterministic,
        labels=dict(info.labels),
    )


def health_response(health: Health) -> Any:
    return pb.HealthResponse(ready=health.ready, status=health.status)
