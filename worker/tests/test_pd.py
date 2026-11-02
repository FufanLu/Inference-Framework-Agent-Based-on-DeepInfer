"""Prefill/decode disaggregation: a prefill worker hands its KV to a decode worker."""

from __future__ import annotations

import json

import grpc
import pytest
from support import EventRecorder, collect, make_request, prompt_ids

from leviathan_worker.engine import Role
from leviathan_worker.hashing import block_hashes
from leviathan_worker.pb import worker_pb2 as pb


async def test_prefill_to_decode_handoff(make_worker, tokenizer):
    prefill = await make_worker(role=Role.PREFILL, worker_id="sim-p0")
    decode = await make_worker(role=Role.DECODE, worker_id="sim-d0")
    unified = await make_worker(worker_id="sim-u0")
    decode_events = await EventRecorder.start(decode.stub)
    prompt = prompt_ids(tokenizer, "Summarize prefill/decode disaggregation. " * 6, thinking=True)
    hints = {"thinking": "true"}
    size = prefill.block_size
    hashes = block_hashes(prompt, size)

    p = await collect(prefill.stub, make_request("pd-1", prompt, seed=8, hints=hints, stage=pb.ROLE_PREFILL))
    assert p.kinds == ["ready", "finished"]
    assert p.finished.reason == pb.FINISH_REASON_STOP and p.finished.completion_tokens == 0
    transfer = p.ready.kv_transfer
    assert p.ready.HasField("kv_transfer")
    assert transfer.source_worker_id == "sim-p0"
    assert transfer.num_tokens == len(hashes) * size
    handle = json.loads(transfer.handle)
    assert handle["block_hashes"] == hashes
    assert handle["parents"] == [None, *hashes[:-1]]
    assert set(hashes) <= prefill.engine.cache.gpu_hashes()

    d = await collect(
        decode.stub, make_request("pd-1", prompt, seed=8, hints=hints, stage=pb.ROLE_DECODE, kv_transfer=transfer)
    )
    assert d.ready.cached_tokens == min(len(hashes) * size, len(prompt) - 1) > 0
    assert d.ready.prefill_time_us > 0  # includes the simulated transfer
    assert d.finished.reason == pb.FINISH_REASON_STOP
    await decode_events.wait_for(lambda rec: set(hashes) <= set(rec.stored(pb.MEDIUM_GPU)))

    # The disaggregated request produces exactly what a unified worker produces.
    u = await collect(unified.stub, make_request("pd-1", prompt, seed=8, hints=hints))
    assert u.ready.cached_tokens == 0
    assert d.token_ids == u.token_ids
    assert d.spec == u.spec
    await decode_events.close()


async def test_decode_reuses_its_own_cache_before_transferring(make_worker, tokenizer):
    prefill = await make_worker(role=Role.PREFILL, worker_id="sim-p0")
    # A slow link (0.1 Gbit/s) makes the transfer time dominate prefill time.
    decode = await make_worker(role=Role.DECODE, worker_id="sim-d0", kv_transfer_gbps=0.1)
    prompt = prompt_ids(tokenizer, "Shared context for two turns. " * 8)
    p = await collect(prefill.stub, make_request("t1", prompt, stage=pb.ROLE_PREFILL))
    transfer = p.ready.kv_transfer
    first = await collect(decode.stub, make_request("t1", prompt, stage=pb.ROLE_DECODE, kv_transfer=transfer))
    second = await collect(decode.stub, make_request("t2", prompt, stage=pb.ROLE_DECODE, kv_transfer=transfer))
    assert first.ready.cached_tokens == second.ready.cached_tokens > 0
    transfer_us = transfer.num_tokens * 1024 * 8 / (0.1 * 1_000)  # fast-dev: 1 KiB of KV per token
    assert first.ready.prefill_time_us >= transfer_us
    # Every block is local the second time: nothing to transfer.
    assert second.ready.prefill_time_us < transfer_us


async def test_decode_without_transfer_runs_both_stages(make_worker, tokenizer):
    decode = await make_worker(role=Role.DECODE)
    result = await collect(decode.stub, make_request("solo", prompt_ids(tokenizer, "hi"), stage=pb.ROLE_DECODE))
    assert result.ready.cached_tokens == 0 and result.token_ids


@pytest.mark.parametrize(
    "mutate",
    [
        lambda h: {**h, "block_hashes": [h["block_hashes"][0] ^ 1, *h["block_hashes"][1:]]},
        lambda h: {**h, "block_size": 32},
        lambda h: {**h, "format": "vllm"},
    ],
    ids=["wrong-prompt", "block-size", "foreign-handle"],
)
async def test_decode_rejects_a_mismatched_handle(make_worker, tokenizer, mutate):
    prefill = await make_worker(role=Role.PREFILL)
    decode = await make_worker(role=Role.DECODE)
    prompt = prompt_ids(tokenizer, "Some context. " * 10)
    p = await collect(prefill.stub, make_request("m", prompt, stage=pb.ROLE_PREFILL))
    transfer = pb.KvTransfer()
    transfer.CopyFrom(p.ready.kv_transfer)
    transfer.handle = json.dumps(mutate(json.loads(transfer.handle)))
    with pytest.raises(grpc.aio.AioRpcError) as caught:
        await collect(decode.stub, make_request("m", prompt, stage=pb.ROLE_DECODE, kv_transfer=transfer))
    assert caught.value.code() == grpc.StatusCode.INVALID_ARGUMENT
