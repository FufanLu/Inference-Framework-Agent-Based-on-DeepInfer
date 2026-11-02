"""End-to-end tests over gRPC: streaming, determinism, limits, errors and cancellation."""

from __future__ import annotations

import asyncio

import grpc
import pytest
from support import TIMEOUT, collect, make_request, prompt_ids

from leviathan_worker.pb import worker_pb2 as pb


async def test_get_info_and_health(worker):
    info = await worker.stub.GetInfo(pb.GetInfoRequest())
    assert info.backend == "sim"
    assert info.worker_id == "sim-test"
    assert info.model == "DeepSeek-V4.1-Flash"
    assert info.role == pb.ROLE_UNIFIED
    assert info.block_size == 16 and info.total_blocks == 512 and info.context_limit == 8192
    assert info.deterministic and info.supports_kv_events and info.supports_constraints and info.supports_logprobs
    assert dict(info.labels) == {"profile": "fast-dev"}
    health = await worker.stub.Health(pb.HealthRequest())
    assert health.ready and health.status == "serving"


async def test_streams_ready_then_tokens_then_finished(worker, tokenizer):
    prompt = prompt_ids(tokenizer, "What does a prefix cache do?")
    result = await collect(worker.stub, make_request("basic", prompt, seed=3))
    kinds = result.kinds
    assert kinds[0] == "ready" and kinds[-1] == "finished"
    assert set(kinds[1:-1]) == {"tokens"} and len(kinds) > 3
    assert result.ready.prompt_tokens == len(prompt)
    assert result.ready.cached_tokens == 0
    assert result.finished.reason == pb.FINISH_REASON_STOP
    assert result.finished.completion_tokens == len(result.token_ids) > 0
    assert result.finished.decode_time_us > 0
    assert tokenizer.eos_id not in result.token_ids
    # The prefill step samples one token; each decode step proposes 5 DSpark drafts and
    # emits one token plus the accepted ones. (Batches merge if the stream lags, so
    # check the totals, which merging preserves.)
    drafted, accepted = result.spec
    assert drafted % 5 == 0 and 0 < accepted <= drafted
    assert len(result.token_ids) == 1 + drafted // 5 + accepted
    assert all(b.token_ids and b.accepted_tokens <= b.draft_tokens for b in result.batches)
    text = tokenizer.decode(result.token_ids)
    assert text and "</think>" not in text and "<think>" not in text
    assert worker.engine.snapshot_state()["requests"] == 0


async def test_output_is_deterministic(make_worker, tokenizer):
    a, b = await make_worker(worker_id="a"), await make_worker(worker_id="b")
    prompt = prompt_ids(tokenizer, "Explain continuous batching.", thinking=True)
    hints = {"thinking": "true"}
    alone = await collect(a.stub, make_request("x1", prompt, seed=42, hints=hints))
    # Same inputs on another worker while it is busy with unrelated requests.
    noise = [
        collect(b.stub, make_request(f"noise-{i}", prompt_ids(tokenizer, f"noise {i}"), seed=i)) for i in range(6)
    ]
    busy, *_ = await asyncio.gather(collect(b.stub, make_request("x2", prompt, seed=42, hints=hints)), *noise)
    assert alone.token_ids == busy.token_ids
    assert alone.spec == busy.spec
    other_seed = await collect(a.stub, make_request("x3", prompt, seed=43, hints=hints))
    assert other_seed.token_ids != alone.token_ids


async def test_logprobs_are_deterministic_and_do_not_change_tokens(worker, tokenizer):
    prompt = prompt_ids(tokenizer, "Tell me about speculative decoding.")
    plain = await collect(worker.stub, make_request("lp0", prompt, seed=9))
    with_lp = await collect(worker.stub, make_request("lp1", prompt, seed=9, logprobs=True, top_logprobs=3))
    again = await collect(worker.stub, make_request("lp2", prompt, seed=9, logprobs=True, top_logprobs=3))
    assert with_lp.token_ids == plain.token_ids
    assert all(not b.logprobs for b in plain.batches)
    entries = [(t, lp) for b in with_lp.batches for t, lp in zip(b.token_ids, b.logprobs)]
    assert len(entries) == len(with_lp.token_ids)
    for token, lp in entries:
        assert lp.logprob <= 0
        assert list(lp.top_token_ids)[0] == token and len(set(lp.top_token_ids)) == 3
        values = list(lp.top_logprobs)
        assert values[0] == pytest.approx(lp.logprob)
        assert all(x > y for x, y in zip(values, values[1:]))
    assert [lp for b in again.batches for lp in b.logprobs] == [lp for _, lp in entries]


async def test_max_tokens_finishes_with_length(worker, tokenizer):
    prompt = prompt_ids(tokenizer, "Write a long essay about KV caches.")
    full = await collect(worker.stub, make_request("len0", prompt, seed=5))
    assert len(full.token_ids) > 5
    result = await collect(worker.stub, make_request("len1", prompt, seed=5, max_tokens=5))
    assert result.finished.reason == pb.FINISH_REASON_LENGTH
    assert result.finished.completion_tokens == 5
    assert result.token_ids == full.token_ids[:5]


async def test_ignore_eos_generates_exactly_max_tokens(worker, tokenizer):
    prompt = prompt_ids(tokenizer, "Keep going.")
    result = await collect(worker.stub, make_request("eos", prompt, seed=2, max_tokens=300, ignore_eos=True))
    assert result.finished.reason == pb.FINISH_REASON_LENGTH
    assert len(result.token_ids) == 300


async def test_stop_token_ids_end_generation_without_streaming_them(worker, tokenizer):
    prompt = prompt_ids(tokenizer, "Say something.")
    full = await collect(worker.stub, make_request("stop0", prompt, seed=11))
    stop = full.token_ids[3]
    cut = full.token_ids.index(stop)
    result = await collect(worker.stub, make_request("stop1", prompt, seed=11, stop_token_ids=[stop]))
    assert result.finished.reason == pb.FINISH_REASON_STOP
    assert result.token_ids == full.token_ids[:cut]


async def test_context_limit_caps_output(make_worker, tokenizer):
    worker = await make_worker(context_limit=64)
    prompt = prompt_ids(tokenizer, "word " * 100)[:60]
    result = await collect(worker.stub, make_request("ctx", prompt, seed=1))
    assert result.finished.reason == pb.FINISH_REASON_LENGTH
    assert len(result.token_ids) == 4


@pytest.mark.parametrize(
    ("prompt", "code"),
    [
        ([], grpc.StatusCode.INVALID_ARGUMENT),
        ([5] * 8192, grpc.StatusCode.INVALID_ARGUMENT),
        ([10**7], grpc.StatusCode.INVALID_ARGUMENT),
    ],
    ids=["empty", "too-long", "unknown-token"],
)
async def test_invalid_prompts_are_rejected(worker, prompt, code):
    with pytest.raises(grpc.aio.AioRpcError) as caught:
        await collect(worker.stub, make_request("bad", prompt))
    assert caught.value.code() == code


async def test_invalid_constraints_are_rejected(worker, tokenizer):
    prompt = prompt_ids(tokenizer, "hi")
    for i, constraint in enumerate(
        [
            (pb.OutputConstraint.KIND_JSON, "{not json"),
            (pb.OutputConstraint.KIND_REGEX, "(unclosed"),
            (pb.OutputConstraint.KIND_TOOL_CALL, "[]"),
        ]
    ):
        with pytest.raises(grpc.aio.AioRpcError) as caught:
            await collect(worker.stub, make_request(f"c{i}", prompt, constraint=constraint))
        assert caught.value.code() == grpc.StatusCode.INVALID_ARGUMENT


async def test_duplicate_request_id_is_rejected(make_worker, tokenizer):
    worker = await make_worker(decode_step_base_us=20_000, mean_output_tokens=400)
    prompt = prompt_ids(tokenizer, "hello")
    call = worker.stub.Generate(make_request("dup", prompt))
    assert (await asyncio.wait_for(call.read(), TIMEOUT)).HasField("ready")
    with pytest.raises(grpc.aio.AioRpcError) as caught:
        await collect(worker.stub, make_request("dup", prompt))
    assert caught.value.code() == grpc.StatusCode.ALREADY_EXISTS
    call.cancel()


async def test_abort_rpc_finishes_the_stream(make_worker, tokenizer):
    worker = await make_worker(decode_step_base_us=20_000, mean_output_tokens=400)
    call = worker.stub.Generate(make_request("ab", prompt_ids(tokenizer, "a long answer please")))
    assert (await asyncio.wait_for(call.read(), TIMEOUT)).HasField("ready")
    assert (await worker.stub.Abort(pb.AbortRequest(request_id="ab"))).found
    responses = []
    while (message := await asyncio.wait_for(call.read(), TIMEOUT)) is not grpc.aio.EOF:
        responses.append(message)
    assert responses[-1].finished.reason == pb.FINISH_REASON_ABORT
    assert not (await worker.stub.Abort(pb.AbortRequest(request_id="ab"))).found
    assert worker.engine.snapshot_state() == {
        "running": 0, "waiting": 0, "requests": 0, "used_blocks": 0, "free_blocks": 512,
    }


async def test_cancelled_stream_frees_the_request(make_worker, tokenizer):
    worker = await make_worker(decode_step_base_us=20_000, mean_output_tokens=400)
    call = worker.stub.Generate(make_request("cx", prompt_ids(tokenizer, "a long answer please " * 8)))
    assert (await asyncio.wait_for(call.read(), TIMEOUT)).HasField("ready")
    assert (await asyncio.wait_for(call.read(), TIMEOUT)).HasField("tokens")
    assert worker.engine.snapshot_state()["running"] == 1
    assert worker.engine.cache.used_blocks > 0
    call.cancel()

    async def freed() -> None:
        while worker.engine.snapshot_state()["requests"]:
            await asyncio.sleep(0.01)

    await asyncio.wait_for(freed(), TIMEOUT)
    assert worker.engine.cache.used_blocks == 0
    worker.engine.cache.check_invariants()


async def test_draining_refuses_new_requests(worker, tokenizer):
    worker.servicer.start_draining()
    health = await worker.stub.Health(pb.HealthRequest())
    assert not health.ready and health.status == "draining"
    with pytest.raises(grpc.aio.AioRpcError) as caught:
        await collect(worker.stub, make_request("late", prompt_ids(tokenizer, "hi")))
    assert caught.value.code() == grpc.StatusCode.UNAVAILABLE
