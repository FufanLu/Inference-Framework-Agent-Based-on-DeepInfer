"""The DeepInfer adapter against a fake engine with DeepInfer's interface.

DeepInfer cannot run here (SM100 / Ascend 950 only); these tests pin down the
adapter's use of the interface: add_request / step / flush_inflight_reqs /
abort_request, one RequestOutput per token, draft_tok / accept_tok, and the
SamplingParams it builds.
"""

import asyncio
from dataclasses import dataclass

import pytest

from leviathan_worker.backends.deepinfer_backend import DeepInferConfig, DeepInferEngine
from leviathan_worker.engine import (
    ConstraintKind,
    Finished,
    FinishReason,
    GenerateRequest,
    InvalidRequestError,
    OutputConstraint,
    Ready,
    Role,
    SamplingParams,
    TokenBatch,
)

EOS = 1


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@dataclass
class FakeOutput:
    request_id: str
    token_id: int
    token_logprobs: float
    draft_tok: int
    accept_tok: int
    finished: bool
    finish_reason: str | None
    prompt_len: int


class FakeLLMEngine:
    """Emits two tokens per step (one accepted draft), then EOS."""

    eos_id = EOS

    def __init__(self, tokens_per_request=4):
        self.requests = {}
        self.params = {}
        self.aborted = []
        self.tokens_per_request = tokens_per_request

    def add_request(self, user_id, request_id, sampling_params, prompt_token_ids, multi_modal_data):
        assert prompt_token_ids.dtype.name == "int32"
        self.requests[request_id] = {"prompt_len": len(prompt_token_ids), "emitted": 0}
        self.params[request_id] = sampling_params

    def abort_request(self, request_id):
        self.aborted.append(request_id)
        self.requests.pop(request_id, None)

    def has_unfinished_requests(self):
        return bool(self.requests)

    def flush_inflight_reqs(self):
        return []

    def step(self):
        outputs = []
        for request_id, state in list(self.requests.items()):
            for k in range(2):
                state["emitted"] += 1
                done = state["emitted"] > self.tokens_per_request
                outputs.append(
                    FakeOutput(
                        request_id=request_id,
                        token_id=EOS if done else 100 + state["emitted"],
                        token_logprobs=-0.5,
                        draft_tok=3 if k == 0 else 0,  # target token of a verify step: 1 + 2 drafts
                        accept_tok=2 if k == 0 else 0,  # 1 + 1 accepted
                        finished=done,
                        finish_reason="stop" if done else None,
                        prompt_len=state["prompt_len"],
                    )
                )
                if done:
                    del self.requests[request_id]
                    break
        return outputs


def make_engine(fake=None):
    fake = fake or FakeLLMEngine()
    engine = DeepInferEngine(
        DeepInferConfig(worker_id="di-0", model_name="DeepSeek-V4.1-Flash"),
        load=lambda config: (fake, lambda raw: raw, FakeSamplingParams),
    )
    return engine, fake


def request(request_id="r1", **kwargs):
    return GenerateRequest(
        request_id=request_id,
        prompt_token_ids=(0, 5, 6, 7),
        sampling=SamplingParams(max_tokens=16, seed=7, stop_token_ids=(EOS,), logprobs=True),
        **kwargs,
    )


async def collect(engine, req):
    return [event async for event in engine.generate(req)]


async def test_streams_tokens_and_drops_eos():
    engine, fake = make_engine()
    await engine.start()
    try:
        events = await collect(engine, request())
    finally:
        await engine.stop()
    assert isinstance(events[0], Ready) and events[0].prompt_tokens == 4
    tokens = [t for e in events if isinstance(e, TokenBatch) for t in e.token_ids]
    assert tokens == [101, 102, 103, 104]
    assert isinstance(events[-1], Finished) and events[-1].reason == FinishReason.STOP
    assert events[-1].completion_tokens == 4
    drafted = sum(e.draft_tokens for e in events if isinstance(e, TokenBatch))
    accepted = sum(e.accepted_tokens for e in events if isinstance(e, TokenBatch))
    assert (drafted, accepted) == (6, 3)  # three verify steps, 2 drafts / 1 accepted each
    params = fake.params["r1"]
    assert params.seq_seed == 7 and params.eos_id == EOS and params.top_logprobs == 0


async def test_json_constraint_becomes_so_args_outside_thinking():
    engine, fake = make_engine()
    await engine.start()
    try:
        schema = '{"type": "object"}'
        await collect(engine, request("plain", constraint=OutputConstraint(ConstraintKind.JSON, schema)))
        await collect(
            engine,
            request("thinking", constraint=OutputConstraint(ConstraintKind.JSON, schema), hints={"thinking": "true"}),
        )
    finally:
        await engine.stop()
    assert fake.params["plain"].so_args == {"schema_type": "json", "schema": schema}
    assert fake.params["thinking"].so_args is None


async def test_cancelling_the_stream_aborts_the_request():
    fake = FakeLLMEngine(tokens_per_request=10_000)
    engine, fake = make_engine(fake)
    await engine.start()
    try:
        stream = engine.generate(request("long"))
        first = await stream.__anext__()
        assert isinstance(first, Ready)
        await stream.aclose()
        for _ in range(100):
            if "long" in fake.aborted:
                break
            await asyncio.sleep(0.01)
    finally:
        await engine.stop()
    assert "long" in fake.aborted


async def test_disaggregated_stages_are_rejected():
    engine, _ = make_engine()
    await engine.start()
    try:
        with pytest.raises(InvalidRequestError):
            await collect(engine, request(stage=Role.PREFILL))
    finally:
        await engine.stop()


async def test_reports_no_kv_events():
    engine, _ = make_engine()
    info = engine.info()
    assert info.backend == "deepinfer" and not info.supports_kv_events
