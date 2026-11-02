"""Test helpers: in-process workers, request builders, stream collectors and checkers."""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import grpc

from leviathan_worker.backends.sim import SimEngine
from leviathan_worker.backends.sim_profile import SimProfile
from leviathan_worker.engine import Role
from leviathan_worker.pb import worker_pb2 as pb
from leviathan_worker.pb import worker_pb2_grpc
from leviathan_worker.server import WorkerServicer, start_server
from leviathan_worker.tokens import BOS, V41Tokenizer

TIMEOUT = 10.0
DSML = "｜DSML｜"


# --------------------------------------------------------------------------- workers


@dataclass
class Worker:
    engine: SimEngine
    server: grpc.aio.Server
    servicer: WorkerServicer
    channel: grpc.aio.Channel
    stub: worker_pb2_grpc.WorkerStub
    port: int

    @property
    def block_size(self) -> int:
        return self.engine.profile.block_size

    async def close(self) -> None:
        await self.channel.close()
        await self.engine.stop()
        await self.server.stop(grace=None)


async def start_worker(
    tokenizer: V41Tokenizer,
    *,
    role: Role = Role.UNIFIED,
    worker_id: str = "sim-test",
    event_queue_size: int = 1024,
    **overrides: Any,
) -> Worker:
    profile = SimProfile.load("fast-dev")
    if overrides:
        profile = profile.override(**overrides)
    engine = SimEngine(
        profile=profile, tokenizer=tokenizer, worker_id=worker_id, role=role, event_queue_size=event_queue_size
    )
    await engine.start()
    server, servicer, port = await start_server(engine, "127.0.0.1", 0)
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    return Worker(engine, server, servicer, channel, worker_pb2_grpc.WorkerStub(channel), port)


# --------------------------------------------------------------------------- prompts and requests


def prompt_ids(
    tokenizer: V41Tokenizer, user: str, *, thinking: bool = False, system: str = "", required_tool: bool = False
) -> list[int]:
    """A V4.1-style prompt: BOS, optional system text, one user turn and the assistant prefix."""
    text = f"{BOS}{system}<｜User｜>{user}<｜Assistant｜>{'<think>' if thinking else '</think>'}"
    if required_tool:
        text += f"\n\n<{DSML} calls>\n"
    return tokenizer.encode(text)


def make_request(
    request_id: str,
    prompt: list[int],
    *,
    seed: int = 1,
    max_tokens: int = 0,
    constraint: tuple[int, str] | None = None,
    hints: dict[str, str] | None = None,
    stage: int = pb.ROLE_UNIFIED,
    kv_transfer: Any = None,
    logprobs: bool = False,
    top_logprobs: int = 0,
    stop_token_ids: list[int] | None = None,
    ignore_eos: bool = False,
) -> Any:
    request = pb.GenerateRequest(
        request_id=request_id,
        prompt_token_ids=prompt,
        sampling=pb.SamplingParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=max_tokens,
            seed=seed,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            stop_token_ids=stop_token_ids or [],
            ignore_eos=ignore_eos,
        ),
        stage=stage,
        hints=hints or {},
    )
    if constraint is not None:
        request.constraint.kind, request.constraint.schema = constraint
    if kv_transfer is not None:
        request.kv_transfer.CopyFrom(kv_transfer)
    return request


@dataclass
class Result:
    responses: list[Any]

    @property
    def kinds(self) -> list[str]:
        return [r.WhichOneof("event") for r in self.responses]

    @property
    def ready(self) -> Any:
        return next(r.ready for r in self.responses if r.HasField("ready"))

    @property
    def finished(self) -> Any:
        return self.responses[-1].finished

    @property
    def batches(self) -> list[Any]:
        return [r.tokens for r in self.responses if r.HasField("tokens")]

    @property
    def token_ids(self) -> list[int]:
        return [t for b in self.batches for t in b.token_ids]

    @property
    def spec(self) -> tuple[int, int]:
        return sum(b.draft_tokens for b in self.batches), sum(b.accepted_tokens for b in self.batches)


async def collect(stub: Any, request: Any) -> Result:
    async def run() -> Result:
        return Result([response async for response in stub.Generate(request)])

    return await asyncio.wait_for(run(), TIMEOUT)


# --------------------------------------------------------------------------- events


@dataclass
class EventRecorder:
    """Reads a SubscribeEvents stream in the background."""

    call: Any
    kv: list[Any] = field(default_factory=list)
    loads: list[Any] = field(default_factory=list)
    error: BaseException | None = None
    _task: asyncio.Task[None] | None = None
    _changed: asyncio.Event = field(default_factory=asyncio.Event)

    @classmethod
    async def start(cls, stub: Any, *, snapshot: bool = False, interval_ms: int = 250) -> EventRecorder:
        """Subscribe and wait for the first load report, which the worker sends right after
        registering the subscription (and after the snapshot, if requested)."""
        call = stub.SubscribeEvents(pb.SubscribeEventsRequest(snapshot=snapshot, load_report_interval_ms=interval_ms))
        recorder = cls(call)
        recorder._task = asyncio.create_task(recorder._read())
        await recorder.wait_for(lambda rec: rec.loads)
        return recorder

    async def _read(self) -> None:
        try:
            async for event in self.call:
                (self.kv if event.HasField("kv") else self.loads).append(event.kv if event.HasField("kv") else event.load)
                self._changed.set()
        except grpc.aio.AioRpcError as exc:
            if exc.code() != grpc.StatusCode.CANCELLED:
                self.error = exc
        finally:
            self._changed.set()

    async def wait_for(self, predicate: Callable[[EventRecorder], bool], timeout: float = TIMEOUT) -> None:
        async def poll() -> None:
            while not predicate(self):
                if self._task is not None and self._task.done():
                    raise AssertionError(f"event stream ended: {self.error}")
                self._changed.clear()
                await self._changed.wait()

        await asyncio.wait_for(poll(), timeout)

    def events(self) -> list[Any]:
        return [e for batch in self.kv for e in batch.events]

    def stored(self, medium: int) -> list[int]:
        return [h for e in self.events() if e.HasField("stored") and e.stored.medium == medium for h in e.stored.block_hashes]

    def removed(self, medium: int) -> list[int]:
        return [h for e in self.events() if e.HasField("removed") and e.removed.medium == medium for h in e.removed.block_hashes]

    def mirror(self) -> dict[int, set[int]]:
        return replay(self.kv)

    async def close(self) -> None:
        self.call.cancel()
        if self._task is not None:
            await asyncio.gather(self._task, return_exceptions=True)


def replay(batches: list[Any]) -> dict[int, set[int]]:
    """The cached hashes per medium after applying KV event batches in order."""
    state: dict[int, set[int]] = {pb.MEDIUM_GPU: set(), pb.MEDIUM_CPU: set()}
    for batch in batches:
        for event in batch.events:
            kind = event.WhichOneof("event")
            if kind == "stored":
                state[event.stored.medium].update(event.stored.block_hashes)
            elif kind == "removed":
                state[event.removed.medium].difference_update(event.removed.block_hashes)
            elif kind == "cleared":
                for hashes in state.values():
                    hashes.clear()
    return state


# --------------------------------------------------------------------------- output checks


TOOL_BLOCK = re.compile(
    r"\n\n<｜DSML｜ calls>\n(?P<invokes>.*)\n</｜DSML｜ calls>\Z",
    re.DOTALL,
)
INVOKE = re.compile(r'<｜DSML｜ invoke name="(?P<name>[^"]+)">\n(?P<params>.*?)\n</｜DSML｜ invoke>', re.DOTALL)
PARAM = re.compile(
    r'<｜DSML｜ parameter name="(?P<key>[^"]+)" string="(?P<string>true|false)">(?P<value>.*?)</｜DSML｜ parameter>',
    re.DOTALL,
)


def parse_invocations(invokes: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse DSML invoke blocks, checking that nothing but separators sits between them."""
    calls = []
    pos = 0
    for match in INVOKE.finditer(invokes):
        assert invokes[pos : match.start()] in ("", "\n"), f"unexpected text between invocations: {invokes!r}"
        pos = match.end()
        params_text = match.group("params")
        args: dict[str, Any] = {}
        ppos = 0
        for p in PARAM.finditer(params_text):
            assert params_text[ppos : p.start()] in ("", "\n"), f"unexpected text between parameters: {params_text!r}"
            ppos = p.end()
            value = p.group("value")
            args[p.group("key")] = value if p.group("string") == "true" else json.loads(value)
        assert params_text[ppos:] == "", f"trailing text in invocation: {params_text!r}"
        calls.append((match.group("name"), args))
    assert invokes[pos:] == "", f"trailing text after invocations: {invokes!r}"
    return calls


def validate(instance: Any, schema: Any, root: Any = None) -> None:
    """A small JSON Schema validator for the keywords the generator supports; raises AssertionError."""
    root = schema if root is None else root
    assert schema is not False, f"{instance!r} is not allowed by the schema `false`"
    if schema is True or schema == {} or schema is None:
        return
    if "$ref" in schema:
        node = root
        for part in schema["$ref"].lstrip("#/").split("/"):
            node = node[part]
        return validate(instance, node, root)
    if "const" in schema:
        assert instance == schema["const"], f"{instance!r} != const {schema['const']!r}"
    if "enum" in schema:
        assert instance in schema["enum"], f"{instance!r} not in enum {schema['enum']!r}"
    if "allOf" in schema:
        for sub in schema["allOf"]:
            validate(instance, sub, root)
    for key in ("anyOf", "oneOf"):
        if key in schema:
            errors = []
            for sub in schema[key]:
                try:
                    validate(instance, sub, root)
                    break
                except AssertionError as exc:
                    errors.append(str(exc))
            else:
                raise AssertionError(f"{instance!r} matches no {key} option: {errors}")
    kind = schema.get("type")
    if kind is not None:
        kinds = kind if isinstance(kind, list) else [kind]
        assert any(_is_type(instance, k) for k in kinds), f"{instance!r} is not of type {kind}"
    if isinstance(instance, dict):
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            assert key in instance, f"missing required property {key!r} in {instance!r}"
        if schema.get("additionalProperties") is False:
            extra = set(instance) - set(properties)
            assert not extra, f"unexpected properties {extra}"
        for key, value in instance.items():
            if key in properties:
                validate(value, properties[key], root)
    if isinstance(instance, list):
        assert len(instance) >= schema.get("minItems", 0), f"too few items in {instance!r}"
        assert len(instance) <= schema.get("maxItems", math.inf), f"too many items in {instance!r}"
        if schema.get("uniqueItems"):
            keys = [json.dumps(i, sort_keys=True) for i in instance]
            assert len(keys) == len(set(keys)), f"items are not unique: {instance!r}"
        prefix = schema.get("prefixItems", [])
        for i, item in enumerate(instance):
            if i < len(prefix):
                validate(item, prefix[i], root)
            elif "items" in schema:
                validate(item, schema["items"], root)
    if isinstance(instance, str):
        assert len(instance) >= schema.get("minLength", 0), f"{instance!r} is too short"
        assert len(instance) <= schema.get("maxLength", math.inf), f"{instance!r} is too long"
        if "pattern" in schema:
            assert re.search(schema["pattern"], instance), f"{instance!r} does not match {schema['pattern']!r}"
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema:
            assert instance >= schema["minimum"], f"{instance} < minimum"
        if "maximum" in schema:
            assert instance <= schema["maximum"], f"{instance} > maximum"
        if "exclusiveMinimum" in schema:
            assert instance > schema["exclusiveMinimum"], f"{instance} <= exclusiveMinimum"
        if "exclusiveMaximum" in schema:
            assert instance < schema["exclusiveMaximum"], f"{instance} >= exclusiveMaximum"
        if "multipleOf" in schema:
            ratio = instance / schema["multipleOf"]
            assert abs(ratio - round(ratio)) < 1e-9, f"{instance} is not a multiple of {schema['multipleOf']}"


def _is_type(value: Any, kind: str) -> bool:
    return {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }[kind]
