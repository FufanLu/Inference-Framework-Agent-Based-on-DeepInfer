"""The worker's gRPC server and command line.

    python -m leviathan_worker.server --backend sim --port 50051 --worker-id sim-0 \\
        --role unified --model DeepSeek-V4.1-Flash --profile h200-vllm

SIGTERM or SIGINT drains the worker: Health turns not-ready, new Generate
calls get UNAVAILABLE, in-flight requests may finish within
``--shutdown-grace`` seconds, then the rest are aborted and the server stops.
A second signal skips the grace period.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import signal
import sys
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import grpc

from . import convert, logs
from .backends.sim import DEFAULT_MAX_WAITING, SimEngine
from .backends.sim_profile import SimProfile, available_profiles
from .engine import (
    DuplicateRequestError,
    Engine,
    EngineError,
    FinishReason,
    Health,
    InvalidRequestError,
    OverloadedError,
    Role,
    SubscriberOverflowError,
    UnavailableError,
)
from .pb import worker_pb2, worker_pb2_grpc
from .tokens import V41Tokenizer, default_tokenizer_path

log = logging.getLogger("leviathan_worker.server")

DEFAULT_LOAD_REPORT_INTERVAL_MS = 250
MIN_LOAD_REPORT_INTERVAL_MS = 10
MAX_MESSAGE_BYTES = 64 * 1024 * 1024
SERVER_OPTIONS = [
    ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
    ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
    ("grpc.so_reuseport", 0),  # a second worker on the same port must fail, not share it
]

_STATUS: dict[type[EngineError], grpc.StatusCode] = {
    InvalidRequestError: grpc.StatusCode.INVALID_ARGUMENT,
    DuplicateRequestError: grpc.StatusCode.ALREADY_EXISTS,
    OverloadedError: grpc.StatusCode.RESOURCE_EXHAUSTED,
    UnavailableError: grpc.StatusCode.UNAVAILABLE,
}


def _status_of(exc: EngineError) -> grpc.StatusCode:
    for kind, code in _STATUS.items():
        if isinstance(exc, kind):
            return code
    return grpc.StatusCode.INTERNAL


class WorkerServicer(worker_pb2_grpc.WorkerServicer):
    """Serves one engine over the ``leviathan.worker.v1.Worker`` service."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._draining = False
        self._in_flight = 0
        self._idle = asyncio.Event()
        self._idle.set()

    @property
    def draining(self) -> bool:
        return self._draining

    def start_draining(self) -> None:
        self._draining = True

    async def wait_idle(self) -> None:
        """Wait until no Generate call is in flight."""
        await self._idle.wait()

    async def Generate(self, request: Any, context: grpc.aio.ServicerContext) -> AsyncIterator[Any]:
        if self._draining:
            await context.abort(grpc.StatusCode.UNAVAILABLE, "worker is shutting down")
        try:
            req = convert.generate_request(request)
        except ValueError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        self._in_flight += 1
        self._idle.clear()
        try:
            async with contextlib.aclosing(self._engine.generate(req)) as events:
                async for event in events:
                    yield convert.generate_response(event)
        except EngineError as exc:
            await context.abort(_status_of(exc), str(exc))
        except Exception as exc:  # an engine bug must still end the stream with Finished
            log.exception("generate failed", extra={"request_id": req.request_id})
            yield worker_pb2.GenerateResponse(
                finished=worker_pb2.Finished(reason=int(FinishReason.ERROR), error=f"internal error: {exc}")
            )
        finally:
            self._in_flight -= 1
            if self._in_flight == 0:
                self._idle.set()

    async def Abort(self, request: Any, context: grpc.aio.ServicerContext) -> Any:
        return worker_pb2.AbortResponse(found=await self._engine.abort(request.request_id))

    async def SubscribeEvents(self, request: Any, context: grpc.aio.ServicerContext) -> AsyncIterator[Any]:
        interval = request.load_report_interval_ms or DEFAULT_LOAD_REPORT_INTERVAL_MS
        try:
            subscription = self._engine.subscribe(
                snapshot=request.snapshot, load_report_interval_ms=max(interval, MIN_LOAD_REPORT_INTERVAL_MS)
            )
        except EngineError as exc:
            await context.abort(_status_of(exc), str(exc))
        try:
            async for event in subscription:
                yield convert.worker_event(event)
        except SubscriberOverflowError as exc:
            await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(exc))
        finally:
            subscription.close()

    async def GetInfo(self, request: Any, context: grpc.aio.ServicerContext) -> Any:
        return convert.worker_info(self._engine.info())

    async def Health(self, request: Any, context: grpc.aio.ServicerContext) -> Any:
        health = Health(ready=False, status="draining") if self._draining else self._engine.health()
        return convert.health_response(health)


async def start_server(engine: Engine, host: str, port: int) -> tuple[grpc.aio.Server, WorkerServicer, int]:
    """Start serving ``engine`` (already started) on ``host:port``; port 0 picks a free one."""
    server = grpc.aio.server(options=SERVER_OPTIONS)
    servicer = WorkerServicer(engine)
    worker_pb2_grpc.add_WorkerServicer_to_server(servicer, server)
    bound = server.add_insecure_port(f"{host}:{port}")
    if not bound:
        raise OSError(f"cannot listen on {host}:{port}")
    await server.start()
    return server, servicer, bound


# --------------------------------------------------------------------------- configuration


@dataclass(frozen=True)
class ServerConfig:
    backend: str
    host: str
    port: int
    worker_id: str
    role: Role
    model: str
    profile: SimProfile
    tokenizer: Path
    max_waiting: int = DEFAULT_MAX_WAITING
    shutdown_grace_s: float = 30.0
    #: vllm: arguments exactly as for ``vllm serve``; deepinfer: JSON LLM kwargs.
    engine_args: str = ""
    kv_events_port: int = 5557
    kv_event_group: int | None = None
    model_path: str = ""


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Shows defaults, except for options whose default means "not set"."""

    def _get_help_string(self, action: argparse.Action) -> str | None:
        if action.default is None or action.default == []:
            return action.help
        return super()._get_help_string(action)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="leviathan-worker",
        description="Leviathan token-in/token-out inference worker.",
        formatter_class=_HelpFormatter,
    )
    parser.add_argument(
        "--backend",
        choices=["sim", "vllm", "deepinfer"],
        default="sim",
        help="engine backend: sim (simulated), vllm (Hopper GPUs), deepinfer (SM100 / Ascend 950 only)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="address to listen on")
    parser.add_argument("--port", type=int, default=50051, help="0 picks a free port")
    parser.add_argument("--worker-id", default="sim-0", help="ID reported in GetInfo and KV hand-offs")
    parser.add_argument("--role", choices=["unified", "prefill", "decode"], default="unified", help="PD role")
    parser.add_argument("--model", default="DeepSeek-V4.1-Flash", help="served model name")
    parser.add_argument(
        "--profile", default="h200-vllm", help=f"bundled profile ({', '.join(available_profiles())}) or a JSON path"
    )
    parser.add_argument("--tokenizer", type=Path, default=None, help="tokenizer.json (default: the repository copy)")
    parser.add_argument("--block-size", type=int, help="override the profile's block_size")
    parser.add_argument("--total-blocks", type=int, help="override the profile's GPU blocks")
    parser.add_argument("--cpu-blocks", type=int, help="override the profile's CPU blocks")
    parser.add_argument("--context-limit", type=int, help="override the profile's context_limit")
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", help="override any profile field (repeatable)"
    )
    real = parser.add_argument_group("real backends")
    real.add_argument(
        "--vllm-args",
        default="",
        help="vllm: engine arguments exactly as for `vllm serve` (model path first)",
    )
    real.add_argument("--kv-events-port", type=int, default=5557, help="vllm: port of vLLM's KV event publisher")
    real.add_argument("--kv-event-group", type=int, help="vllm: KV cache group to follow (default: first full-attention group)")
    real.add_argument("--deepinfer-model", default="", help="deepinfer: HF model ID or local checkpoint")
    real.add_argument("--deepinfer-kwargs", default="{}", help="deepinfer: JSON keyword arguments for LLM(...)")
    parser.add_argument("--max-waiting", type=int, default=DEFAULT_MAX_WAITING, help="queued requests before refusing")
    parser.add_argument("--shutdown-grace", type=float, default=30.0, help="seconds to let requests finish on SIGTERM")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"], help="log level")
    parser.add_argument("--log-format", default="json", choices=["json", "text"], help="log line format")
    return parser


def config_from_args(args: argparse.Namespace) -> ServerConfig:
    """Resolve and validate the command line; raises ValueError with a user-facing message."""
    overrides: dict[str, Any] = {}
    for item in args.set:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        overrides[key.strip()] = value.strip()
    for key in ("block_size", "total_blocks", "cpu_blocks", "context_limit"):
        if getattr(args, key) is not None:
            overrides[key] = getattr(args, key)
    profile = SimProfile.load(args.profile)
    if overrides:
        profile = profile.override(**overrides)
    tokenizer = args.tokenizer or default_tokenizer_path()
    if tokenizer is None or not Path(tokenizer).is_file():
        raise ValueError(f"tokenizer not found ({tokenizer}); pass --tokenizer or set LEVIATHAN_TOKENIZER")
    if args.max_waiting <= 0:
        raise ValueError("--max-waiting must be positive")
    engine_args = ""
    if args.backend == "vllm":
        if not args.vllm_args.strip():
            raise ValueError("--backend vllm needs --vllm-args (the model path and `vllm serve` flags)")
        engine_args = args.vllm_args
    elif args.backend == "deepinfer":
        try:
            json.loads(args.deepinfer_kwargs)
        except json.JSONDecodeError as exc:
            raise ValueError(f"--deepinfer-kwargs is not JSON: {exc}") from exc
        engine_args = args.deepinfer_kwargs
    return ServerConfig(
        backend=args.backend,
        host=args.host,
        port=args.port,
        worker_id=args.worker_id,
        role=Role[args.role.upper()],
        model=args.model,
        profile=profile,
        tokenizer=Path(tokenizer),
        max_waiting=args.max_waiting,
        shutdown_grace_s=max(0.0, args.shutdown_grace),
        engine_args=engine_args,
        kv_events_port=args.kv_events_port,
        kv_event_group=args.kv_event_group,
        model_path=args.deepinfer_model,
    )


def build_engine(config: ServerConfig) -> Engine:
    if config.backend == "sim":
        return SimEngine(
            profile=config.profile,
            tokenizer=V41Tokenizer.from_file(config.tokenizer),
            worker_id=config.worker_id,
            role=config.role,
            model=config.model,
            max_waiting=config.max_waiting,
        )
    if config.backend == "vllm":
        from .backends.vllm_backend import VllmConfig, VllmEngine  # noqa: PLC0415 - needs vLLM

        return VllmEngine(
            VllmConfig(
                worker_id=config.worker_id,
                model_name=config.model,
                role=config.role,
                vllm_args=config.engine_args,
                kv_events_port=config.kv_events_port,
                kv_event_group=config.kv_event_group,
            )
        )
    if config.backend == "deepinfer":
        from .backends.deepinfer_backend import DeepInferConfig, DeepInferEngine  # noqa: PLC0415

        kwargs = json.loads(config.engine_args or "{}")
        return DeepInferEngine(
            DeepInferConfig(
                worker_id=config.worker_id,
                model_name=config.model,
                llm_kwargs=kwargs,
                model_path=config.model_path or "deepseek-ai/DeepSeek-V4.1-Flash-1017",
                context_limit=config.profile.context_limit or 131_072,
            )
        )
    raise ValueError(f"unknown backend {config.backend!r}")


# --------------------------------------------------------------------------- serving


async def serve(config: ServerConfig) -> None:
    engine = build_engine(config)
    await engine.start()
    try:
        server, servicer, port = await start_server(engine, config.host, config.port)
    except BaseException:
        await engine.stop()
        raise
    log.info(
        "worker listening",
        extra={"address": f"{config.host}:{port}", "worker_id": config.worker_id, "backend": config.backend,
               "role": config.role.name.lower(), "model": config.model, "profile": config.profile.name},
    )

    shutdown, force = asyncio.Event(), asyncio.Event()

    def on_signal(signum: int) -> None:
        if not shutdown.is_set():
            log.info("shutting down", extra={"signal": signal.Signals(signum).name})
            shutdown.set()
        else:
            log.warning("second signal: stopping without waiting for requests")
            force.set()

    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(signum, on_signal, signum)
        except NotImplementedError:  # pragma: no cover - Windows
            signal.signal(signum, lambda s, _f: loop.call_soon_threadsafe(on_signal, s))

    try:
        await shutdown.wait()
        servicer.start_draining()
        drained = await _first(servicer.wait_idle(), force.wait(), timeout=config.shutdown_grace_s)
        if not drained:
            log.warning("grace period over; aborting in-flight requests")
    finally:
        await engine.stop()  # aborts what is left and ends event subscriptions
        await server.stop(grace=1.0)
        log.info("worker stopped", extra={"worker_id": config.worker_id})


async def _first(idle: Any, forced: Any, *, timeout: float) -> bool:
    """Wait for ``idle`` (True) or ``forced``/timeout (False)."""
    idle_task, forced_task = asyncio.ensure_future(idle), asyncio.ensure_future(forced)
    try:
        done, _ = await asyncio.wait({idle_task, forced_task}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        return idle_task in done
    finally:
        for task in (idle_task, forced_task):
            task.cancel()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logs.configure(args.log_level, args.log_format)
    try:
        config = config_from_args(args)
    except ValueError as exc:
        log.error("invalid configuration: %s", exc)
        return 2
    try:
        asyncio.run(serve(config))
    except OSError as exc:
        log.error("cannot start worker: %s", exc)
        return 1
    except Exception:
        log.exception("worker failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
