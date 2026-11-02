"""The command line: start a worker process, use it, and shut it down with SIGTERM."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import grpc
import pytest
from support import make_request

from leviathan_worker.pb import worker_pb2 as pb
from leviathan_worker.pb import worker_pb2_grpc
from leviathan_worker.tokens import default_tokenizer_path

WORKER_DIR = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_worker(*args: str) -> subprocess.Popen[str]:
    env = {**os.environ, "PYTHONPATH": str(WORKER_DIR)}
    return subprocess.Popen(
        [sys.executable, "-m", "leviathan_worker.server", *args],
        cwd=WORKER_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def wait_ready(stub: worker_pb2_grpc.WorkerStub, proc: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        assert proc.poll() is None, proc.communicate()[1]
        try:
            if stub.Health(pb.HealthRequest(), timeout=1).ready:
                return
        except grpc.RpcError:
            time.sleep(0.05)
    raise AssertionError("worker did not become ready")


def test_worker_process_serves_and_drains_on_sigterm(tokenizer):
    port = free_port()
    proc = run_worker(
        "--backend", "sim", "--host", "127.0.0.1", "--port", str(port), "--worker-id", "cli-p0",
        "--role", "prefill", "--model", "DeepSeek-V4.1-Flash", "--profile", "fast-dev",
        "--tokenizer", str(default_tokenizer_path()), "--block-size", "32", "--total-blocks", "64",
        "--set", "mean_output_tokens=10", "--log-format", "json",
    )
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = worker_pb2_grpc.WorkerStub(channel)
            wait_ready(stub, proc)
            info = stub.GetInfo(pb.GetInfoRequest(), timeout=5)
            assert (info.worker_id, info.role, info.block_size, info.total_blocks) == ("cli-p0", pb.ROLE_PREFILL, 32, 64)
            prompt = tokenizer.encode("<｜User｜>" + "hello " * 40 + "<｜Assistant｜></think>")
            responses = list(stub.Generate(make_request("cli-1", prompt, stage=pb.ROLE_PREFILL), timeout=10))
            assert responses[0].ready.kv_transfer.source_worker_id == "cli-p0"
            assert responses[-1].finished.reason == pb.FINISH_REASON_STOP
    finally:
        proc.send_signal(signal.SIGTERM)
        _, stderr = proc.communicate(timeout=20)
    assert proc.returncode == 0, stderr
    records = [json.loads(line) for line in stderr.splitlines() if line.startswith("{")]
    messages = [r["msg"] for r in records]
    assert "worker listening" in messages and "shutting down" in messages and "worker stopped" in messages
    listening = next(r for r in records if r["msg"] == "worker listening")
    assert listening["address"] == f"127.0.0.1:{port}" and listening["role"] == "prefill"


@pytest.mark.parametrize(
    "args",
    [["--profile", "no-such-profile"], ["--set", "bogus=1"], ["--block-size", "0"], ["--tokenizer", "/nonexistent.json"]],
)
def test_invalid_configuration_exits_with_status_2(args):
    proc = run_worker("--port", "0", *args)
    _, stderr = proc.communicate(timeout=20)
    assert proc.returncode == 2, stderr
    assert "invalid configuration" in stderr


def test_real_backends_need_their_engine_arguments():
    import pytest

    from leviathan_worker.server import build_parser, config_from_args

    with pytest.raises(ValueError, match="--vllm-args"):
        config_from_args(build_parser().parse_args(["--backend", "vllm"]))
    config = config_from_args(
        build_parser().parse_args(["--backend", "vllm", "--vllm-args", "/models/v41 --tensor-parallel-size 4", "--kv-events-port", "5558"])
    )
    assert config.engine_args.startswith("/models/v41") and config.kv_events_port == 5558
    with pytest.raises(ValueError, match="not JSON"):
        config_from_args(build_parser().parse_args(["--backend", "deepinfer", "--deepinfer-kwargs", "{nope"]))
