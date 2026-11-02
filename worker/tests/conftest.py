"""Shared fixtures: generated stubs, the V4.1 tokenizer, and in-process workers."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

import pytest

WORKER_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = WORKER_DIR.parent
PROTO = REPO_DIR / "proto" / "leviathan" / "worker" / "v1" / "worker.proto"
GEN_DIR = WORKER_DIR / "leviathan_worker" / "_gen"


def _ensure_stubs() -> None:
    """Generate the protobuf stubs when missing or older than the proto (same flags as `make proto-py`)."""
    stub = GEN_DIR / "leviathan" / "worker" / "v1" / "worker_pb2.py"
    if stub.is_file() and stub.stat().st_mtime >= PROTO.stat().st_mtime:
        return
    import grpc_tools
    from grpc_tools import protoc

    GEN_DIR.mkdir(parents=True, exist_ok=True)
    include = Path(grpc_tools.__file__).parent / "_proto"
    code = protoc.main(
        [
            "grpc_tools.protoc",
            f"-I{REPO_DIR / 'proto'}",
            f"-I{include}",
            f"--python_out={GEN_DIR}",
            f"--grpc_python_out={GEN_DIR}",
            f"--pyi_out={GEN_DIR}",
            str(PROTO),
        ]
    )
    if code != 0:
        raise RuntimeError("protoc failed; run `make proto-py` from the repository root")


_ensure_stubs()

from support import Worker, start_worker  # noqa: E402

from leviathan_worker.engine import Role  # noqa: E402
from leviathan_worker.tokens import V41Tokenizer, default_tokenizer_path  # noqa: E402


@pytest.fixture(scope="session")
def tokenizer() -> V41Tokenizer:
    path = default_tokenizer_path()
    assert path is not None, "assets/tokenizers/v41/tokenizer.json is missing"
    return V41Tokenizer.from_file(path)


WorkerFactory = Callable[..., Awaitable[Worker]]


@pytest.fixture
async def make_worker(tokenizer: V41Tokenizer) -> AsyncIterator[WorkerFactory]:
    """Start in-process workers (fast-dev profile plus overrides); all are stopped after the test."""
    workers: list[Worker] = []

    async def factory(*, role: Role = Role.UNIFIED, worker_id: str = "sim-test", **overrides: object) -> Worker:
        worker = await start_worker(tokenizer, role=role, worker_id=worker_id, **overrides)
        workers.append(worker)
        return worker

    yield factory
    for worker in workers:
        await worker.close()


@pytest.fixture
async def worker(make_worker: WorkerFactory) -> Worker:
    return await make_worker()
