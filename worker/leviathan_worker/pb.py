"""Generated protobuf and gRPC modules for ``leviathan.worker.v1``.

``make proto-py`` writes them to ``leviathan_worker/_gen`` (gitignored). The
generated gRPC module imports ``leviathan.worker.v1.worker_pb2`` as a top-level
package, so ``_gen`` has to be on ``sys.path`` before importing it.
"""

from __future__ import annotations

import sys
from pathlib import Path

GEN_DIR = Path(__file__).resolve().parent / "_gen"
_STUB = GEN_DIR / "leviathan" / "worker" / "v1" / "worker_pb2.py"

if not _STUB.is_file():
    raise ImportError(
        f"generated protobuf stubs are missing ({_STUB}); "
        "run `make proto-py` from the repository root"
    )
if str(GEN_DIR) not in sys.path:
    sys.path.insert(0, str(GEN_DIR))

from leviathan.worker.v1 import worker_pb2, worker_pb2_grpc  # noqa: E402

__all__ = ["GEN_DIR", "worker_pb2", "worker_pb2_grpc"]
