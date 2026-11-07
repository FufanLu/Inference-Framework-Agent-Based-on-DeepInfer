"""Fail a GPU job in seconds when the environment cannot use this GPU.

A job waits hours in the queue; finding out then that the PyTorch build needs
a newer driver (job 11634318: torch +cu130 on driver 570, every step failed)
wastes the wait. This checks, on the allocated node, that CUDA initialises,
a cuBLAS GEMM runs, Triton compiles and launches a kernel, and NCCL loads.
Exit code 0 means go.

    python deploy/slurm/preflight_gpu.py
"""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True
    )
    print("nvidia-smi:", smi.stdout.strip() or smi.stderr.strip())
    import torch

    print("torch", torch.__version__, "built for CUDA", torch.version.cuda)
    if not torch.cuda.is_available():
        print("FAIL: torch.cuda.is_available() is False (driver older than the CUDA this torch was built for?)")
        return 1
    a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    ref = (a.float() @ a.float()).cpu()
    got = (a @ a).float().cpu()
    if not torch.allclose(got, ref, rtol=2e-2, atol=1.0):
        print("FAIL: cuBLAS GEMM disagrees with fp32")
        return 1

    import triton
    import triton.language as tl

    @triton.jit
    def add(x, y, out, n, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        tl.store(out + i, tl.load(x + i, mask=m) + tl.load(y + i, mask=m), mask=m)

    x, y = torch.randn(1000, device="cuda"), torch.randn(1000, device="cuda")
    out = torch.empty_like(x)
    add[(4,)](x, y, out, 1000, BLOCK=256)
    torch.cuda.synchronize()
    if not torch.equal(out, x + y):
        print("FAIL: Triton kernel result is wrong")
        return 1
    print("triton", triton.__version__, "ok")
    print("nccl", ".".join(map(str, torch.cuda.nccl.version())))
    print("preflight ok on", torch.cuda.get_device_name(0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
