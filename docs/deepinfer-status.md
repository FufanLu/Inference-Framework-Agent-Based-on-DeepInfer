# DeepInfer in this project

**Bottom line: the `deepinfer` backend is written against DeepInfer's source interfaces and covered by interface tests, but it has never run on real hardware.** The GPUs available to this project are the Northeastern Explorer cluster's H200 / H100 / A100 (SM90 or older), while DeepInfer's open release supports only NVIDIA SM100 (B200/GB200) and Ascend 950.

## Evidence (DeepInfer `main`, checked 2026-11-01)

1. The **README** lists `NVIDIA SM100 GPUs` and `Huawei Ascend 950 NPUs` as the supported hardware.
2. **Engine construction fails elsewhere**: `deepinfer/engine/llm_engine.py:304` calls `get_bf16_tflops_limit()`, which in `deep_cluster/hardware/device.py` recognizes only `is_sm_major_10()` (compute capability 10.x) and Ascend, and otherwise raises `RuntimeError("unknown GPU type")`.
3. **The core kernels need FP4 tensor cores**, which Hopper does not have:
   - MoE: `deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous` at `deepinfer/model_executor/models/v4_modules/mlp.py:119`, and `deep_gemm.fp8_fp4_mega_moe` at `mlp.py:255`;
   - sparse indexer attention: `fp8_fp4_mqa_logits` / `fp8_fp4_paged_mqa_logits` and friends from `deepinfer/model_executor/models/v4_modules/index_attn.py:299` on.

Point 2 is easy to work around. Point 3 means rewriting the MoE and indexer kernels for Hopper (or dequantizing the MXFP4 experts to FP8 at load time, doubling their memory), which is a kernel project in its own right and out of scope here.

## How this project uses DeepInfer anyway

- **The serving layer is exactly what DeepInfer says is missing.** Its README's "Required for a complete inference service (not included in this release)" lists a KV cache system, VE/Prefill/Decode disaggregation and load balancing across data-parallel replicas. They map to this project's prefix index with KV events, PD pairing and cache-aware routing.
- **The `deepinfer` adapter** (`worker/leviathan_worker/backends/deepinfer_backend.py`) drives `deepinfer.entrypoints.llm.LLM`'s `llm_engine` online: an engine thread runs the `add_request` / `step` / `flush_inflight_reqs` loop (a collective over the data-parallel group, so every rank keeps stepping), copies outputs with `RequestOutput.clone`, maps `SamplingParams(so_args=..., seq_seed=...)`, and reports DSpark statistics from `draft_tok` / `accept_tok`. A fake engine with the same signatures tests it (`worker/tests/test_deepinfer_adapter.py`). DeepInfer publishes no KV cache events, so the worker reports `supports_kv_events=false` and the router uses its shadow index for it; PD is not wired in the open release, so only unified requests are accepted.
- **The real-GPU path is vLLM**: vLLM ≥ 0.31 officially runs the same model, DeepSeek-V4.1-Flash, on H200 (TP4 with Engram CPU offload), so the prompt the gateway renders with recipe is exactly what the model sees ([illustrated 07](illustrated/07-vllm-on-h200.md)).
