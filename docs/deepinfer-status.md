# DeepInfer 在本项目中的状态

**结论：`deepinfer` 后端按 DeepInfer 源码接口编写并有接口测试，但没有在真实硬件上运行过。** 本项目能拿到的 GPU 是 Northeastern Explorer 集群的 H200 / H100 / A100（均为 SM90 或更早），而 DeepInfer 开源版只支持 NVIDIA SM100（B200/GB200）和昇腾 950。

## 证据（DeepInfer `main`，2026-11-01 检查）

1. **README** 的 Supported hardware 只列出 `NVIDIA SM100 GPUs` 和 `Huawei Ascend 950 NPUs`。
2. **引擎初始化即失败**：`deepinfer/engine/llm_engine.py:304` 调用 `get_bf16_tflops_limit()`；该函数在 `deep_cluster/hardware/device.py` 中只认 `is_sm_major_10()`（计算能力主版本号为 10）和昇腾，其余一律 `raise RuntimeError("unknown GPU type")`。
3. **核心 kernel 依赖 FP4 tensor core**（Hopper 没有这一硬件）：
   - MoE：`deepinfer/model_executor/models/v4_modules/mlp.py:119` 的 `deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous`，以及 `mlp.py:255` 的 `deep_gemm.fp8_fp4_mega_moe`；
   - 稀疏索引注意力：`deepinfer/model_executor/models/v4_modules/index_attn.py:299` 起的 `fp8_fp4_mqa_logits` / `fp8_fp4_paged_mqa_logits` 等。

第 2 条容易绕过，第 3 条意味着要为 Hopper 重写 MoE 和索引注意力 kernel（或在加载时把 MXFP4 专家反量化为 FP8，专家权重显存翻倍），这本身就是一个独立的 kernel 项目，不在本项目范围内。

## 本项目如何对待 DeepInfer

- **服务层补的正是 DeepInfer 声明缺失的部分**：README 的 “Required for a complete inference service (not included in this release)” 列出 KV cache 系统、VE/Prefill/Decode 分离、DP 副本负载均衡，分别对应本项目的前缀索引 + KV 事件、PD 配对调度、cache-aware 路由。
- **`deepinfer` 适配器**：对照 `deepinfer.entrypoints.llm.LLM`、`llm.llm_engine`、`deepinfer.sampling_params.SamplingParams(so_args=..., seq_seed=...)` 和 `RequestOutput`（`token_id`、`token_logprobs`、`draft_tok`、`accept_tok`、`finish_reason`）编写，用签名相同的 fake engine 做接口测试。
- **`deepinfer-sm100` 模拟 profile**：sim 后端可以加载一份按 DeepInfer README 公开的 prefill/decode Pareto 图拟合的参数，用于“换成 SM100 + DeepInfer 会怎样”的推演，结果会明确标注数据来源。
- **真实 GPU 路径用 vLLM**：vLLM ≥ 0.31 官方支持在 H200 上运行同一个模型 DeepSeek-V4.1-Flash（TP4 + Engram CPU offload），所以网关用 recipe 渲染出来的 prompt 就是模型真正看到的输入。
