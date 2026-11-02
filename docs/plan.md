# Leviathan：基于 deepseek-recipe（+ DeepInfer 设计）的企业级 DeepSeek 推理与智能体平台

## 背景

DeepSeek 于 2026 年 10 月开源了两个组件：
- **DeepInfer**（`deepseek-ai/DeepInfer`，Apache-2.0）：生产推理引擎。开源版**只有离线推理**，只支持 DeepSeek-V4.1-Flash，只能跑在 **SM100 / 昇腾 950** 上。README 明确写了要组成完整服务还缺三样：**KV cache 系统、Prefill/Decode 分离、DP 副本负载均衡**。
- **deepseek-recipe**（`deepseek-ai/deepseek-recipe`，MIT，Rust crate + Python 绑定）：协议层。负责三种 API 格式与 Conversation 之间的转换、渲染 V4/V4.1 prompt、挂上 tokenizer 编解码，并通过 `StreamProcessor` 输出 SSE。支持思考模式、工具调用、结构化输出、图片（`DestinationFilter` 防 SSRF）和 logprobs。**推理、HTTP、鉴权要调用方自己提供。**

Leviathan 补的是这两个库之间缺失的服务层，结构是“服务平台 + 一层薄的应用层”。项目用途是简历和面试展示，所以优先保证：一条命令能跑起来的 demo、精致的控制台、在真实 GPU 上测出来的 benchmark、清晰的架构。

**已定：真实 GPU 只用 Explorer H200 + vLLM。DeepInfer 只写适配器，不运行，文档里如实说明。**
- DeepInfer 在 H200（SM90）上跑不了，源码证据：
  - `deepinfer/engine/llm_engine.py:304` 调用 `get_bf16_tflops_limit()`，非 SM100 直接抛 `RuntimeError`。
  - MoE 用 `deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous` / `fp8_fp4_mega_moe`，索引注意力用 `fp8_fp4_*_mqa_logits`，两者都依赖 FP4 tensor core。
- **vLLM ≥ 0.31 官方支持 V4.1-Flash 跑在 H200 上**（recipes.vllm.ai）：每个副本 TP4，配合 Engram CPU offload，每卡约有 38 GiB KV。所以**一台 8×H200 节点 = 2 个 DP 副本**，正好用来测路由和负载均衡。而且模型和 recipe 的 V4.1 编码完全一致，网关渲染出来的 prompt 就是模型真正看到的输入，整条链路都是真的。
- Explorer 情况：`gpu` 分区单作业上限 8h，有 4 个 H200×8 节点（每个 512 GB 内存）。`/scratch/lu.fu` 还有空间，checkpoint 约 511 GB。集群没有 vllm 模块，要用 miniconda 装 `vllm>=0.31`。

## 架构

```
 客户端（OpenAI SDK / Anthropic SDK / curl / 控制台 / Agent Studio）
        │  /v1/chat/completions  /v1/responses  /v1/messages
 ┌──────▼─────────────── lev-gateway（Rust，Axum）────────────────────┐
 │ 鉴权（API key→租户）· 限流（RPM/TPM 令牌桶，Redis）· 配额              │
 │ deepseek-recipe：转换 → V4.1 渲染 → tokenizer 编码 → resolve_max_tokens │
 │ StreamProcessor（挂 tokenizer）→ SSE · 用量计量 · 审计/重放            │
 └──────┬──────────────────────────────────────────────────────────────┘
 ┌──────▼─────────────── lev-router（Rust crate，进程内）───────────────┐
 │ 前缀索引：精确模式（订阅 worker 的 KV 事件）/ 近似模式（路由器侧影子 radix 树 + TTL）│
 │ 策略：round_robin | least_load | p2c | cache_aware（命中 − α·负载）    │
 │ PD 分离：prefill 池 → KV handle → decode 池；会话亲和；故障重路由      │
 └──────┬──────────────────────────────▲──────────────────────────────┘
        │ gRPC Generate（流式，token 进 token 出）  │ KV 事件 + 负载上报
 ┌──────▼────────── leviathan-worker（Python，grpc.aio）──────────────┐
 │ vllm：进程内 AsyncLLM + TokensPrompt，DSpark 投机解码，ZMQ KV 事件      │
 │ sim：标定过的耗时模型 + 前缀缓存 LRU + KV 事件（本地开发 / CI / demo）  │
 │ deepseek-api：官方 API 直通（文本模式，无 GPU 时 demo 用）            │
 │ deepinfer：按源码接口写的适配器 + fake engine 接口测试（不运行，需 SM100）│
 └─────────────────────────────────────────────────────────────────────┘
 控制面 lev-control（sqlx + Postgres）· OTel + Prometheus + Grafana · 控制台（Next.js）· Agent Studio
```

关键设计（也是面试时的讲点）：
- **tokenize 在网关做，worker 只做 token 进 token 出。** 这样前缀索引才精确，用量（含 `reasoning_tokens`）也由 recipe 统一统计。
- **路由器支持两种前缀索引模式。** 精确模式基于 vLLM 的 KV 事件；近似模式在路由器侧根据请求历史维护一棵影子树，用于 V4.1 混合注意力下 KV 事件不可用、或者 worker 是 `deepseek-api` 的情况。benchmark 时两种模式都要测。
- **基线对比用 `vllm-router`**（vLLM 官方路由），不只是跟自己的 round_robin 比。
- **sim 闭环标定。** 先在 H200 上用 vLLM 跑一遍 sweep，拟合出 sim 的耗时参数，即 `h200-vllm` profile，这样本地 sim 能复现真实硬件的行为。另外再提供一个按 DeepInfer 公开 Pareto 曲线拟合的 `deepinfer-sm100` profile，用于“换成 SM100 + DeepInfer 会怎样”的推演，并标明数据来源。
- **重放与 diff。** 审计日志记录 token IDs、采样参数和 seed。sim 能保证逐字复现。在 vLLM 上，如果 `VLLM_BATCH_INVARIANT=1` 支持这个模型就用；不支持就由 diff 视图标出第一个分歧的 token 位置，如实展示。
- **安全**：用 `DestinationFilter` 拦截内网图片 URL；API key 以 argon2 哈希存储；租户之间隔离。

## 仓库结构（新建 `~/leviathan`，git init）

```
leviathan/
  Cargo.toml（workspace）
  crates/
    lev-proto/      tonic + prost；proto/worker.proto（Generate、KvEvents、Health、Stats）
    lev-gateway/    Axum：三种协议路由、鉴权中间件、SSE、metrics、admin API
    lev-router/     radix 索引（精确/近似）、策略、PD 配对、worker 注册表
    lev-control/    sqlx migrations；租户/密钥/配额/用量
    lev-bench/      压测器：多轮对话、共享 system prompt 的 agent 负载；输出 jsonl
  worker/           （Python 3.12）
    leviathan_worker/server.py、backends/{vllm_backend,sim,deepseek_api,deepinfer_backend}.py、kvcache/、profiles/*.json
    tests/
  apps/console/     Next.js + Tailwind + shadcn/ui + Recharts
  apps/agent/       Agent Studio 后端（OpenAI SDK 指向网关；工具注册 + 运行轨迹）
  deploy/compose/   sim 模式一条命令启动：gateway、4 个 sim worker（2P+2D）、pg、redis、prom、grafana、console
  deploy/slurm/     setup_env.sh（conda + vllm>=0.31 + rustup）、download_model.sh、h200_serve.sbatch、h200_bench.sbatch
  deploy/helm/      Helm chart
  dashboards/grafana/*.json
  bench/            plot_pareto.py、fit_sim_profile.py、results/
  docs/             architecture.md、design-decisions.md、benchmarks.md、deepinfer-status.md
```

## 实施阶段

**P0 脚手架**：workspace、proto、CI（cargo test、pytest、pnpm build）、Makefile、compose 骨架。

**P1 协议链路（网关 + sim 端到端打通）**
- 依赖 `deepseek-recipe@0.2` 及 `-encoding`、`-image`、`-core`，以 recipe 仓库的 `server-rs` 为模板。
- 实现三条路由、SSE、`resolve_max_tokens`、把 `OutputConstraint` 转发给 worker、模型别名。
- sim 后端：连续批处理、前缀 LRU、KV 事件，按 seed 确定性输出，生成合法的 reasoning/工具调用片段。
- 验收：OpenAI 和 Anthropic 官方 SDK 的流式、非流式、工具调用、json_schema、思考模式用例全部通过。

**P2 路由器 + KV cache**
- 精确/近似两种索引模式、4 种策略、心跳注册表、PD 配对、会话亲和、故障重路由。
- 测试：前缀匹配、淘汰一致性、proptest 属性测试。

**P3 vLLM 后端 + Explorer 上线**（先读 vLLM 0.31 源码确认 API）
- `vllm_backend`：
  - 参数照 vLLM recipe 的 H200 配置：`--tokenizer-mode deepseek_v41`、TP4、`--kv-cache-dtype fp8`、`--engram-config '{"cpu_offload":true}'`，以及 DSpark 的 `--speculative-config`。
  - 进程内用 `AsyncLLM` 接收 `TokensPrompt`，流式返回 token id 和 logprobs，透传 DSpark 的 draft/accept 统计。
  - 结构化输出映射到 vLLM 的 structured outputs（json/regex/grammar）。
  - 开启 `--kv-events-config`（ZMQ），转换成 worker 的 KvEvents。
  - 先只支持纯文本，图片作为延伸目标。
- Slurm：
  - `download_model.sh`：在传输节点下载 `DeepSeek-V4.1-Flash-1017` 到 `/scratch/lu.fu/hf`。
  - `h200_serve.sbatch`：申请 1 节点 × 8 张 H200，跑 2 个 TP4 worker。网关和 bench 也在同一个作业里运行，数据更干净。控制台 demo 通过 SSH 隧道连回 Mac。
  - 如果内存不够（512 GB 主机内存要容纳 2 份 Engram offload），就关闭 offload，改为纯显存。
  - PD 分离：先尝试单节点 1P(TP4)+1D(TP4)，KV 通过 NIXL 传。它在 H200 上没有官方验证，跑不通就只在 sim 里演示，并写明原因。
- `deepseek_api` 后端：文本直通。
- `deepinfer_backend`：
  - 对照 DeepInfer 源码的 `LLM` / `llm_engine` / `SamplingParams(so_args, seq_seed)` / `RequestOutput`（`token_id`、`token_logprobs`、`draft_tok`、`accept_tok`、`finish_reason`）编写。
  - 用一个签名相同的 fake engine 做接口测试。
  - 在 `docs/deepinfer-status.md` 写明它未在真实硬件上运行，以及原因（附上面的源码证据）。

**P4 企业功能**
- 租户、密钥、模型权限、RPM/TPM 令牌桶、月度配额、用量账本。
- 审计、重放和 diff。
- OTel 链路追踪、Prometheus 指标（TTFT、TPOT、命中率、队列深度、DSpark 接受率）、Grafana 看板。
- 优雅下线、背压（429 + Retry-After）。

**P5 控制台 + Agent Studio（fancy 层）**
- 先加载 `impeccable` / `design-taste-frontend` skill 再做设计。
- 页面：
  - 总览。
  - 集群拓扑：P/D 池、每个 worker 的 KV 热力图、路由决策的实时动画。
  - Playground：三种协议并排对比，带 Prompt Inspector（高亮特殊 token）。
  - 租户 / 用量。
  - 重放 diff。
  - Agent Studio：思考 → 工具调用 → 结果的时间线。

**P6 Benchmark + 文档**
- H200：在同样负载下对比 4 种策略和 `vllm-router`，输出 TTFT p50/p99、吞吐、命中率、DSpark 接受率，画成 Pareto 图。
- 用 H200 数据拟合 `h200-vllm` sim profile，再在本地用 sim 复现同样的趋势。
- 写 `docs/benchmarks.md` 和架构图，最后用 `project-description` skill 生成面试讲义。

## 验证
- `cargo test --workspace`；`pytest worker/tests`（含 deepinfer fake engine 接口测试）；`pnpm -C apps/console build`
- `docker compose -f deploy/compose/docker-compose.yml up`，然后：
  - curl 三条路由
  - 运行 `scripts/smoke.py`（OpenAI 和 Anthropic SDK：工具调用、json_schema、思考模式）
  - 重放一个请求，diff 应完全一致
  - 打开 localhost:3000，拓扑页能看到实时路由
- 本地：`cargo run -p lev-bench -- --policy all --workload agent-shared-prefix`，cache_aware 在 TTFT p99 和命中率上应明显优于 round_robin
- Explorer：`sbatch deploy/slurm/h200_bench.sbatch` 跑完生成 `bench/results/h200/*.jsonl` 和 Pareto 图；同一作业里用 `scripts/smoke.py` 对真实 V4.1 模型跑一遍三协议冒烟

## 风险
- H200 排队长，首次加载约 1h（`VLLM_ENGINE_READY_TIMEOUT_S=3600`），所以用 8h 的 `gpu` 分区，每个作业里把 benchmark 一次跑完。
- V4.1 的混合注意力下，vLLM 的 KV 事件和前缀缓存语义要实测；不可用时退回近似模式。
- 2 个 TP4 副本的 Engram offload 加起来，可能超过 512 GB 主机内存，退路见 P3。
- PD 分离在 H200 上没有官方验证，退路见 P3。
