# 图解 Leviathan

用**一个请求的一生**串起整个系统：从客户端发出 `POST /v1/chat/completions`，到最后一个 SSE 帧返回，每一站都配图，并附上**实际运行得到的数据**（真实渲染的 prompt、token ID、块哈希、路由打分），不使用编造的示意值。

| 篇 | 主题 | 一句话 | 状态 |
| --- | --- | --- | --- |
| [01](01-request-lifecycle.md) | 一个请求的一生 | 全链路时序图；`Ready` 是能否重试的分界线；Ticket 记负载 | ✅（sim 实测记录待补） |
| [02](02-recipe-protocol.md) | deepseek-recipe 把请求变成 token | 三种协议渲染成同一串 token，因此共享 KV cache；特殊 token 防注入；输出端的 `</think>` 和 DSML 解析 | ✅ |
| [03](03-prefix-index.md) | 块哈希与前缀索引 | 链式哈希让一个 u64 代表整段前缀；位图索引；精确 / 近似两种来源 | ✅ |
| [04](04-cache-aware-routing.md) | cache-aware 路由 | 把缓存命中和负载统一换算成预计 TTFT；真实拐点约 51 条序列 | ✅ |
| 05 | PD 分离 | prefill 池 → KV 交接 → decode 池 | 待写（P2/P3） |
| 06 | 模拟引擎 | 连续批处理、分层 KV、DSpark 投机解码的模拟，以及用 H200 实测标定 | 待写 |
| 07 | H200 上的 vLLM | DeepSeek-V4.1-Flash 在 Explorer 集群上的部署与 KV 事件翻译 | 待写（P3） |
| 08 | 可观测性与重放 | 一个请求的 trace、指标，以及逐 token 的重放 diff | 待写（P4） |

每篇末尾都有“为什么这样设计”一节，回答最常被追问的设计取舍。
