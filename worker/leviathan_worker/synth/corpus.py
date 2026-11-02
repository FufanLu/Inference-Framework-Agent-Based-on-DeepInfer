"""A small bilingual corpus about LLM inference serving for synthetic output.

The text never spells special tokens or DSML markup and avoids ``<`` and ``>``,
so the gateway's parser can only see markup the synthesizer adds on purpose.
"""

from __future__ import annotations

ANSWER = {
    "en": (
        "Prefix caching lets requests that share a system prompt skip most of their prefill.",
        "Continuous batching admits new sequences at every decode step instead of waiting for the whole batch to drain.",
        "Time to first token is dominated by queueing and prefill, while time per output token tracks the decode batch size.",
        "Chunked prefill splits a long prompt across several steps so that running decodes are not stalled.",
        "Speculative decoding drafts several tokens cheaply and verifies them in one forward pass of the target model.",
        "When the acceptance rate is high, each decode step emits several tokens and the effective latency drops.",
        "A cache-aware router sends a request to the replica that already holds the longest matching prefix.",
        "Routing purely by cache hits can overload one replica, so the score also subtracts a load penalty.",
        "Prefill/decode disaggregation runs the compute-bound prefill and the memory-bound decode on separate pools.",
        "The prefill worker hands its KV cache to the decode worker, which continues generation without recomputing the prompt.",
        "KV blocks that are no longer referenced stay cached until memory pressure evicts them in LRU order.",
        "Offloading evicted blocks to host memory turns a full recompute into a cheaper PCIe transfer.",
        "Block hashes are chained, so a single hash identifies the entire prefix up to and including its block.",
        "Tenant quotas and token buckets keep one noisy client from starving everyone else.",
        "Deterministic sampling with a fixed seed makes it possible to replay a request and diff the output token by token.",
        "Mixture-of-experts models activate only a fraction of their parameters per token, which keeps decode cheap.",
        "Tensor parallelism splits each layer across GPUs, while data parallelism adds independent replicas.",
        "The scheduler has a token budget per step and fills it with decode tokens first, then with prefill chunks.",
        "Throughput rises with batch size until the KV cache runs out of free blocks.",
        "Load reports carry queue depth and KV usage so that the router can avoid replicas that are about to saturate.",
        "Structured output decoding masks the logits so that every generated token keeps the output valid.",
        "Streaming responses send tokens as soon as they are decoded, which makes long answers feel responsive.",
    ),
    "zh": (
        "前缀缓存让共享系统提示词的请求跳过大部分 prefill 计算。",
        "连续批处理在每个 decode 步都可以加入新序列，不必等整批请求结束。",
        "首 token 延迟主要由排队和 prefill 决定，而每个输出 token 的耗时取决于 decode 批大小。",
        "分块 prefill 把长提示词拆到多个步骤里执行，避免阻塞正在 decode 的请求。",
        "投机解码先用很小的代价起草若干 token，再由目标模型一次前向验证。",
        "接受率越高，每个 decode 步输出的 token 越多，实际延迟也越低。",
        "缓存感知路由会把请求发给已经持有最长匹配前缀的副本。",
        "只看缓存命中会把流量集中到一个副本上，所以打分时还要减去负载惩罚。",
        "PD 分离把计算密集的 prefill 和访存密集的 decode 放到不同的资源池。",
        "prefill 节点算完之后把 KV cache 交给 decode 节点，decode 节点不用重新计算提示词。",
        "不再被引用的 KV 块会继续留在缓存里，直到显存紧张时按 LRU 顺序淘汰。",
        "把淘汰的块卸载到主机内存，可以用一次 PCIe 传输代替完整的重算。",
        "块哈希是链式计算的，一个哈希就能代表到该块为止的整段前缀。",
        "租户配额和令牌桶限流可以防止单个客户端挤占其他人的资源。",
        "固定随机种子的确定性采样让请求可以重放，并逐 token 对比输出。",
        "混合专家模型每个 token 只激活一小部分参数，因此 decode 的开销较低。",
        "张量并行把每一层切分到多张 GPU 上，数据并行则增加独立的副本。",
        "调度器每一步有固定的 token 预算，先分给 decode，再分给 prefill 分块。",
        "吞吐会随批大小上升，直到 KV cache 的空闲块用完为止。",
        "负载上报包含队列长度和 KV 使用率，路由器据此避开快要饱和的副本。",
        "结构化输出解码会屏蔽不合法的 logits，保证每个生成的 token 都让输出保持合法。",
        "流式响应在 token 解码后立即发送，长回答也能很快看到内容。",
    ),
}

REASONING = {
    "en": (
        "Let me figure out what the question is really asking.",
        "The user wants to understand how the serving system behaves under load.",
        "First, I should separate the prefill cost from the decode cost.",
        "Prefill scales with the number of uncached prompt tokens, so the cache hit rate matters.",
        "Decode is limited by memory bandwidth, and batching amortizes the weight reads.",
        "I should also consider what happens when the KV cache is nearly full.",
        "If the request shares a prefix with earlier traffic, most of the prompt is already cached.",
        "Speculative decoding changes the picture, because each step can emit more than one token.",
        "Let me double-check that the explanation stays consistent with the numbers.",
        "A concise answer with one concrete example should be enough here.",
        "I will structure the answer from the cause to the practical recommendation.",
        "Wait, the routing policy also affects the cache hit rate, so I should mention it.",
    ),
    "zh": (
        "先弄清楚用户真正想问的是什么。",
        "用户想了解推理服务在高负载下的表现。",
        "首先要把 prefill 的开销和 decode 的开销分开看。",
        "prefill 的耗时和未命中缓存的提示词 token 数成正比，所以命中率很关键。",
        "decode 受显存带宽限制，批处理可以摊薄权重读取的成本。",
        "还要考虑 KV cache 快满的时候会发生什么。",
        "如果请求和之前的流量共享前缀，提示词的大部分已经在缓存里了。",
        "投机解码会改变结论，因为每一步可以输出不止一个 token。",
        "再检查一下解释和数字是否一致。",
        "给出简洁的回答，再配一个具体例子就够了。",
        "回答按照从原因到建议的顺序组织。",
        "等一下，路由策略也会影响命中率，这一点也要提到。",
    ),
}

#: Openers for an answer that follows a tool result.
AFTER_TOOL = {
    "en": (
        "Based on the tool result, here is the answer.",
        "The tool returned the data needed to answer the question.",
        "Using the result of the tool call, the situation is clear.",
    ),
    "zh": (
        "根据工具返回的结果，可以得出结论。",
        "工具已经返回了回答问题所需的数据。",
        "结合工具调用的结果，情况已经很清楚了。",
    ),
}

#: Short terms for names, titles, keywords and enum-like strings.
TERMS = {
    "en": (
        "prefix cache", "continuous batching", "chunked prefill", "speculative decoding", "KV cache",
        "cache-aware routing", "decode pool", "prefill pool", "load balancing", "token budget",
        "tenant quota", "replay", "throughput", "latency",
    ),
    "zh": (
        "前缀缓存", "连续批处理", "分块预填充", "投机解码", "键值缓存", "缓存感知路由", "解码池",
        "预填充池", "负载均衡", "令牌预算", "租户配额", "请求重放", "吞吐", "延迟",
    ),
}

CITIES = {
    "en": ("San Francisco", "Boston", "Seattle", "London", "Tokyo", "Singapore"),
    "zh": ("北京", "上海", "杭州", "深圳", "成都", "南京"),
}

LANGUAGES = ("en", "zh")
