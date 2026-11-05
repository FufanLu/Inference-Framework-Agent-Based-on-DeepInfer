# Leviathan, illustrated

One request's life ties the whole system together: from a client's `POST /v1/chat/completions` to the last SSE frame, every stop has a diagram and **data from actual runs** (real rendered prompts, token IDs, block hashes, routing scores, benchmark tables), never made-up example values.

| # | Topic | In one line | Status |
| --- | --- | --- | --- |
| [01](01-request-lifecycle.md) | The life of a request | The full sequence; `Ready` is where retries stop being safe; tickets carry load | ✅ |
| [02](02-recipe-protocol.md) | How deepseek-recipe turns a request into tokens | Three protocols render to the same tokens, so they share KV cache; special tokens resist injection; parsing `</think>` and DSML on the way out | ✅ |
| [03](03-prefix-index.md) | Block hashes and the prefix index | A chained hash lets one u64 stand for a whole prefix; bitmask index; exact vs approximate sources | ✅ |
| [04](04-cache-aware-routing.md) | Cache-aware routing | Hits and load in one unit, estimated TTFT; a bug found with benchmark data | ✅ |
| [05](05-pd-disaggregation.md) | Prefill/decode disaggregation | Measured: it wins TPOT on long unique prompts and loses TTFT; the P:D ratio is the knob | ✅ |
| [06](06-sim-engine.md) | The simulated engine | Same protocol as vLLM, step-level cost model, real tokens for the real parser | ✅ |
| [07](07-vllm-on-h200.md) | vLLM on H200 | Same model, another engine; why vLLM's salted hashes are re-hashed; hybrid attention groups | ✅ (measurements pending) |
| [08](08-observability-and-replay.md) | Observability and replay | One drop settles metrics, audit, usage and live events; stage timelines; token-level replay | ✅ |
| [09](09-bursts-and-kafka.md) | Bursts, Kafka and admission control | A 2000-request burst sent directly pushes interactive TTFT p99 to 38 s; queued in Kafka and admitted by the router's TTFT model, 103 ms; the knobs trace a Pareto curve; two gateways, one killed, no job lost | ✅ |
| [10](10-lev-engine.md) | Leviathan's own engine | Paged MLA latent cache (30.4 KiB/token vs 270 KiB expanded), chunked prefill and preemption traced step by step, a prefix cache in the router's hash, batch-invariant kernels and sampling | ✅ (H200 numbers pending) |
| [11](11-speculative-decoding.md) | Speculative decoding | Drafts decide how many tokens a step confirms, never which: output equals plain decoding bit for bit; n-gram lookup and a draft model sampling with the target's noise (20/24 accepted vs 0/92 greedy at T=1, tiny checkpoints); a prefix-cache bug it exposed | ✅ (H200 numbers pending) |

Every chapter ends with a "Why it is built this way" section answering the design questions people ask most.

The sim numbers use a placeholder H200 profile; relative comparisons are meaningful, absolute values wait for the H200 runs.
