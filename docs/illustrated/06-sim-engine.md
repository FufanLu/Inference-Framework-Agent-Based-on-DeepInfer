# Illustrated 06: The simulated engine

> Code: `worker/leviathan_worker/backends/sim.py`, `kvcache/tiered.py`, `events.py`, `synth/`, `profiles/*.json`
> Tests: `worker/tests/` (158 tests, `make test-py`)
> Previous: [05 PD disaggregation](05-pd-disaggregation.md) · Next: [07 vLLM on H200](07-vllm-on-h200.md)

## Why simulate

DeepSeek-V4.1-Flash is a 552B-parameter MoE model (about 511 GB of weights) that needs four H200s per replica. Developing a router, a gateway and a console against that would mean one GPU job per code change. The sim backend is a worker that **speaks exactly the same gRPC protocol as the vLLM backend** and behaves like an engine (continuous batching, chunked prefill, a tiered prefix cache, KV events, speculative decoding, PD hand-offs), but computes its timing from a cost model instead of running a model.

Everything above the worker (gateway, recipe, router, control plane, console, benchmarks) cannot tell the difference, which is the point: the code paths exercised locally are the production ones.

## The scheduler loop

One asyncio task plays the engine's scheduler, the way vLLM's does:

```
every iteration
  1. plan a step: every decoding sequence, then prefill chunks up to max_batched_tokens (8192),
     then admissions from the FIFO queue if their whole KV budget fits
  2. sleep for the step's modelled duration
  3. apply it: prompt blocks enter the prefix cache as chunks complete;
     finished prefills send Ready and their first token;
     each decoding sequence emits 1 token plus its accepted DSpark drafts
```

Step duration (profile `h200-vllm`):

```
step_us = decode_step_base_us (14000)
        + decode_step_per_seq_us (120) × decoding sequences
        + prefill_us_per_token (30)    × prefill tokens computed in this step
        (a prefill-only step pays prefill_base_us (8000) instead of the decode base)
```

That single formula reproduces the effect PD disaggregation is about ([05](05-pd-disaggregation.md)): prefill tokens sharing a step lengthen it for every decoding sequence, so TPOT rises when long prompts arrive.

## The cache

`TieredPrefixCache` keeps blocks of 64 tokens keyed by the chained hashes of [03](03-prefix-index.md):

| Tier | Size (profile) | Behaviour |
| --- | --- | --- |
| GPU | 20000 blocks (1.28 M tokens) | LRU; blocks used by running requests are pinned |
| CPU | 32768 blocks | evicted GPU blocks are demoted here; a hit costs `cpu_load_us_per_block` (40 µs) and promotes the block |

- Matching walks the chain across both tiers and stops at the first miss. The last prompt token is always recomputed (as in vLLM), so `cached_tokens ≤ prompt_tokens − 1`.
- Admission budgets a request's whole KV lifetime (prompt plus planned output), so the sim never preempts. Slots are claimed lazily, so cached blocks stay until the space is actually needed.
- Every change becomes a KV event (`BlockStored` with its parent, `BlockRemoved`, per tier). The backend-neutral `EventHub` fans them out, keeps a mirror for snapshots and turns counters into load-report rates. The vLLM backend reuses the same hub.

## Speculative decoding (DSpark)

Each decode step drafts `spec_num_draft` (5) tokens. Drafts are accepted one by one with probability `spec_accept_prob` (0.72) until the first rejection, so a step emits `1 + accepted` tokens; on average about 3.07 tokens per step with a 41% draft acceptance rate. Per-request random streams make the counts deterministic, and they reach the gateway as `TokenBatch.draft_tokens` / `accepted_tokens`, the same fields vLLM's DSpark statistics fill.

## Output that the real parser accepts

Synthetic output is generated as text, tokenized with the real V4.1 tokenizer, and streamed as token IDs, so special tokens arrive as their IDs (`</think>` = 128822, `｜DSML｜` = 128825) and recipe's parser does real work. The shape follows the request's hints and constraint:

| Request | Output (real samples, seed 2026) |
| --- | --- |
| thinking | `Decode is limited by memory bandwidth, and batching amortizes the weight reads. …</think>Streaming responses send tokens as soon as they are decoded, …` |
| `json_schema` | `{"answer": "Time to first token is dominated by queueing and prefill, …", "confidence": 0.29, "sources": ["https://example.com/chunked-prefill"]}` |
| tool call (auto) | `\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="get_weather">\n<｜DSML｜ parameter name="city" string="true">深圳</｜DSML｜ parameter>\n…</｜DSML｜ calls>` |
| tool call (required) | starts at `<｜DSML｜ invoke …`, because the prompt already ends inside the opened block |
| regex | a string generated from the pattern's parse tree that full-matches it |

The output depends only on (prompt tokens, seed, constraint, hints), never on what else is running, so a replay is token-for-token identical (test `replay_reproduces_the_output`, and the console's replay diff).

## Measured behaviour (profile `h200-vllm`)

| Measurement | Value |
| --- | --- |
| 1809-token prompt, cold | TTFT 70 ms |
| same prompt, 1792 tokens cached | TTFT 13 ms |
| single-stream decode | about 182 tokens/s |

## Honest limits

- **The profile is a placeholder.** `h200-vllm.json` says so in its `source` field. The plan is to fit it to an Explorer H200 sweep (prefill cost per token, decode step cost against batch size, DSpark acceptance) so that the sim reproduces the real hardware's curves locally. Until then, only relative comparisons between policies or layouts are meaningful.
- Scheduling is strict FIFO and `priority` is ignored; there is no preemption.
- KV transfers and CPU loads delay only their own request, not the shared step (as with vLLM's asynchronous KV connectors).

## Why it is built this way

- **Why the real tokenizer and real special tokens?** A simulator that emitted plain text would leave the most fragile part of the system, recipe's token-level output parsing, untested. With real token IDs, the sim exercises the same parser paths as vLLM.
- **Why a step-level cost model instead of per-request latencies?** Interference between requests (prefill stretching decode steps, batch size raising TPOT) only appears when requests share steps. A per-request latency model cannot show why PD disaggregation or cache-aware routing helps.
- **Why deterministic output?** Benchmarks need identical request sequences under every policy, and replay needs a reference to diff against.
