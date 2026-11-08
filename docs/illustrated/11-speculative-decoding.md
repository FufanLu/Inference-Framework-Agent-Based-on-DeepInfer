# Illustrated 11: Speculative decoding that changes nothing but speed

> Code: `engine/lev_engine/speculative.py` (`verify`, `NgramProposer`, `DraftProposer`), `scheduler.py` (lookahead slots), `engine.py` (`_draft`, `_resolve`), `cuda_graph.py` (verify graphs)
> Tests: `engine/tests/test_speculative.py`, `test_triton_kernels.py::test_speculation_is_bit_for_bit_plain_decoding_on_invariant_kernels`, `test_cuda_graph.py` (GPU)
> Real data: traces from the tiny test checkpoints (below); H200 measurements from `deploy/slurm/spec_h200.sbatch`, job 11725168 (see the end)
> Previous: [10 Leviathan's own engine](10-lev-engine.md) · Next: [12 Two GPUs](12-two-gpus.md)

A decode step reads every weight and the whole KV context to produce one token per sequence; at small batch the GPU is waiting on memory, not arithmetic, so running 5 tokens per sequence through the same step costs little more than 1. Speculative decoding spends that slack: guess k tokens cheaply, run the target model on all of them in one step, keep the ones the target agrees with.

`lev-engine`'s version has one property the usual one does not: **the output is exactly the output of plain decoding**, token for token, and bit for bit in the logprobs on batch-invariant kernels, whatever proposes the drafts. That keeps [08](08-observability-and-replay.md)'s replay and the RL rollouts reproducible with speculation on.

## One verify step

```
                      last   d1    d2    d3    d4          drafts from a proposer
positions              n    n+1   n+2   n+3   n+4
target logits at       ▼     ▼     ▼     ▼     ▼           one forward, k+1 rows per sequence
target samples        t0    t1    t2    t3    t4          token j: argmax(logits/T + g(seed, j))

keep t0; keep t(i+1) while d(i+1) == t(i):
   d1 == t0 ✓   d2 == t1 ✓   d3 ≠ t2 ✗   →   emit t0 t1 t2   (3 tokens, 1 step)
```

The sampler of [10](10-lev-engine.md) draws output token `j` as `argmax(logits_j / T + g(seed, j))`, with noise that is a pure function of the seed, the output index and the vocab id. So the token plain decoding would pick at position `n+1+i` is already determined once the context is: the verify step computes the same logits for the same context and draws with the same noise. Each `t_i` is kept only while every draft before it matched, i.e. while its context is the real one. The drafts decide how many tokens a step confirms, never which tokens.

That is the whole of `verify`:

```python
def verify(tokens, draft):            # tokens = t0..tk, draft = d1..dk
    out = [tokens[0]]
    for i, d in enumerate(draft):
        if d != out[-1]:
            break
        out.append(tokens[i + 1])
    return out
```

## A trace

A real run on the tiny test checkpoint (random DeepSeek-V2 weights), n-gram prompt lookup with k = 4, greedy, a 17-token prompt, 24 output tokens:

```
step  tokens in  draft (n-gram)            emitted                      accepted
0     17         –                         318                          prefill
1-11  1 each     – (no earlier match)      276 181 124 228 196 244 258 296 52 124 228
12    5          196 244 258 296           99                           0   "124 228" occurred before: guess its continuation; wrong
13-18 1 each     –                         59 293 316 295 59 293
19    5          316 295 59 293            316 295 59 293 268           4   + the target's own next token
```

20 steps instead of 24, and the 24 tokens are those of the plain engine (checked in the same run). Step 12 shows a miss costing nothing but the extra rows; step 19 shows a loop the model fell into being confirmed four tokens at a time. The proposer is an index from every n-gram (n = 4 down to 2) to where the continuation of its latest occurrence starts, extended incrementally as the sequence grows (`_NgramIndex`); a lookup is one dictionary probe per n.

## Draft models and shared noise

The second proposer is a small model with the same tokenizer (Qwen3-0.6B for Qwen3-8B on the H200). It runs k autoregressive steps per decode, and it samples **with the target's noise**: same seed, same output index. Where the draft's distribution q is close to the target's p, `argmax(log q + g)` and `argmax(log p + g)` land on the same token, at any temperature: a Gumbel coupling. Drafting greedily instead (the common choice) only matches a *sampled* target token with probability p(argmax q).

Measured on the tiny checkpoints, temperature 1, k = 4, 24 output tokens:

| Draft for the tiny V2 target | Drafts sampled with the target's noise | Drafted greedily |
| --- | --- | --- |
| the target itself | 20 of 20 accepted, 6 steps | 0 of 92 accepted, 24 steps |
| an unrelated random Qwen3 (TV(p, q) = 0.092 on average) | 20 of 24 accepted, 7 steps | 0 of 92 accepted, 24 steps |

Both tiny models are near-uniform (largest probability 0.0054 and 0.0104 over 320 tokens), so their distributions are close (total variation 0.092) even though their top tokens disagree: p(argmax q) is 0.003. The coupling bound for that TV, (1 − 0.092) / (1 + 0.092) = 0.83, is what the run gave (20 / 24 = 0.83). Every run emitted exactly the plain engine's tokens.

The price of exactness: standard speculative sampling (accept d with probability min(1, p(d)/q(d)), resample from the residual on rejection) achieves acceptance 1 − TV(p, q), the maximum possible, but its tokens match plain decoding only in distribution. The coupling achieves at least (1 − TV) / (1 + TV) (Daliri, Musco and Suresh, *Coupling without Communication and Drafter-Invariant Speculative Decoding*, arXiv:2408.07978). For a deterministic draft (n-grams, a greedy draft model) the two coincide: P(accept d) = p(d) either way.

### One allocator, two caches

```
target cache (27 layers)  [num_blocks, 16, 576]   ┐
                                                   ├─ indexed by the same seq.block_ids
draft cache  (its layers) [num_blocks, 16, D_dft]  ┘

seq.num_computed        tokens whose target KV is cached
seq.num_draft_computed  tokens whose draft KV is cached
```

The draft model's cache has the target's block layout and uses the target's block tables, so it needs no allocator, no eviction policy and no prefix index of its own. `follow()` runs the draft over whatever the target computes in a step (prefill chunks, plain decodes); `propose()` catches up to the last token (usually that token alone; two tokens when the previous verify accepted every draft, because the draft never ran its own last guess) and drafts. Every row stays in every draft step so the batch keeps its shape and the drafts stay on the GPU until one read-back; a row past its own lookahead rewrites a position the draft cache does not claim. After verification `num_draft_computed = min(num_draft_computed, num_computed)`. A cached prefix block's draft KV is trusted on a prefix hit; when it is stale (a block filled by `score()`), the drafts get worse and the output does not change.

## The scheduler's side: lookahead slots

| Rule | Why |
| --- | --- |
| A decode gets `lookahead = min(k, budget − 1, max_model_len − num_tokens, free room)` slots | the step's token budget counts drafts; positions stay inside the RoPE table |
| Slots come from free blocks only | a guess never preempts a real sequence |
| Prefill chunks and recomputation after preemption get none | there is nothing to guess until the prompt is in |
| Overlapped scheduling is refused with speculation | the next step's inputs depend on how many drafts were accepted, which is only known after the read-back |

Nothing is rolled back after a miss. Rejected positions lie beyond `num_computed`; their KV is garbage that the next write to those slots overwrites, and the prefix cache registers full blocks of `num_computed` only, so it never sees a guess. A stop token that was an accepted draft leaves its own KV computed, which can complete one more cacheable block than plain decoding: never fewer (tested).

With CUDA graphs, verify steps replay graphs captured for k+1 tokens per sequence. Short drafts are padded to k with a filler token, which is harmless by construction: a filler the target happens to agree with is as correct an accept as any other. The draft model's single-token steps replay its own decode graphs.

## Found on the way

**Blocks after a prefix hit were never cached.** `num_registered` counts blocks, but admission after a prefix hit set it to `len(hit) * block_size`, a token count. A request that extended a 7-block cached prefix registered nothing until it reached block 28, so the router's exact index (chapter 03) never learned the longer prefix: the follow-up turns of the agent workload would not have found their own history. Speculative runs exposed it (their stored-block sets differed from plain runs after preemption); `test_blocks_after_a_prefix_hit_are_cached_too` failed before the one-line fix and passes after.

**Two holes in batch invariance: one in the model, one in the test rig.** The bit-for-bit test passed in the local Linux image and failed on some CI runs by ~1e-8 in the logprobs, passing on others (GitHub's runners are not all the same CPU). The search found a real hole first: every GEMM went through the fixed-tile kernel except DeepSeekMoE's router, `h.float() @ gate.T`, a bare matmul on BLAS (cuBLAS on the GPU), free to change its reduction order with the batch shape; it now goes through `ops.linear`. The failures continued, so a whole-model test now traces every op for a sequence alone and in a batch and names the first that differs. On an AMD EPYC 9V74 runner it said: routing ids and weights equal, first difference in `moe_routed`, 3.5e-10. Under `TRITON_INTERPRET=1`, `tl.dot` is numpy's matmul on the host BLAS, and on that CPU a row's bits depend on its position in the 16-row expert tile, which moves with the batch (the same check on Explorer's Intel login node: identical for all 63 row shifts). The interpreter now computes dots as an in-order sum over K, and the tests pass on EPYC 9V74, 9V45 and 7763 runners. On the GPU the kernels use `tl.dot` as before, and the H200 job checks them bit for bit on real tensor cores. The same change makes every Triton kernel multiply fp32 inputs in IEEE fp32 instead of `tl.dot`'s default TF32.

**Building a batch waited for the GPU.** `torch.tensor(values, device="cuda")` is a blocking copy, and PyTorch implements blocking host-to-device copies with a stream synchronisation, so every `ForwardBatch.build` waited for all queued kernels. That defeats overlapped scheduling (chapter 10) and would have serialised the draft loop. Batches are now copied with `non_blocking=True` from pageable memory, which the CUDA driver stages without waiting on the device. What this changes on the H200 is measured, not assumed: the first H200 job ran before the change. With the fix, overlapped scheduling still gained nothing measurable on the H200, because scheduling is only 4 % of a run (see the end).

## Correctness

| Test | What it pins down |
| --- | --- |
| `test_ngram_speculation_matches_plain_decoding[256, 22]` | 6 requests (greedy, T = 1.3, top-k, EOS stops, one to the context limit), priorities, a 24-token budget that truncates lookahead; with 22 blocks also preemption: same tokens and finish reasons as plain decoding, fewer steps |
| `test_a_draft_identical_to_the_target_is_always_accepted` | the coupled draft accepts every draft at T = 0 and T = 1.3; greedy drafting does not; outputs equal plain decoding |
| `test_an_unrelated_draft_costs_speed_not_correctness` | a random Qwen3 drafting for the V2 target, with preemption: outputs equal plain decoding |
| `test_speculation_is_bit_for_bit_plain_decoding_on_invariant_kernels` | Triton batch-invariant kernels: n-gram and draft-model runs equal plain decoding in tokens **and** logprobs, bit for bit |
| `test_speculative_verify_graphs_match_plain_eager_decoding` (GPU) | verify graphs with padded drafts and draft-model graphs: still plain decoding's bits |

On the reference kernels the logprobs agree to 1e-9, not bitwise: verify batches have other shapes than decode batches, and the MoE router's fp32 GEMM on the CPU changes its reduction order with the shape.

## Measured on H200

Job 11725168 (`deploy/slurm/spec_h200.sbatch`, one H200 on node d4054, CUDA graphs on; raw JSON: `bench/results/spec-h200/11725168/`). k = 4 drafts per step, 256 output tokens. Two workloads of real text: **edit** (a file from this repository and an instruction to return it complete with one change; 413–954 prompt tokens) and **chat** (open questions; 15–29 prompt tokens). Every run is compared with plain decoding on the same kernels, request by request. The GPU tests (speculation, CUDA graphs, Triton kernels compiled on the H200): 20 passed.

### Exact, on invariant kernels

Batch 8, batch-invariant kernels, output tok/s and speedup over plain decoding:

| model | workload, T | plain | n-gram | draft model (Qwen3-0.6B, coupled) | greedy draft |
| --- | --- | --- | --- | --- | --- |
| DeepSeek-V2-Lite | edit, 0 | 347 | **1 056 (3.0×)**, 88 % accepted | – | – |
| | edit, 1 | 344 | 453 (1.3×), 52 % | – | – |
| | chat, 0 | 341 | 291 (0.85×), 2 % | – | – |
| Qwen3-8B | edit, 0 | 440 | **971 (2.2×)**, 70 % | 551 (1.25×), 90 % | – |
| | edit, 1 | 440 | **1 068 (2.4×)**, 72 % | 581 (1.3×), 90 % | 599 (1.4×), 92 % |
| | chat, 0 | 466 | 443 (0.95×), 2 % | 354 (0.76×), 40 % | – |
| | chat, 1 | 467 | 437 (0.94×), 1 % | 338 (0.72×), 35 % | 335 (0.72×), 36 % |

In all 14 speculative runs on invariant kernels, every request's tokens **and logprobs** are identical to plain decoding, bit for bit, at temperature 0 and 1. That is the claim of this chapter checked on real models and real tensor cores.

### What happens without invariance

The same runs on the fast kernels (cuBLAS GEMMs), share of requests whose tokens equal plain decoding at **temperature 0**:

| model | batch 1 | batch 8 | batch 32 |
| --- | --- | --- | --- |
| DeepSeek-V2-Lite, n-gram (edit / chat) | 100 % / **0 %** | 100 % / 25 % | 62 % / 6 % |
| Qwen3-8B, n-gram / draft (edit) | 100 % / 100 % | 100 % / 88 % | 81 % / 88 % |
| Qwen3-8B, n-gram / draft (chat) | 100 % / 100 % | 38 % / 38 % | 19 % / 19 % |

Greedy decoding with speculation is supposed to return the same text. It does not here, because a verify step computes k + 1 rows per sequence where a decode step computes one, and on shape-dependent kernels a near-tie in the argmax flips. For V2-Lite this happens even for a single chat request. No logprob in any of the V2-Lite runs matched plain decoding exactly.

### Where speculation pays

The fastest path (non-invariant kernels) on Qwen3-8B, edit workload:

| batch | plain | n-gram | draft model |
| --- | --- | --- | --- |
| 1 | 115 tok/s | **502 (4.4×)**, 94 % accepted, 4.7 tokens per verify | 141 (1.2×), 99.5 % accepted |
| 8 | 812 | **1 840 (2.3×)** | 783 (0.96×) |
| 32 | 2 301 | **3 356 (1.5×)** | 2 106 (0.92×) |

- **N-gram lookup wins on copy-heavy output and loses a little on everything else.** On edits at temperature 0 it accepts 54–94 % and gives 1.4–4.4×. On chat, nothing in the prompt predicts the answer (1–4 % accepted): from 5 % faster (Qwen3-8B, batch 1) to 30 % slower (Qwen3-8B, batch 32), as the wasted verify rows start to cost GPU time.
- **The draft model is accepted more and is faster less.** At batch 1 it is accepted 99.5 % of the time on edits, yet gives 1.2× where n-gram gives 4.4×. A verify step took 34 ms: four 0.6B forward passes and one 8B pass. Plain 8B decoding takes 8.7 ms per token, so one 0.6B pass costs at most about 6.4 ms, up to 74 % of an 8B pass, for 7 % of the parameters. At small batch a step's cost is mostly fixed per-step work, not FLOPs (chapter 10's graph numbers below say the same), so a small draft model is not much cheaper than the target.
- **The speedup shrinks with batch.** At batch 32 the GPU has work to do on every step, so verify rows are no longer free: n-gram falls from 4.4× to 1.5×, and the draft model costs more than it saves.

### Shared noise on real models

On the tiny checkpoints, drafting with the target's noise was accepted 20 of 24 times where greedy drafting got 0 of 92. On Qwen3-0.6B drafting for Qwen3-8B at temperature 1 there is no difference: 90 % vs 92 % on edits, 35 % vs 36 % on chat. A greedy draft d is accepted with probability p(d), and real models are confident on most tokens, so p(argmax q) is high. The coupling's advantage needs flat distributions, as in the near-uniform tiny models (largest probability 0.005). The coupled draft is still the one to use: it is never worse here, and it keeps output independent of the draft's sampling.

### CUDA graphs and overlap (chapter 10's open question)

The same throughput workloads as chapter 10 (512 prompt, 256 output tokens, DeepSeek-V2-Lite), now with CUDA graphs:

| engine | 1 request: ms/token | 1 request: tok/s | 256 requests: tok/s |
| --- | --- | --- | --- |
| lev Triton, eager (job 11707247) | 27.9 | 35.8 | 4 666 |
| lev Triton, CUDA graphs | **9.5** | **104** | **5 336** |
| lev Triton, CUDA graphs + overlapped scheduling | 9.5 | 103 | 5 299 |
| lev Triton invariant, eager | 26.9 | 37.2 | 4 168 |
| lev Triton invariant, CUDA graphs | 17.0 | 58.4 | 4 435 |
| vLLM 0.31.0 | – | 318 | 11 709 |

Graphs cut a single request's step from 27.9 to 9.5 ms: two thirds of the eager step was launching kernels from Python, as chapter 10 expected. With that removed, the invariant kernels' own cost shows: 17.0 vs 9.5 ms at batch 1 (fixed tiles, no split-KV), where in eager mode both were hidden under the launch overhead. At batch 256 the GPU was already busy, so graphs add 14 % (6 % invariant). Overlapped scheduling adds nothing measurable: scheduling is 0.48 s of a 12.4 s run (4 %), so there is little to hide. vLLM remains 3.0× faster at batch 1 and 2.2× at batch 256.

## Why it is built this way

**Why not standard rejection sampling, which accepts more?** It makes the output depend on the draft. Leviathan replays requests token for token and trains on its own rollouts; a speedup that changes the tokens would make both approximate again. For n-gram drafts there is nothing to give up, and for a draft model the bound above says what is given up.

**Why can't speculation and overlapped scheduling run together?** Overlap launches step N+1 before step N's tokens are read back. With drafts, step N+1's inputs (and its block usage) depend on how many of step N's drafts were accepted. Engines that combine them speculate on the speculation; `lev-engine` refuses the combination instead of half-supporting it.

**Why let the draft share the target's block tables?** A second allocator would have to agree with the first on every admission, preemption and eviction. Sharing the tables makes the draft cache a function of the target's, so the only state to keep is one counter per sequence, and every mistake in it is a performance bug, never a correctness bug.

**Why n-gram lookup at all, with a draft model available?** It costs no GPU memory and no extra forward passes, and the traffic Leviathan is built for copies a lot: agents re-quote tool output, code edits return most of the file. The H200 run settled it: on edits n-gram lookup is 4.4× at batch 1 against the draft model's 1.2×, because at small batch a 0.6B forward pass costs up to 74 % of an 8B one.
