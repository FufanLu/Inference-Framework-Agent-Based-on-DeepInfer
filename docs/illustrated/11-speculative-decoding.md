# Illustrated 11: Speculative decoding that changes nothing but speed

> Code: `engine/lev_engine/speculative.py` (`verify`, `NgramProposer`, `DraftProposer`), `scheduler.py` (lookahead slots), `engine.py` (`_draft`, `_resolve`), `cuda_graph.py` (verify graphs)
> Tests: `engine/tests/test_speculative.py`, `test_triton_kernels.py::test_speculation_is_bit_for_bit_plain_decoding_on_invariant_kernels`, `test_cuda_graph.py` (GPU)
> Real data: traces from the tiny test checkpoints (below); H200 measurements from `deploy/slurm/spec_h200.sbatch` (pending, see the end)
> Previous: [10 Leviathan's own engine](10-lev-engine.md)

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

**The router was outside batch invariance.** The bit-for-bit test passed in the local Linux image and failed on some CI runs by ~1e-8 in the logprobs (and passed on others: GitHub's runners are not all the same CPU). Looking for the cause found a real hole: every GEMM went through the fixed-tile kernel except DeepSeekMoE's router, `h.float() @ gate.T`, a bare matmul on BLAS (cuBLAS on the GPU), free to change its reduction order with the batch shape. It now goes through `ops.linear`. Per-kernel invariance tests cannot see an op that is not a kernel, so there is also a whole-model test (a sequence's logits alone vs in a batch, prefill and decode) that names the first op whose rows differ when it fails; whether the router was the only cause on CI is what that test is there to answer. The same change makes every Triton kernel multiply fp32 inputs in IEEE fp32 instead of `tl.dot`'s default TF32.

**Building a batch waited for the GPU.** `torch.tensor(values, device="cuda")` is a blocking copy, and PyTorch implements blocking host-to-device copies with a stream synchronisation, so every `ForwardBatch.build` waited for all queued kernels. That defeats overlapped scheduling (chapter 10) and would have serialised the draft loop. Batches are now copied with `non_blocking=True` from pageable memory, which the CUDA driver stages without waiting on the device. What this changes on the H200 is measured, not assumed: the first H200 job ran before the change.

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

Pending: `deploy/slurm/spec_h200.sbatch` runs n-gram lookup on DeepSeek-V2-Lite-Chat and Qwen3-8B with a Qwen3-0.6B draft (coupled and greedy) on two real-text workloads (code edits from this repository; open questions) at batch 1, 8 and 32, temperature 0 and 1, and checks every run against plain decoding on the same kernels. The tables go here when it has run.

## Why it is built this way

**Why not standard rejection sampling, which accepts more?** It makes the output depend on the draft. Leviathan replays requests token for token and trains on its own rollouts; a speedup that changes the tokens would make both approximate again. For n-gram drafts there is nothing to give up, and for a draft model the bound above says what is given up.

**Why can't speculation and overlapped scheduling run together?** Overlap launches step N+1 before step N's tokens are read back. With drafts, step N+1's inputs (and its block usage) depend on how many of step N's drafts were accepted. Engines that combine them speculate on the speculation; `lev-engine` refuses the combination instead of half-supporting it.

**Why let the draft share the target's block tables?** A second allocator would have to agree with the first on every admission, preemption and eviction. Sharing the tables makes the draft cache a function of the target's, so the only state to keep is one counter per sequence, and every mistake in it is a performance bug, never a correctness bug.

**Why n-gram lookup at all, with a draft model available?** It costs no GPU memory and no extra forward passes, and the traffic Leviathan is built for copies a lot: agents re-quote tool output, code edits return most of the file. The H200 runs measure where each wins.
