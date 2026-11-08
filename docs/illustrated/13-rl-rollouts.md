# Illustrated 13: RL with lev-engine rollouts

> Code: `rl/lev_rl/loop.py` (the loop), `rl/lev_rl/grpo.py` (advantages, loss, TIS), `rl/lev_rl/task.py` (generated arithmetic, exact reward), `LLMEngine.update_weights` and `LLMEngine.score` in `engine/lev_engine/engine.py`
> Tests: `rl/tests/test_rl.py` (the loop end to end on the tiny Qwen3 checkpoint, CPU, float64)
> Real data: `deploy/slurm/rl_h200.sbatch`, job 11717781, one H200 on node d4055 (raw metrics: `bench/results/rl-h200/11717781/{invariant,cublas}/metrics.jsonl`)
> Previous: [12 Two GPUs](12-two-gpus.md)

RL for LLMs is an inference workload with a trainer attached: most of a step is sampling completions, and the gradient uses the log-probabilities the sampler reported. If the sampler and the trainer compute different numbers for the same tokens, the update is not exactly on-policy. Batch invariance (chapter 10) is one way the two can differ. This chapter measures how much it accounts for.

## The loop

```
             ┌──── 16 prompts × 8 samples, T = 1, seeded ────┐
             ▼                                                │
  lev-engine (bf16, Triton) ──tokens, logp_rollout──▶ reward: 1 if <answer>N</answer> is right
             ▲                                                │ group-relative advantages
             │ update_weights: 310 tensors, device copy       ▼
  trainer (transformers, fp32 weights, bf16 autocast) ◀── GRPO loss with TIS ◀── logp_old (trainer, no grad)
```

Both live in one process on one GPU ("colocated"); weight sync is a device-to-device copy into the engine's parameters, after which its prefix cache is dropped (every cached block was computed with the old weights). The task is generated arithmetic (`Compute 59 - 63 * 15. Think briefly, then give the final integer as <answer>N</answer>.`), so the reward is exact and no dataset is downloaded.

Each step records two mismatches on the sampled tokens:

- **trainer vs rollout**: |logp_old − logp_rollout|, the trainer's log-probability against the engine's, for the same tokens. This is what the truncated importance weight `tis = min(exp(logp_old − logp_rollout), 2)` corrects.
- **rescore vs rollout**: the engine scores its own samples again, each sequence alone as one prefill (`LLMEngine.score`), and compares with what it reported while decoding them in a batch of 128.

## Measured

Qwen3-1.7B, 30 steps, lr 1e-6, two runs that differ only in the engine's GEMMs:

| | batch-invariant | cuBLAS GEMMs |
| --- | --- | --- |
| reward, steps 0–4 → 25–29 | 0.82 → 0.91 | 0.82 → 0.89 |
| completion tokens, steps 0–4 → 25–29 | 64.8 → 54.7 | 65.9 → 50.5 |
| **rescore vs rollout: bit-equal tokens** | **100 % in every step** | 1.7–6.7 % (mean 4.4 %) |
| rescore vs rollout: max abs diff | **0.0** | up to 2.74 nats |
| trainer vs rollout: mean abs diff | 0.0059 | 0.0061 |
| trainer vs rollout: max abs diff | up to 2.27 | up to 1.44 |
| TIS weight, per-step mean | 0.9993–1.0010 | 0.9987–1.0007 |
| step time (median) | 5.96 s | 6.25 s |
| of which rollout (128 completions) | 1.93 s | 1.65 s |
| weight sync | 7.9 ms | 7.9 ms |

Both runs learn: reward rises from 0.82 to about 0.9 in 30 steps, and the completions get shorter. Thirty steps cannot separate the two runs' reward curves, and nothing here claims they differ.

## What the numbers say

**The engine agrees with itself only with invariant kernels.** With cuBLAS, a token's log-probability depends on whether it was computed in a decode batch of 128 or alone in a prefill: 96 % of the tokens get a different value, by up to 2.7 nats. With the invariant kernels every one of the ~200 000 scored tokens is equal bit for bit, in every step. That is what makes a rollout reproducible and checkable afterwards: rescoring is a test, not an estimate.

**But the trainer-rollout gap did not move: 0.0059 vs 0.0061.** The trainer is transformers in fp32 with bf16 autocast, running PyTorch's SDPA and cuBLAS. It is a different implementation from the engine's bf16 Triton kernels, so it rounds differently whether or not the engine is batch-invariant. The batch-dependent noise from cuBLAS (rescore column) is real, but it is small next to the difference between the two implementations, so it does not show in the trainer comparison. Invariance on the inference side alone does not make the update on-policy; the trainer would have to compute its forward pass with the same kernels too, which needs backward passes for them.

**TIS has almost nothing to correct here.** The mean importance weight stays within 0.001 of 1. The mismatch is a few very different tokens (max 1.4–2.3 nats) in a sea of near-equal ones (mean 0.006). In this short run, the cap of 2 only matters for those few.

**Rollout is a third of the step.** 128 completions of ~55 tokens take 1.7–1.9 s on the eager engine. The rest is the trainer's two passes (logp_old without grad, then the update with gradient checkpointing). The first step is slower (47 s with invariant kernels) because the Triton kernels are compiled on first use.

## One bug the run found

The first submission (job 11715386) failed in its first engine step: `ValueError: too many dimensions 'str'`. In transformers 5, `apply_chat_template(..., tokenize=True)` returns a `BatchEncoding`, a mapping that is not a `dict`, and `list()` of a mapping gives its keys. The engine was handed the prompt `["input_ids", "attention_mask"]`. The CPU tests use a tokenizer without a chat template, so they never took that branch. `render` now reads `input_ids` from any mapping and checks that every token id is an integer, and a test covers the mapping case. Before resubmitting, the fix was checked with the real Qwen3 tokenizer in a 10-minute CPU job, not on a GPU allocation.

## Why it is built this way

**Why colocate trainer and engine on one GPU?** At 1.7 B parameters both fit, and colocation turns weight sync into a device copy: 8 ms for 310 tensors. Disaggregated RL (trainer and samplers on different GPUs) pays for broadcasting weights over NCCL every step; chapter 12's collective numbers are what that would cost.

**Why the engine's own sampler, not vLLM?** The rescore check needs an engine that can recompute a logprob exactly, and lev-engine's counter-based sampling and invariant kernels make that a bit-for-bit comparison. vLLM is non-deterministic between its own cached and uncached runs (chapter 10).

**Why generated arithmetic?** It needs no download, the difficulty is a parameter, and the reward is an exact parse with no judge model, so a reward change is a policy change and not grader noise.
