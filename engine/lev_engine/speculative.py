"""Speculative decoding: guess k tokens cheaply, check them in one target step.

Verification is exact by construction. The sampler draws output token j of
a request as argmax(logits_j / T + g(seed, j)), with noise g that is a pure
function of (seed, j, vocab index) (see `sampler`). So the token the target
model will pick at each position is already fixed by the context and the
seed. A verify step runs the target on the last token and the k drafts
(k+1 positions, one forward), draws t_0..t_k with that same noise, and keeps

    t_0, t_1, ..., t_a        a = number of leading drafts with d_{i+1} == t_i

which are exactly the tokens decoding one at a time produces: t_i is
conditioned on a prefix that matches the real one. The drafts only decide
how many tokens one step confirms (1 to k+1). A bad draft costs speed, never
correctness, and the output does not depend on the drafter at all: with
batch-invariant kernels, tokens *and* logprobs are bitwise those of plain
decoding (replay and RL rollouts stay reproducible with it on).

What that costs: standard speculative sampling (accept d with probability
min(1, p/q), resample on rejection) accepts with probability 1 - TV(p, q),
the best any scheme can do, but its tokens only match plain decoding in
distribution. Here a deterministic draft (n-grams, a greedy draft model)
loses nothing: P(accept d) = p(d) in both. A sampling draft model uses the
target's noise (same seed, same output index), so where q is close to p,
argmax(log q + g) and argmax(log p + g) coincide: a Gumbel coupling, which
accepts with probability >= (1 - TV) / (1 + TV) (Daliri, Musco, Suresh,
"Coupling without Communication and Drafter-Invariant Speculative
Decoding", arXiv:2408.07978), against p(argmax q) for a greedy draft at
temperature > 0.

Proposers:

* `NgramProposer` (prompt lookup): the continuation of the latest earlier
  occurrence of the sequence's last n tokens. No model; pays off where the
  output copies its context (agents re-quoting tool output, code edits, RAG).
* `DraftProposer`: a small model with the same tokenizer, k autoregressive
  steps, sampling with the coupled noise (or greedy, `draft_coupled=False`).
  Its KV cache is indexed by the target's block tables, so it needs no
  allocator of its own.
"""

from __future__ import annotations

import weakref
from dataclasses import replace

import torch

from .batch import ForwardBatch
from .config import EngineConfig, ModelSpec
from .models import model_class
from .ops import get_ops
from .sampler import sample_tensors
from .scheduler import ScheduledSeq
from .sequence import Sequence
from .weights import load_state_dict


def verify(tokens: list[int], draft: list[int]) -> list[int]:
    """The target's samples t_0..t_k for a row with drafts d_1..d_k: keep
    t_0, and t_{i+1} while d_{i+1} == t_i."""
    out = [tokens[0]]
    for i, d in enumerate(draft):
        if d != out[-1]:
            break
        out.append(tokens[i + 1])
    return out


class _NgramIndex:
    """n-gram -> where the continuation of its latest occurrence starts, for
    the n-grams that end before the last token (so the suffix never finds
    itself). Extended incrementally as the sequence grows."""

    def __init__(self, n_min: int, n_max: int):
        self.maps: dict[int, dict[tuple[int, ...], int]] = {n: {} for n in range(n_max, n_min - 1, -1)}
        self.upto = 0  # n-grams ending at positions < upto are indexed

    def extend(self, tokens: list[int]) -> None:
        last = len(tokens) - 1
        for end in range(self.upto, last):
            for n, m in self.maps.items():
                if end + 1 >= n:
                    m[tuple(tokens[end + 1 - n : end + 1])] = end + 1
        self.upto = max(self.upto, last)

    def lookup(self, tokens: list[int], k: int) -> list[int]:
        for n, m in self.maps.items():  # longest first
            if len(tokens) >= n and (start := m.get(tuple(tokens[-n:]))) is not None:
                return tokens[start : start + k]
        return []


class NgramProposer:
    def __init__(self, config: EngineConfig):
        if not 1 <= config.ngram_min <= config.ngram_max:
            raise ValueError("need 1 <= ngram_min <= ngram_max")
        self.n_min, self.n_max = config.ngram_min, config.ngram_max
        self._index: weakref.WeakKeyDictionary[Sequence, _NgramIndex] = weakref.WeakKeyDictionary()

    def follow(self, scheduled: list[ScheduledSeq]) -> None:
        pass  # nothing to keep in step with the target

    def propose(self, rows: list[ScheduledSeq]) -> list[list[int]]:
        out = []
        for row in rows:
            index = self._index.get(row.seq)
            if index is None:
                index = self._index[row.seq] = _NgramIndex(self.n_min, self.n_max)
            tokens = row.seq.tokens
            index.extend(tokens)
            out.append(index.lookup(tokens, row.lookahead))
        return out


class DraftProposer:
    """A small model run alongside the target. Its cache has the target's
    block layout ([num_blocks, block_size, D_draft] per layer) and is
    addressed by the same block tables; `seq.num_draft_computed` says how
    much of a sequence it holds. It follows the target's prefill chunks
    (`follow`), then drafts k tokens per decode (`propose`)."""

    def __init__(self, config: EngineConfig, num_blocks: int, device: torch.device, dtype: torch.dtype):
        self.config = config
        self.device = device
        self.spec = ModelSpec.from_pretrained(config.draft_model)
        weights = load_state_dict(config.draft_model, dtype, device)
        ops = get_ops(config.kernel_backend, config.batch_invariant)
        self.model = model_class(self.spec)(self.spec, weights, ops, config.max_model_len, device)
        shape = self.model.kv_cache_shape(num_blocks, config.block_size)
        self.kv_caches = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(self.spec.num_layers)]
        self.graphs = None  # DecodeGraphs over the draft model, set by the engine

    @staticmethod
    def cache_bytes_per_token(path: str, dtype: torch.dtype) -> int:
        spec = ModelSpec.from_pretrained(path)
        return spec.num_layers * spec.kv_cache_dim * dtype.itemsize

    def _forward(self, batch: ForwardBatch) -> torch.Tensor:
        if self.graphs is not None and self.graphs.can_run(batch):
            return self.graphs.run(batch)
        return self.model.forward(batch, self.kv_caches)

    def follow(self, scheduled: list[ScheduledSeq]) -> None:
        """Compute the draft KV of what the target computes in this step
        (prefill chunks, plain decodes), from where the draft cache ends."""
        rows = [s for s in scheduled if s.seq.num_draft_computed < s.start + s.num_new]
        if not rows:
            return
        bs = self.config.block_size
        batch = ForwardBatch.build(
            [s.seq.tokens[s.seq.num_draft_computed : s.start + s.num_new] for s in rows],
            [s.seq.num_draft_computed for s in rows],
            [s.seq.block_ids for s in rows],
            bs,
            [True] * len(rows),  # one row each: no empty-logits edge case
            self.device,
        )
        self.model.forward(batch, self.kv_caches)
        for s in rows:
            s.seq.num_draft_computed = s.start + s.num_new

    def propose(self, rows: list[ScheduledSeq]) -> list[list[int]]:
        """k draft tokens per row, from K = max lookahead batched draft steps.

        Step 0 catches up to the last token (usually it alone; two tokens
        when the previous verify accepted every draft, whose last one the
        draft never ran) and samples d_1; step i runs d_i at position n+i
        and samples d_{i+1}. Every row stays in every step, so the batch
        keeps its shape (CUDA-graph friendly) and the drafts stay on the
        device until one read-back at the end. A row past its own lookahead
        L re-writes position n+L, which lies beyond what the draft cache
        is said to hold, and its sample is dropped."""
        seqs = [r.seq for r in rows]
        looks = [r.lookahead for r in rows]
        # The target's sampling settings and noise (or greedy); only the
        # target's logprobs count.
        if self.config.draft_coupled:
            params = [replace(s.params, logprobs=False) for s in seqs]
        else:
            params = [replace(s.params, logprobs=False, temperature=0.0) for s in seqs]
        bs = self.config.block_size
        tables = [s.block_ids for s in seqs]
        last = [s.num_tokens - 1 for s in seqs]  # position n of each last token
        batch = ForwardBatch.build(
            [s.tokens[s.num_draft_computed :] for s in seqs], [s.num_draft_computed for s in seqs], tables, bs,
            [True] * len(seqs), self.device,
        )  # fmt: skip
        tokens, _ = sample_tensors(self._forward(batch), params, [len(s.output) for s in seqs])
        drafts = [tokens]
        for i in range(1, max(looks)):
            positions = [n + min(i, k) for n, k in zip(last, looks, strict=True)]
            batch = ForwardBatch.build([[0]] * len(seqs), positions, tables, bs, [True] * len(seqs), self.device)
            batch.input_ids = drafts[-1]
            tokens, _ = sample_tensors(self._forward(batch), params, [len(s.output) + i for s in seqs])
            drafts.append(tokens)
        table = torch.stack(drafts, dim=1).tolist()  # [S, K]: the one read-back
        for s, n, k in zip(seqs, last, looks, strict=True):
            s.num_draft_computed = n + k  # positions <= n + k - 1 hold draft KV
        return [row[:k] for row, k in zip(table, looks, strict=True)]


def make_proposer(config: EngineConfig, num_blocks: int, device: torch.device, dtype: torch.dtype):
    if config.speculative == "ngram":
        return NgramProposer(config)
    if config.speculative == "draft":
        if not config.draft_model:
            raise ValueError('speculative="draft" needs draft_model')
        return DraftProposer(config, num_blocks, device, dtype)
    raise ValueError(f"unknown speculative method {config.speculative!r} (ngram, draft)")
