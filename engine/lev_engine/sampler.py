"""Seeded sampling whose result depends only on the request, never the batch.

Noise comes from a counter-based hash of (seed, output index, vocab index),
not from a stateful generator shared by the batch, and sampling is the
Gumbel-max trick, which draws exactly from softmax(logits / T):

    token = argmax_v( logits_v / T + g_v ),   g_v = -log(-log(u_v))

So a request with a given seed produces the same tokens whether it ran alone,
next to 255 others, before or after a preemption, or on replay.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .sequence import SamplingParams

_MASK64 = (1 << 64) - 1


def _to_signed(x: int) -> int:
    return x - (1 << 64) if x >= 1 << 63 else x


_GOLDEN = _to_signed(0x9E3779B97F4A7C15)
_M1 = _to_signed(0xBF58476D1CE4E5B9)
_M2 = _to_signed(0x94D049BB133111EB)


def _shr(x: torch.Tensor, n: int) -> torch.Tensor:
    """Logical right shift of int64 (torch's >> is arithmetic)."""
    return (x >> n) & ((1 << (64 - n)) - 1)


def splitmix64(x: torch.Tensor) -> torch.Tensor:
    """SplitMix64 finaliser on int64 tensors (wrapping arithmetic)."""
    x = x + _GOLDEN
    x = (x ^ _shr(x, 30)) * _M1
    x = (x ^ _shr(x, 27)) * _M2
    return x ^ _shr(x, 31)


def uniform_noise(seeds: torch.Tensor, steps: torch.Tensor, vocab: int) -> torch.Tensor:
    """u in (0, 1), shape [S, vocab], float64; a pure function of (seed, step, index)."""
    key = splitmix64(splitmix64(seeds) ^ steps)  # [S]
    idx = torch.arange(vocab, device=seeds.device, dtype=torch.int64)
    bits = splitmix64(key[:, None] ^ splitmix64(idx)[None, :])
    mantissa = _shr(bits, 11).to(torch.float64)  # 53 random bits
    return (mantissa + 0.5) / float(1 << 53)


@dataclass
class SampleResult:
    tokens: list[int]
    logprobs: list[float] | None


def sample(logits: torch.Tensor, params: list[SamplingParams], steps: list[int]) -> SampleResult:
    """Pick one token per row of `logits` [S, V] (fp32 or fp64)."""
    tokens, logprobs = sample_tensors(logits, params, steps)
    return SampleResult(tokens.tolist(), logprobs.tolist() if logprobs is not None else None)


def sample_tensors(
    logits: torch.Tensor, params: list[SamplingParams], steps: list[int]
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """`sample` without reading the result back: tokens [S] (and logprobs)
    stay on the device, so the next step can be launched before they land."""
    rows, vocab = logits.shape
    device = logits.device
    temps = torch.tensor([p.temperature for p in params], dtype=torch.float64, device=device)
    greedy = temps <= 0
    scaled = logits.to(torch.float64) / torch.where(greedy, 1.0, temps)[:, None]

    top_k = [p.top_k for p in params]
    if any(0 < k < vocab for k in top_k):
        k_max = max(k for k in top_k if k > 0)
        kth = torch.topk(scaled, min(k_max, vocab), dim=-1).values
        ks = torch.tensor([min(k, vocab) if k > 0 else vocab for k in top_k], device=device)
        threshold = kth.gather(1, (ks.clamp(max=kth.shape[1]) - 1)[:, None])
        threshold = torch.where((ks < vocab)[:, None], threshold, torch.full_like(threshold, -torch.inf))
        scaled = scaled.masked_fill(scaled < threshold, -torch.inf)

    top_p = torch.tensor([p.top_p for p in params], dtype=torch.float64, device=device)
    if any(p.top_p < 1.0 for p in params):  # decided on the host: no sync
        sorted_logits, order = scaled.sort(dim=-1, descending=True, stable=True)
        probs = sorted_logits.softmax(dim=-1)
        # Drop a token when the tokens strictly before it already reach top_p.
        drop = (probs.cumsum(dim=-1) - probs) >= top_p[:, None]
        drop[:, 0] = False
        sorted_logits = sorted_logits.masked_fill(drop, -torch.inf)
        scaled = torch.full_like(scaled, -torch.inf).scatter(1, order, sorted_logits)

    seeds = torch.tensor([p.seed for p in params], dtype=torch.int64, device=device)
    step_t = torch.tensor(steps, dtype=torch.int64, device=device)
    gumbel = -torch.log(-torch.log(uniform_noise(seeds, step_t, vocab)))
    noisy = torch.where(greedy[:, None], scaled, scaled + gumbel)
    tokens = noisy.argmax(dim=-1)

    logprobs = None
    if any(p.logprobs for p in params):
        logprobs = scaled.log_softmax(dim=-1).gather(1, tokens[:, None])[:, 0]
    return tokens, logprobs
