"""Deterministic fake log probabilities for synthetic tokens."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from ..engine import TokenLogprob

#: The most alternatives a request may ask for (as in the OpenAI API).
MAX_TOP_LOGPROBS = 20
_FIRST_PLAIN_ID = 3  # IDs 0-2 are BOS, EOS and padding


def fake_logprobs(
    token_ids: Sequence[int], rng: random.Random, *, top_n: int, plain_vocab_size: int
) -> tuple[TokenLogprob, ...]:
    """One entry per token, drawn from ``rng`` in token order.

    The sampled token is the most likely one, so ``top_token_ids`` starts with
    it, followed by ``top_n - 1`` distinct alternatives. Each alternative takes
    part of the probability mass the previous ones left and is less likely
    than its predecessor, so the values decrease strictly and sum to at most 1.
    """
    top_n = min(max(top_n, 0), MAX_TOP_LOGPROBS)
    span = plain_vocab_size - _FIRST_PLAIN_ID
    out = []
    for token_id in token_ids:
        p = min(1.0 - min(rng.expovariate(6.0), 0.65), 0.999)  # mostly close to 1
        ids: list[int] = []
        probs: list[float] = []
        if top_n:
            ids.append(token_id)
            probs.append(p)
            remaining = 1.0 - p
            for _ in range(top_n - 1):
                alt_p = min(remaining * (0.3 + 0.4 * rng.random()), probs[-1] * 0.9)
                remaining -= alt_p
                probs.append(alt_p)
                alt = _FIRST_PLAIN_ID + rng.randrange(span)
                while alt in ids:
                    alt = _FIRST_PLAIN_ID + (alt - _FIRST_PLAIN_ID + 1) % span
                ids.append(alt)
        out.append(
            TokenLogprob(
                logprob=math.log(p),
                top_token_ids=tuple(ids),
                top_logprobs=tuple(math.log(q) for q in probs),
            )
        )
    return tuple(out)
