"""A verifiable task for RL: arithmetic with an exact, parseable answer.

Problems are generated from a seed (no dataset download), with a difficulty
knob, and graded by parsing the last integer inside <answer>...</answer>.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

_ANSWER = re.compile(r"<answer>\s*(-?\d+)\s*</answer>")


@dataclass(frozen=True)
class Problem:
    question: str
    answer: int


def make_problems(n: int, seed: int, digits: int = 2, terms: int = 3) -> list[Problem]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        nums = [rng.randrange(10 ** (digits - 1), 10**digits) for _ in range(terms)]
        ops = [rng.choice("+-*") for _ in range(terms - 1)]
        expr = str(nums[0])
        for op, num in zip(ops, nums[1:]):
            expr += f" {op} {num}"
        out.append(
            Problem(
                question=f"Compute {expr}. Think briefly, then give the final integer as <answer>N</answer>.",
                answer=eval(expr),  # noqa: S307 - generated arithmetic only
            )
        )
    return out


def reward(completion: str, problem: Problem) -> float:
    """1 for the right answer in the required format, 0 otherwise."""
    found = _ANSWER.findall(completion)
    return 1.0 if found and int(found[-1]) == problem.answer else 0.0
