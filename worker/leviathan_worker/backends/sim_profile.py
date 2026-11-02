"""Timing and capacity profiles for the simulated backend.

A profile is a JSON file in ``leviathan_worker/profiles/`` (or any path) whose
keys are the fields of :class:`SimProfile`. All keys are required except
``name`` (defaults to the file stem) and ``context_limit`` (defaults to the
GPU tier's token capacity); unknown keys are rejected so typos cannot go
unnoticed.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

PROFILE_DIR = Path(str(resources.files("leviathan_worker") / "profiles"))


@dataclass(frozen=True)
class SimProfile:
    name: str
    #: Where the numbers come from (measured, fitted, placeholder).
    source: str
    notes: str
    #: Fixed cost of a step that runs prefill only.
    prefill_base_us: float
    #: Cost of each uncached prompt token computed in a step.
    prefill_us_per_token: float
    #: Fixed cost of a step that runs at least one decode.
    decode_step_base_us: float
    #: Cost of each decoding sequence in a step.
    decode_step_per_seq_us: float
    max_num_seqs: int
    #: Token budget per step (chunked prefill); a decode uses 1 + spec_num_draft of it.
    max_batched_tokens: int
    block_size: int
    #: GPU tier capacity in blocks.
    total_blocks: int
    #: CPU tier capacity in blocks (0 disables the tier).
    cpu_blocks: int
    #: Cost of loading one block from the CPU tier.
    cpu_load_us_per_block: float
    kv_bytes_per_token: int
    #: KV transfer bandwidth between prefill and decode workers, in gigabits per second.
    kv_transfer_gbps: float
    #: DSpark draft length; 0 disables speculative decoding.
    spec_num_draft: int
    #: Probability that each successive draft token is accepted.
    spec_accept_prob: float
    mean_output_tokens: int
    #: Longest prompt plus output, in tokens; 0 means total_blocks * block_size.
    context_limit: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, name: str = "") -> SimProfile:
        fields = {f.name: f for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - set(fields))
        if unknown:
            raise ValueError(f"unknown profile keys: {', '.join(unknown)}")
        values = {"name": name, **data}
        missing = sorted(f for f, spec in fields.items() if f not in values and spec.default is dataclasses.MISSING)
        if missing:
            raise ValueError(f"missing profile keys: {', '.join(missing)}")
        return cls(**{k: _coerce(fields[k].type, k, v) for k, v in values.items()}).validated()

    @classmethod
    def load(cls, name_or_path: str) -> SimProfile:
        """Load a bundled profile by name (``h200-vllm``) or a JSON file by path."""
        path = Path(name_or_path)
        if path.suffix != ".json":
            path = PROFILE_DIR / f"{name_or_path}.json"
        if not path.is_file():
            raise ValueError(f"no sim profile {name_or_path!r}; bundled profiles: {', '.join(available_profiles())}")
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"{path}: a profile must be a JSON object")
        try:
            return cls.from_dict(data, name=path.stem)
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from None

    def override(self, **changes: Any) -> SimProfile:
        """A copy with ``changes`` applied (values may be strings, as from the command line)."""
        fields = {f.name: f for f in dataclasses.fields(self)}
        unknown = sorted(set(changes) - set(fields))
        if unknown:
            raise ValueError(f"unknown profile keys: {', '.join(unknown)}")
        coerced = {k: _coerce(fields[k].type, k, v) for k, v in changes.items()}
        return dataclasses.replace(self, **coerced).validated()

    def validated(self) -> SimProfile:
        positive = ["max_num_seqs", "max_batched_tokens", "block_size", "total_blocks", "kv_bytes_per_token",
                    "kv_transfer_gbps", "mean_output_tokens"]
        non_negative = ["prefill_base_us", "prefill_us_per_token", "decode_step_base_us", "decode_step_per_seq_us",
                        "cpu_blocks", "cpu_load_us_per_block", "spec_num_draft", "context_limit"]
        for key in positive:
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        for key in non_negative:
            if getattr(self, key) < 0:
                raise ValueError(f"{key} must not be negative")
        if not 0.0 <= self.spec_accept_prob <= 1.0:
            raise ValueError("spec_accept_prob must be within [0, 1]")
        if self.max_batched_tokens <= self.max_num_seqs * (1 + self.spec_num_draft):
            raise ValueError(
                "max_batched_tokens must exceed max_num_seqs * (1 + spec_num_draft), "
                "or a full decode batch would leave no budget for prefill"
            )
        return self

    @property
    def effective_context_limit(self) -> int:
        """The context limit, capped by what the GPU tier can hold."""
        capacity = self.total_blocks * self.block_size
        return min(self.context_limit, capacity) if self.context_limit else capacity


def _coerce(kind: Any, key: str, value: Any) -> Any:
    kind = kind if isinstance(kind, str) else getattr(kind, "__name__", str(kind))
    try:
        if kind == "int":
            if isinstance(value, float) and not value.is_integer():
                raise ValueError
            return int(value)
        if kind == "float":
            return float(value)
        return str(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be {kind}, got {value!r}") from None


def available_profiles() -> list[str]:
    return sorted(p.stem for p in PROFILE_DIR.glob("*.json"))
