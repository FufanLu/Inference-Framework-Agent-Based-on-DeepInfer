"""Synthetic model output that the gateway's deepseek-recipe parser accepts.

The synthesizer writes the text a DeepSeek-V4.1 model would produce for the
request, then tokenizes it with the V4.1 tokenizer. Tokenizing the whole text
at once matters: the parser recognizes ``</think>`` and ``｜DSML｜`` only when
they arrive as their special-token IDs, which the tokenizer produces for any
text that spells them.

Shapes, following ``deepseek-recipe-encoding/src/v4/dsv41.rs``:

- thinking mode (the prompt ends with ``<think>``): ``reasoning</think>answer``;
- otherwise (the prompt ends with ``</think>``): ``answer``;
- tool call (tool choice auto)::

      \\n\\n<｜DSML｜ calls>\\n<｜DSML｜ invoke name="NAME">\\n
      <｜DSML｜ parameter name="k" string="true">v</｜DSML｜ parameter>\\n
      </｜DSML｜ invoke>\\n</｜DSML｜ calls>

- tool call required (the prompt already ends with ``\\n\\n<｜DSML｜ calls>\\n``):
  the invocations and ``\\n</｜DSML｜ calls>``, without reasoning.
"""

from __future__ import annotations

import json
import math
import random
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import xxhash

from ..engine import ConstraintKind, OutputConstraint
from ..tokens import DSML, THINK_END, V41Tokenizer
from . import regex_gen, schema_gen
from .corpus import AFTER_TOOL, ANSWER, LANGUAGES, REASONING

TOOL_CALLS_OPEN = f"\n\n<{DSML} calls>\n"
TOOL_CALLS_CLOSE = f"\n</{DSML} calls>"

#: Share of the sampled length spent on reasoning in thinking mode.
REASONING_SHARE = 0.6
#: Spread (sigma of the underlying normal) of the lognormal output length.
LENGTH_SIGMA = 0.5
EFFORT_SCALE = {"low": 0.5, "medium": 0.8, "high": 1.0, "xhigh": 1.25, "max": 1.6}
#: Prompt tokens decoded to pick the answer language.
LANGUAGE_WINDOW = 256

_MASK64 = (1 << 64) - 1


class SynthesisError(ValueError):
    """The constraint cannot be satisfied (bad schema, regex or tool list)."""


@dataclass(frozen=True)
class Synthesized:
    text: str
    token_ids: tuple[int, ...]


def request_seed(
    prompt: Sequence[int], seed: int, constraint: OutputConstraint | None, hints: Mapping[str, str]
) -> int:
    """A 64-bit seed that depends only on the request's inputs, never on timing or batching."""
    base = xxhash.xxh3_64_intdigest(struct.pack(f"<{len(prompt)}I", *prompt), seed=seed & _MASK64)
    shape = json.dumps(
        [int(constraint.kind) if constraint else 0, constraint.schema if constraint else "", sorted(hints.items())],
        ensure_ascii=False,
    )
    return xxhash.xxh3_64_intdigest(shape.encode(), seed=base)


def derive_seed(base: int, label: str) -> int:
    """An independent seed for one purpose (content, speculation, logprobs, ...)."""
    return xxhash.xxh3_64_intdigest(label.encode(), seed=base)


class OutputSynthesizer:
    def __init__(self, tokenizer: V41Tokenizer, *, mean_output_tokens: int) -> None:
        if mean_output_tokens <= 0:
            raise ValueError("mean_output_tokens must be positive")
        self._tok = tokenizer
        self._mean = mean_output_tokens
        sentences = sorted({s for table in (ANSWER, REASONING, AFTER_TOOL) for lang in LANGUAGES for s in table[lang]})
        self._sentence_tokens = dict(zip(sentences, map(len, tokenizer.encode_batch(sentences))))

    def synthesize(
        self,
        prompt: Sequence[int],
        rng: random.Random,
        constraint: OutputConstraint | None,
        hints: Mapping[str, str],
        *,
        min_tokens: int = 0,
    ) -> Synthesized:
        """The output for one request; ``min_tokens`` pads it with more prose (for ignore_eos)."""
        lang = self.language(prompt)
        kind = constraint.kind if constraint else ConstraintKind.UNSPECIFIED
        length = self._sample_length(rng)
        if kind is ConstraintKind.TOOL_CALL_REQUIRED:
            tool = _pick_tool(constraint.schema if constraint else "")
            text = self._invocation(tool, rng, lang) + TOOL_CALLS_CLOSE
        else:
            text = ""
            thinking = self._thinking(prompt, hints)
            if thinking:
                effort = EFFORT_SCALE.get(hints.get("effort", "").lower(), 1.0)
                budget = max(8, int(length * REASONING_SHARE * effort))
                text = self._prose(REASONING[lang], budget, rng, lang) + THINK_END
                length = max(16, int(length * (1 - REASONING_SHARE)))
            text += self._answer(kind, constraint, hints, length, rng, lang)
        ids = self._tok.encode(text)
        while len(ids) < min_tokens:
            more = "\n\n" + self._prose(ANSWER[lang], min_tokens - len(ids), rng, lang)
            text += more
            ids = self._tok.encode(text)
        return Synthesized(text=text, token_ids=tuple(ids))

    # ------------------------------------------------------------------ decisions

    def language(self, prompt: Sequence[int]) -> str:
        """Answer in Chinese when the end of the prompt (the latest turn) is mostly Chinese."""
        tail = self._tok.decode(prompt[-LANGUAGE_WINDOW:], skip_special_tokens=True)
        cjk = sum(1 for ch in tail if "一" <= ch <= "鿿")
        latin = sum(1 for ch in tail if ch.isascii() and ch.isalpha())
        return "zh" if cjk >= 2 and cjk * 4 >= latin else "en"

    def _thinking(self, prompt: Sequence[int], hints: Mapping[str, str]) -> bool:
        hint = hints.get("thinking")
        if hint is not None:
            return hint.strip().lower() in ("true", "1", "yes")
        return bool(prompt) and prompt[-1] == self._tok.think_start_id

    def _sample_length(self, rng: random.Random) -> int:
        """Output tokens drawn from a lognormal whose mean is ``mean_output_tokens``."""
        mu = math.log(self._mean) - LENGTH_SIGMA**2 / 2
        drawn = int(round(rng.lognormvariate(mu, LENGTH_SIGMA)))
        return min(max(drawn, 8), self._mean * 6)

    # ------------------------------------------------------------------ content

    def _answer(
        self,
        kind: ConstraintKind,
        constraint: OutputConstraint | None,
        hints: Mapping[str, str],
        length: int,
        rng: random.Random,
        lang: str,
    ) -> str:
        schema_text = constraint.schema if constraint else ""
        if kind is ConstraintKind.JSON:
            try:
                schema = json.loads(schema_text) if schema_text.strip() else {}
            except json.JSONDecodeError as exc:
                raise SynthesisError(f"json constraint schema is not valid JSON: {exc}") from None
            try:
                value = schema_gen.generate(schema, rng, lang=lang)
            except schema_gen.SchemaError as exc:
                raise SynthesisError(f"json constraint: {exc}") from None
            return json.dumps(value, ensure_ascii=False, indent=2)
        if kind is ConstraintKind.REGEX:
            try:
                return regex_gen.generate(schema_text, rng)
            except regex_gen.RegexError as exc:
                raise SynthesisError(f"regex constraint: {exc}") from None
        if kind is ConstraintKind.TOOL_CALL:
            tool = _pick_tool(schema_text)
            if hints.get("last_role") != "tool":
                return TOOL_CALLS_OPEN + self._invocation(tool, rng, lang) + TOOL_CALLS_CLOSE
            opener = rng.choice(AFTER_TOOL[lang])
            return opener + ("\n\n" + self._prose(ANSWER[lang], length, rng, lang) if length > 24 else "")
        return self._prose(ANSWER[lang], length, rng, lang)

    def _prose(self, sentences: Sequence[str], budget: int, rng: random.Random, lang: str) -> str:
        """Paragraphs of 2-4 sentences, about ``budget`` tokens in total."""
        sep = " " if lang == "en" else ""
        paragraphs: list[str] = []
        used = 0
        previous = None
        while used < budget:
            size = rng.randint(2, 4)
            paragraph: list[str] = []
            while len(paragraph) < size and used < budget:
                sentence = rng.choice(sentences)
                if sentence == previous:
                    sentence = sentences[(sentences.index(sentence) + 1) % len(sentences)]
                paragraph.append(sentence)
                previous = sentence
                used += self._sentence_tokens.get(sentence) or len(self._tok.encode(sentence))
            paragraphs.append(sep.join(paragraph))
            used += 1  # the paragraph break
        return "\n\n".join(paragraphs)

    def _invocation(self, tool: Mapping[str, Any], rng: random.Random, lang: str) -> str:
        function = tool["function"]
        try:
            arguments = schema_gen.generate_arguments(function.get("parameters"), rng, lang=lang)
        except schema_gen.SchemaError as exc:
            raise SynthesisError(f"tool {function['name']!r} parameters: {exc}") from None
        return render_invocation(function["name"], arguments)


def render_invocation(name: str, arguments: Mapping[str, Any]) -> str:
    """One DSML ``invoke`` block, as dsv41.rs ``render_tool_calls`` writes it."""
    params = []
    for key, value in arguments.items():
        if isinstance(value, str):
            params.append(f'<{DSML} parameter name="{key}" string="true">{value}</{DSML} parameter>')
        else:
            encoded = json.dumps(value, ensure_ascii=False)  # Python separators, as recipe's stringify_python_style
            params.append(f'<{DSML} parameter name="{key}" string="false">{encoded}</{DSML} parameter>')
    return f'<{DSML} invoke name="{name}">\n' + "\n".join(params) + f"\n</{DSML} invoke>"


def _pick_tool(schema: str) -> Mapping[str, Any]:
    """The first function tool that is not deferred (else the first tool)."""
    try:
        tools = json.loads(schema) if schema.strip() else None
    except json.JSONDecodeError as exc:
        raise SynthesisError(f"tool_call constraint schema is not valid JSON: {exc}") from None
    if not isinstance(tools, list) or not tools:
        raise SynthesisError("tool_call constraint schema must be a non-empty JSON array of tools")
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict) or not isinstance(function.get("name"), str) or not function["name"]:
            raise SynthesisError('every tool must look like {"type": "function", "function": {"name": ...}}')
    return next((t for t in tools if not t.get("defer_loading")), tools[0])
