#!/usr/bin/env python3
"""Smoke-test a Leviathan gateway with the official OpenAI and Anthropic SDKs.

    python scripts/smoke.py --base-url http://127.0.0.1:8080 [--model deepseek-flash]

Every check prints PASS or FAIL; the exit code is the number of failures.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections.abc import Callable

import anthropic
import openai

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather of a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

CHECKS: list[tuple[str, Callable[[argparse.Namespace], str]]] = []


def check(name: str):
    def register(fn):
        CHECKS.append((name, fn))
        return fn

    return register


def oa(args) -> openai.OpenAI:
    return openai.OpenAI(base_url=f"{args.base_url}/v1", api_key=args.api_key)


def an(args) -> anthropic.Anthropic:
    return anthropic.Anthropic(base_url=args.base_url, api_key=args.api_key)


def extra(obj, field: str):
    value = getattr(obj, field, None)
    if value is None and getattr(obj, "model_extra", None):
        value = obj.model_extra.get(field)
    return value


@check("openai chat: non-streaming")
def chat_basic(args):
    resp = oa(args).chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": "用一句话介绍你自己"}],
        extra_body={"thinking": {"type": "disabled"}},
        max_tokens=64,
    )
    text = resp.choices[0].message.content
    assert text, "empty content"
    assert resp.usage.prompt_tokens > 0, "missing usage"
    return f"{resp.usage.prompt_tokens} prompt / {resp.usage.completion_tokens} completion tokens: {text[:60]!r}"


@check("openai chat: streaming + usage")
def chat_stream(args):
    stream = oa(args).chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": "Say hello"}],
        extra_body={"thinking": {"type": "disabled"}},
        max_tokens=64,
        stream=True,
        stream_options={"include_usage": True},
    )
    parts, usage, first = [], None, None
    start = time.perf_counter()
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            first = first or time.perf_counter()
            parts.append(chunk.choices[0].delta.content)
        if chunk.usage:
            usage = chunk.usage
    assert parts, "no content chunks"
    assert usage is not None, "no usage chunk"
    return f"{len(parts)} chunks, TTFT {(first - start) * 1e3:.0f} ms"


@check("openai chat: thinking mode")
def chat_thinking(args):
    resp = oa(args).chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": "1+1 等于几？"}],
        extra_body={"thinking": {"type": "enabled"}},
        reasoning_effort="low",
        max_tokens=256,
    )
    message = resp.choices[0].message
    reasoning = extra(message, "reasoning_content")
    assert reasoning, "no reasoning_content"
    assert message.content, "no answer after reasoning"
    return f"reasoning {len(reasoning)} chars, answer {message.content[:40]!r}"


@check("openai chat: tool call")
def chat_tools(args):
    resp = oa(args).chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": "北京今天天气怎么样？"}],
        tools=[WEATHER_TOOL],
        extra_body={"thinking": {"type": "disabled"}},
        max_tokens=256,
    )
    choice = resp.choices[0]
    assert choice.finish_reason == "tool_calls", f"finish_reason={choice.finish_reason}"
    call = choice.message.tool_calls[0]
    arguments = json.loads(call.function.arguments)
    assert call.function.name == "get_weather"
    return f"{call.function.name}({arguments})"


@check("openai chat: json_schema response format")
def chat_json_schema(args):
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}, "temp_c": {"type": "integer"}},
        "required": ["city", "temp_c"],
        "additionalProperties": False,
    }
    resp = oa(args).chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": "Give me a weather report as JSON"}],
        response_format={"type": "json_schema", "json_schema": {"name": "report", "schema": schema}},
        extra_body={"thinking": {"type": "disabled"}},
        max_tokens=128,
    )
    value = json.loads(resp.choices[0].message.content)
    assert set(schema["required"]) <= set(value), f"missing keys in {value}"
    return json.dumps(value, ensure_ascii=False)


@check("openai responses: non-streaming")
def responses_basic(args):
    # effort "none" disables thinking, so the whole budget goes to the answer.
    resp = oa(args).responses.create(
        model=args.model, input="Say hello", max_output_tokens=64, reasoning={"effort": "none"}
    )
    text = resp.output_text
    assert text, f"empty output_text (status={resp.status})"
    return f"status={resp.status} {text[:50]!r}"


@check("openai responses: reasoning items")
def responses_reasoning(args):
    resp = oa(args).responses.create(
        model=args.model, input="Why is the sky blue?", max_output_tokens=2048, reasoning={"effort": "low"}
    )
    kinds = [item.type for item in resp.output]
    assert "reasoning" in kinds and resp.output_text, f"output items {kinds}"
    details = resp.usage.output_tokens_details
    return f"items={kinds} reasoning_tokens={details.reasoning_tokens}"


@check("anthropic messages: non-streaming")
def messages_basic(args):
    resp = an(args).messages.create(
        model=args.model,
        max_tokens=64,
        messages=[{"role": "user", "content": "Say hello"}],
        thinking={"type": "disabled"},
    )
    text = "".join(block.text for block in resp.content if block.type == "text")
    assert text, "no text block"
    return f"stop_reason={resp.stop_reason} {text[:50]!r}"


@check("anthropic messages: streaming")
def messages_stream(args):
    with an(args).messages.stream(
        model=args.model,
        max_tokens=64,
        messages=[{"role": "user", "content": "Say hello"}],
        thinking={"type": "disabled"},
    ) as stream:
        text = "".join(stream.text_stream)
        final = stream.get_final_message()
    assert text, "empty stream"
    return f"{final.usage.input_tokens} input / {final.usage.output_tokens} output tokens"


@check("anthropic messages: tool use")
def messages_tools(args):
    resp = an(args).messages.create(
        model=args.model,
        max_tokens=256,
        messages=[{"role": "user", "content": "What's the weather in Boston?"}],
        tools=[
            {
                "name": "get_weather",
                "description": "Current weather of a city",
                "input_schema": WEATHER_TOOL["function"]["parameters"],
            }
        ],
        thinking={"type": "disabled"},
    )
    blocks = [block for block in resp.content if block.type == "tool_use"]
    assert blocks, f"no tool_use block, stop_reason={resp.stop_reason}"
    return f"{blocks[0].name}({blocks[0].input}) stop_reason={resp.stop_reason}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--api-key", default="sk-leviathan-local")
    parser.add_argument("-k", "--only", help="run checks whose name contains this text")
    args = parser.parse_args()

    failures = 0
    for name, fn in CHECKS:
        if args.only and args.only not in name:
            continue
        try:
            detail = fn(args)
            print(f"PASS  {name}: {detail}")
        except Exception as error:  # noqa: BLE001 - report every failure
            failures += 1
            print(f"FAIL  {name}: {error}")
            if "-v" in sys.argv:
                traceback.print_exc()
    print(f"\n{len(CHECKS) - failures if not args.only else '?'} passed, {failures} failed")
    return failures


if __name__ == "__main__":
    sys.exit(main())
