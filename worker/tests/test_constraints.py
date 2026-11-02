"""Synthetic output must be valid DeepSeek-V4.1 output for deepseek-recipe's parser.

Each test decodes the streamed token IDs with ``skip_special_tokens=False`` and
checks the text, and also checks that markup (``</think>``, ``｜DSML｜``) arrives
as its special-token ID, which is what recipe's StreamProcessor matches.
"""

from __future__ import annotations

import json
import re

import pytest
from support import TOOL_BLOCK, collect, make_request, parse_invocations, prompt_ids, validate

from leviathan_worker.pb import worker_pb2 as pb

JSON = pb.OutputConstraint.KIND_JSON
REGEX = pb.OutputConstraint.KIND_REGEX
TOOL = pb.OutputConstraint.KIND_TOOL_CALL
TOOL_REQUIRED = pb.OutputConstraint.KIND_TOOL_CALL_REQUIRED

WEATHER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather forecast for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    "days": {"type": "integer", "minimum": 1, "maximum": 7},
                    "include_hourly": {"type": "boolean"},
                    "fields": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
                },
                "required": ["city", "unit", "days", "include_hourly", "fields"],
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {"name": "search_docs", "description": "Search.", "parameters": {"type": "object"}, "strict": None},
    },
]


def split_thinking(tokenizer, ids: list[int]) -> tuple[str, str]:
    assert ids.count(tokenizer.think_end_id) == 1, "thinking output has exactly one </think> token"
    cut = ids.index(tokenizer.think_end_id)
    return tokenizer.decode(ids[:cut]), tokenizer.decode(ids[cut + 1 :])


async def test_thinking_output_is_reasoning_then_answer(worker, tokenizer):
    prompt = prompt_ids(tokenizer, "Why is decode memory bound?", thinking=True)
    assert prompt[-1] == tokenizer.think_start_id
    result = await collect(worker.stub, make_request("think", prompt, hints={"thinking": "true"}))
    ids = result.token_ids
    assert tokenizer.think_start_id not in ids
    reasoning, answer = split_thinking(tokenizer, ids)
    assert reasoning.strip() and answer.strip()
    text = tokenizer.decode(ids)
    assert text == f"{reasoning}</think>{answer}"
    assert tokenizer.encode(text) == ids


async def test_non_thinking_output_has_no_thinking_markup(worker, tokenizer):
    prompt = prompt_ids(tokenizer, "Why is decode memory bound?")
    result = await collect(worker.stub, make_request("plain", prompt, hints={"thinking": "false"}))
    ids = result.token_ids
    assert tokenizer.think_end_id not in ids and tokenizer.think_start_id not in ids
    assert tokenizer.dsml_id not in ids


async def test_answer_language_follows_the_prompt(worker, tokenizer):
    zh = await collect(worker.stub, make_request("zh", prompt_ids(tokenizer, "请解释一下前缀缓存的作用。")))
    en = await collect(worker.stub, make_request("en", prompt_ids(tokenizer, "Please explain prefix caching.")))
    assert re.search(r"[一-鿿]", tokenizer.decode(zh.token_ids))
    assert not re.search(r"[一-鿿]", tokenizer.decode(en.token_ids))


SCHEMAS = {
    "nested": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 3},
            "kind": {"enum": ["router", "worker", "gateway"]},
            "replicas": {"type": "integer", "minimum": 1, "maximum": 8},
            "ratio": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1},
            "enabled": {"type": "boolean"},
            "mode": {"type": "string", "default": "auto"},
            "tags": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 4, "uniqueItems": True},
            "limits": {
                "type": "object",
                "properties": {"rpm": {"type": "integer", "multipleOf": 10}, "burst": {"type": ["integer", "null"]}},
                "required": ["rpm", "burst"],
                "additionalProperties": False,
            },
            "owner": {"$ref": "#/$defs/person"},
        },
        "required": ["name", "kind", "replicas", "ratio", "enabled", "mode", "tags", "limits", "owner"],
        "additionalProperties": False,
        "$defs": {
            "person": {
                "type": "object",
                "properties": {"email": {"type": "string", "format": "email"}, "id": {"type": "string", "pattern": "^u-[0-9]{4}$"}},
                "required": ["email", "id"],
            }
        },
    },
    "array": {"type": "array", "items": {"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]}},
    "any-of": {"anyOf": [{"type": "string", "enum": ["yes", "no"]}, {"type": "integer"}]},
}


@pytest.mark.parametrize("thinking", [False, True], ids=["answer", "thinking"])
@pytest.mark.parametrize("name", list(SCHEMAS))
async def test_json_constraint_output_matches_the_schema(worker, tokenizer, name, thinking):
    schema = SCHEMAS[name]
    prompt = prompt_ids(tokenizer, "Describe the deployment as JSON.", thinking=thinking)
    for seed in range(4):
        result = await collect(
            worker.stub,
            make_request(
                f"json-{seed}", prompt, seed=seed, constraint=(JSON, json.dumps(schema)),
                hints={"thinking": str(thinking).lower()},
            ),
        )
        ids = result.token_ids
        if thinking:
            reasoning, answer = split_thinking(tokenizer, ids)
            assert reasoning.strip()
        else:
            answer = tokenizer.decode(ids)
        assert result.finished.reason == pb.FINISH_REASON_STOP
        validate(json.loads(answer), schema)


async def test_json_constraint_with_empty_schema_is_an_object(worker, tokenizer):
    result = await collect(worker.stub, make_request("json-any", prompt_ids(tokenizer, "JSON please"), constraint=(JSON, "")))
    value = json.loads(tokenizer.decode(result.token_ids))
    assert isinstance(value, dict) and value


@pytest.mark.parametrize(
    "pattern",
    [
        r"\d{3}-\d{4}",
        r"(yes|no|maybe)",
        r"[A-Z][a-z]{2,6}( [A-Z][a-z]+)*",
        r"[^\s@]{3,8}@example\.(com|org)",
        r"(?P<word>ab+)-(?P=word)",
        r"0x[0-9a-fA-F]{8}",
        r"(?i)ok|done",
    ],
)
async def test_regex_constraint_output_fullmatches(worker, tokenizer, pattern):
    prompt = prompt_ids(tokenizer, "Answer in the required format.")
    for seed in range(3):
        result = await collect(worker.stub, make_request(f"re-{seed}", prompt, seed=seed, constraint=(REGEX, pattern)))
        assert re.fullmatch(pattern, tokenizer.decode(result.token_ids))


async def test_regex_constraint_in_thinking_mode(worker, tokenizer):
    prompt = prompt_ids(tokenizer, "Pick a code.", thinking=True)
    result = await collect(
        worker.stub, make_request("re-think", prompt, constraint=(REGEX, r"[A-Z]{3}-\d{2}"), hints={"thinking": "true"})
    )
    _, answer = split_thinking(tokenizer, result.token_ids)
    assert re.fullmatch(r"[A-Z]{3}-\d{2}", answer)


def check_tool_block(tokenizer, ids: list[int], *, opened_by_prompt: bool = False) -> list[tuple[str, dict]]:
    text = tokenizer.decode(ids)
    # Every DSML tag spelled in the text must arrive as the special token, or the parser ignores it.
    assert ids.count(tokenizer.dsml_id) == text.count("｜DSML｜")
    assert tokenizer.encode(text) == ids
    block = "\n\n<｜DSML｜ calls>\n" + text if opened_by_prompt else text
    match = TOOL_BLOCK.fullmatch(block)
    assert match, f"not a DSML tool call block: {block!r}"
    return parse_invocations(match.group("invokes"))


async def test_tool_call_uses_the_exact_dsml_format(worker, tokenizer):
    prompt = prompt_ids(tokenizer, "What's the weather in Beijing?")
    result = await collect(
        worker.stub,
        make_request("tool", prompt, constraint=(TOOL, json.dumps(WEATHER_TOOLS)), hints={"last_role": "user"}),
    )
    calls = check_tool_block(tokenizer, result.token_ids)
    assert len(calls) == 1
    name, args = calls[0]
    assert name == "get_weather"
    validate(args, WEATHER_TOOLS[0]["function"]["parameters"])
    text = tokenizer.decode(result.token_ids)
    # Strings are raw text with string="true"; other values are JSON with string="false".
    assert f'name="city" string="true">{args["city"]}<' in text
    assert f'name="days" string="false">{args["days"]}<' in text
    assert f'name="fields" string="false">{json.dumps(args["fields"], ensure_ascii=False)}<' in text
    assert result.finished.reason == pb.FINISH_REASON_STOP


async def test_tool_call_after_reasoning_in_thinking_mode(worker, tokenizer):
    prompt = prompt_ids(tokenizer, "Weather in Shanghai?", thinking=True)
    result = await collect(
        worker.stub,
        make_request(
            "tool-think", prompt, constraint=(TOOL, json.dumps(WEATHER_TOOLS)),
            hints={"thinking": "true", "last_role": "user"},
        ),
    )
    ids = result.token_ids
    cut = ids.index(tokenizer.think_end_id)
    assert tokenizer.decode(ids[:cut]).strip()
    check_tool_block(tokenizer, ids[cut + 1 :])


async def test_tool_result_turn_is_answered_in_text(worker, tokenizer):
    prompt = prompt_ids(tokenizer, "<tool_result>{\"temp\": 21}</tool_result>")
    result = await collect(
        worker.stub,
        make_request("tool-done", prompt, constraint=(TOOL, json.dumps(WEATHER_TOOLS)), hints={"last_role": "tool"}),
    )
    assert tokenizer.dsml_id not in result.token_ids
    assert tokenizer.decode(result.token_ids).strip()


@pytest.mark.parametrize("thinking", [False, True], ids=["answer", "thinking"])
async def test_required_tool_call_continues_the_opened_block(worker, tokenizer, thinking):
    prompt = prompt_ids(tokenizer, "Look it up.", thinking=thinking, required_tool=True)
    assert tokenizer.decode(prompt).endswith("\n\n<｜DSML｜ calls>\n")
    result = await collect(
        worker.stub,
        make_request(
            "tool-req", prompt, constraint=(TOOL_REQUIRED, json.dumps(WEATHER_TOOLS)),
            hints={"thinking": str(thinking).lower(), "last_role": "user"},
        ),
    )
    ids = result.token_ids
    assert tokenizer.think_end_id not in ids
    text = tokenizer.decode(ids)
    assert text.startswith('<｜DSML｜ invoke name="get_weather">\n')
    assert text.endswith("\n</｜DSML｜ calls>")
    # Together with the prompt's opening this is a complete block.
    calls = check_tool_block(tokenizer, ids, opened_by_prompt=True)
    assert [name for name, _ in calls] == ["get_weather"]
    validate(calls[0][1], WEATHER_TOOLS[0]["function"]["parameters"])
