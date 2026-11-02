"""Unit tests of the synthetic output generators and the sim profiles."""

from __future__ import annotations

import json
import math
import random
import re

import pytest
from support import validate

from leviathan_worker.backends.sim_profile import PROFILE_DIR, SimProfile, available_profiles
from leviathan_worker.engine import ConstraintKind, OutputConstraint
from leviathan_worker.synth import OutputSynthesizer, fake_logprobs, render_invocation, request_seed
from leviathan_worker.synth import regex_gen, schema_gen

REGEXES = [
    r"abc",
    r"\d+",
    r"\w{2,5}\s\W",
    r"[a-c]{3}|[x-z]{2}",
    r"(foo|bar)(baz)?",
    r"[^abc]{4}",
    r"[^\d\s]+",
    r".{3}",
    r"a*?b+?c??d{0,2}",
    r"(?:ab|cd){2,3}",
    r"(?P<q>['\"]).*(?P=q)",
    r"^start-\d{2}$",
    r"(a)?(?(1)b|c)",
    r"[一-龥]{2}",
    r"(?x) a \d  # verbose",
    r"[A-Z]{2}\d{3}\.\d",
]


@pytest.mark.parametrize("pattern", REGEXES)
def test_regex_generator_fullmatches(pattern):
    for seed in range(40):
        text = regex_gen.generate(pattern, random.Random(seed))
        assert re.fullmatch(pattern, text), (pattern, text)


@pytest.mark.parametrize("pattern", ["(", "[z-a]", r"(?=a)b", r"[^\s\S]"])
def test_regex_generator_rejects_invalid_or_unsatisfiable_patterns(pattern):
    with pytest.raises(regex_gen.RegexError):
        regex_gen.generate(pattern, random.Random(0))


SCHEMAS = [
    {},
    {"type": "string"},
    {"type": ["string", "null"]},
    {"type": "integer", "minimum": -5, "exclusiveMaximum": 5},
    {"type": "integer", "minimum": 0, "maximum": 1000, "multipleOf": 7},
    {"type": "number", "exclusiveMinimum": 0.5, "maximum": 0.75},
    {"type": "number", "minimum": 1, "exclusiveMinimum": True},  # draft 4
    {"type": "string", "minLength": 30, "maxLength": 40},
    {"type": "string", "pattern": "^[a-f0-9]{8}$"},
    {"type": "string", "format": "date-time"},
    {"enum": [1, "two", None]},
    {"const": {"fixed": True}},
    {"type": "array", "items": {"type": "boolean"}, "minItems": 2, "maxItems": 2},
    {"type": "array", "items": {"enum": ["a", "b", "c"]}, "minItems": 3, "uniqueItems": True},
    {"type": "array", "prefixItems": [{"type": "integer"}, {"type": "string"}], "items": False},
    {"anyOf": [{"type": "integer"}, {"type": "object", "properties": {"k": {"type": "string"}}, "required": ["k"]}]},
    {"allOf": [{"properties": {"a": {"type": "integer"}}, "required": ["a"]}, {"required": ["b"]}]},
    {
        "type": "object",
        "properties": {"node": {"$ref": "#/$def/node"}},
        "required": ["node"],
        "$def": {
            "node": {
                "type": "object",
                "properties": {"value": {"type": "integer"}, "next": {"anyOf": [{"$ref": "#/$def/node"}, {"type": "null"}]}},
                "required": ["value", "next"],
            }
        },
    },
    {"type": "object", "additionalProperties": {"type": "number"}},
]


@pytest.mark.parametrize("schema", SCHEMAS, ids=[json.dumps(s)[:40] for s in SCHEMAS])
def test_schema_generator_conforms(schema):
    for seed in range(40):
        for lang in ("en", "zh"):
            value = schema_gen.generate(schema, random.Random(seed), lang=lang)
            validate(value, schema)
            json.dumps(value)


def test_schema_generator_uses_defaults_and_resolves_refs_into_dropped_defs():
    assert schema_gen.generate({"type": "string", "default": "auto"}, random.Random(0)) == "auto"
    # recipe keeps `$def` but leaves a `$ref` into `$defs` as written.
    schema = {"type": "object", "properties": {"p": {"$ref": "#/$defs/p"}}, "required": ["p"],
              "$def": {"p": {"type": "integer", "minimum": 3, "maximum": 3}}}
    assert schema_gen.generate(schema, random.Random(0)) == {"p": 3}
    unresolved = {"type": "object", "properties": {"x": {"$ref": "#/nowhere"}}, "required": ["x"]}
    assert "x" in schema_gen.generate(unresolved, random.Random(0))


@pytest.mark.parametrize(
    "schema",
    [False, {"type": "integer", "minimum": 5, "maximum": 1}, {"enum": []}, {"type": "rational"},
     {"type": "integer", "minimum": 1, "maximum": 2, "multipleOf": 5}],
)
def test_schema_generator_rejects_unsatisfiable_schemas(schema):
    with pytest.raises(schema_gen.SchemaError):
        schema_gen.generate(schema, random.Random(0))


def test_strings_fit_their_property_names():
    rng = random.Random(3)
    args = schema_gen.generate_arguments(
        {"type": "object", "properties": {"city": {"type": "string"}, "contact_email": {"type": "string"}},
         "required": ["city", "contact_email"]},
        rng,
        lang="zh",
    )
    assert re.fullmatch(r"[一-鿿]+", args["city"])
    assert args["contact_email"].endswith("@example.com")


def test_render_invocation_matches_recipe_dsv41():
    text = render_invocation("get_weather", {"city": "北京", "days": 3, "opts": {"a": [1, 0.5]}, "ok": True, "x": None})
    assert text == (
        '<｜DSML｜ invoke name="get_weather">\n'
        '<｜DSML｜ parameter name="city" string="true">北京</｜DSML｜ parameter>\n'
        '<｜DSML｜ parameter name="days" string="false">3</｜DSML｜ parameter>\n'
        '<｜DSML｜ parameter name="opts" string="false">{"a": [1, 0.5]}</｜DSML｜ parameter>\n'
        '<｜DSML｜ parameter name="ok" string="false">true</｜DSML｜ parameter>\n'
        '<｜DSML｜ parameter name="x" string="false">null</｜DSML｜ parameter>\n'
        "</｜DSML｜ invoke>"
    )
    assert render_invocation("ping", {}) == '<｜DSML｜ invoke name="ping">\n\n</｜DSML｜ invoke>'


TOOLS = json.dumps(
    [{"type": "function", "function": {"name": "lookup", "description": "", "strict": None,
                                        "parameters": {"type": "object", "properties": {"q": {"type": "string"}},
                                                       "required": ["q"]}}}]
)
CASES = [
    (None, {"thinking": "false"}),
    (None, {"thinking": "true", "effort": "max"}),
    (OutputConstraint(ConstraintKind.JSON, ""), {"thinking": "true"}),
    (OutputConstraint(ConstraintKind.REGEX, r"[a-z]{4}\d"), {}),
    (OutputConstraint(ConstraintKind.TOOL_CALL, TOOLS), {"last_role": "user", "thinking": "true"}),
    (OutputConstraint(ConstraintKind.TOOL_CALL, TOOLS), {"last_role": "tool"}),
    (OutputConstraint(ConstraintKind.TOOL_CALL_REQUIRED, TOOLS), {"thinking": "true"}),
]


@pytest.mark.parametrize(("constraint", "hints"), CASES)
def test_synthesized_text_round_trips_through_the_tokenizer(tokenizer, constraint, hints):
    synth = OutputSynthesizer(tokenizer, mean_output_tokens=120)
    required = constraint is not None and constraint.kind is ConstraintKind.TOOL_CALL_REQUIRED
    reasons = hints.get("thinking") == "true" and not required
    for seed, user in enumerate(["What is KV cache offloading?", "什么是 KV cache 卸载？"] * 3):
        prompt = tokenizer.encode(f"<｜User｜>{user}<｜Assistant｜></think>")
        out = synth.synthesize(prompt, random.Random(seed), constraint, hints)
        assert tokenizer.decode(out.token_ids) == out.text
        assert tokenizer.encode(out.text) == list(out.token_ids)
        assert tokenizer.eos_id not in out.token_ids
        assert ("</think>" in out.text) == reasons


def test_thinking_falls_back_to_the_prompt_suffix(tokenizer):
    synth = OutputSynthesizer(tokenizer, mean_output_tokens=40)
    thinking = tokenizer.encode("<｜User｜>hi<｜Assistant｜><think>")
    answering = tokenizer.encode("<｜User｜>hi<｜Assistant｜></think>")
    assert "</think>" in synth.synthesize(thinking, random.Random(0), None, {}).text
    assert "</think>" not in synth.synthesize(answering, random.Random(0), None, {}).text
    assert "</think>" not in synth.synthesize(thinking, random.Random(0), None, {"thinking": "false"}).text


def test_ignore_eos_padding_reaches_the_requested_length(tokenizer):
    synth = OutputSynthesizer(tokenizer, mean_output_tokens=20)
    prompt = tokenizer.encode("<｜User｜>hi<｜Assistant｜></think>")
    assert len(synth.synthesize(prompt, random.Random(0), None, {}, min_tokens=500).token_ids) >= 500


def test_request_seed_depends_only_on_inputs():
    json_c = OutputConstraint(ConstraintKind.JSON, "{}")
    base = request_seed([1, 2, 3], 7, None, {"thinking": "true"})
    assert base == request_seed((1, 2, 3), 7, None, {"thinking": "true"})
    assert len({
        base,
        request_seed([1, 2, 4], 7, None, {"thinking": "true"}),
        request_seed([1, 2, 3], 8, None, {"thinking": "true"}),
        request_seed([1, 2, 3], 7, None, {"thinking": "false"}),
        request_seed([1, 2, 3], 7, json_c, {"thinking": "true"}),
    }) == 5


def test_fake_logprobs_are_consistent():
    ids = list(range(100, 160))
    first = fake_logprobs(ids, random.Random(5), top_n=5, plain_vocab_size=128000)
    assert first == fake_logprobs(ids, random.Random(5), top_n=5, plain_vocab_size=128000)
    for token, entry in zip(ids, first):
        assert entry.top_token_ids[0] == token and len(set(entry.top_token_ids)) == 5
        assert all(3 <= t < 128000 for t in entry.top_token_ids[1:])
        assert all(a > b for a, b in zip(entry.top_logprobs, entry.top_logprobs[1:]))
        assert sum(math.exp(v) for v in entry.top_logprobs) <= 1 + 1e-9
    bare = fake_logprobs(ids, random.Random(5), top_n=0, plain_vocab_size=128000)
    assert all(not e.top_token_ids and e.logprob <= 0 for e in bare)


# --------------------------------------------------------------------------- profiles


def test_bundled_profiles_load():
    assert set(available_profiles()) >= {"h200-vllm", "fast-dev"}
    h200 = SimProfile.load("h200-vllm")
    assert "PLACEHOLDER" in h200.source
    assert (h200.block_size, h200.spec_num_draft) == (64, 5)
    assert SimProfile.load("fast-dev").effective_context_limit == 8192


def test_profile_overrides_coerce_command_line_strings():
    profile = SimProfile.load("fast-dev").override(block_size="32", spec_accept_prob="0.5", total_blocks=4)
    assert (profile.block_size, profile.spec_accept_prob, profile.total_blocks) == (32, 0.5, 4)
    assert profile.effective_context_limit == 128


@pytest.mark.parametrize(
    "changes",
    [{"bogus": 1}, {"block_size": 0}, {"spec_accept_prob": 1.5}, {"max_batched_tokens": 100}, {"total_blocks": "x"}],
)
def test_profile_validation(changes):
    with pytest.raises(ValueError):
        SimProfile.load("fast-dev").override(**changes)


def test_profile_files_reject_unknown_and_missing_keys(tmp_path):
    good = json.loads((PROFILE_DIR / "fast-dev.json").read_text(encoding="utf-8"))
    (tmp_path / "typo.json").write_text(json.dumps({**good, "prefil_base_us": 1}))
    (tmp_path / "short.json").write_text(json.dumps({k: v for k, v in good.items() if k != "block_size"}))
    (tmp_path / "custom.json").write_text(json.dumps({**good, "mean_output_tokens": 7}))
    with pytest.raises(ValueError, match="unknown profile keys: prefil_base_us"):
        SimProfile.load(str(tmp_path / "typo.json"))
    with pytest.raises(ValueError, match="missing profile keys: block_size"):
        SimProfile.load(str(tmp_path / "short.json"))
    custom = SimProfile.load(str(tmp_path / "custom.json"))
    assert (custom.name, custom.mean_output_tokens) == ("custom", 7)
    with pytest.raises(ValueError, match="no sim profile"):
        SimProfile.load("deepinfer-sm100")
