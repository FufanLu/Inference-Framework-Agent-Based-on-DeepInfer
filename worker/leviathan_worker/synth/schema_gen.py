"""Generate a JSON value that conforms to a JSON Schema.

Covers what structured-output requests use in practice: ``type`` (single or a
list), ``properties`` / ``required`` / ``additionalProperties``, ``items`` /
``prefixItems`` / ``minItems`` / ``maxItems`` / ``uniqueItems``, ``enum``,
``const``, ``default``, numeric bounds (draft 4 and later) and ``multipleOf``,
string ``minLength`` / ``maxLength`` / ``pattern`` / ``format``, ``anyOf`` /
``oneOf`` / ``allOf``, and local ``$ref`` (``$defs``, ``definitions`` and the
``$def`` that deepseek-recipe keeps when it normalizes strict schemas).
String values are plausible for their property name (a city for ``city``, a
sentence for ``description``) and follow the request's language.
"""

from __future__ import annotations

import copy
import json
import math
import random
import re
import uuid
from collections.abc import Callable
from typing import Any

from . import regex_gen
from .corpus import ANSWER, CITIES, TERMS

MAX_DEPTH = 8
#: Probability of including an optional property.
OPTIONAL_PROBABILITY = 0.6
#: Probability of choosing ``null`` from a ``type`` list that allows it.
NULL_PROBABILITY = 0.1
_DEF_KEYS = ("$defs", "definitions", "$def")


class SchemaError(ValueError):
    """The schema is malformed or unsatisfiable."""


def generate(schema: Any, rng: random.Random, *, lang: str = "en") -> Any:
    """A value conforming to ``schema``; an empty schema yields a small JSON object."""
    return _Generator(schema, rng, lang).value(schema, name="", depth=0)


def generate_arguments(schema: Any, rng: random.Random, *, lang: str = "en") -> dict[str, Any]:
    """Tool-call arguments for a function's ``parameters`` schema (always an object)."""
    if not schema:
        return {}
    # Depth 1: an object without properties means "no arguments", not "any object".
    value = _Generator(schema, rng, lang).value(schema, name="", depth=1)
    if not isinstance(value, dict):
        return {}
    return {k: v for k, v in value.items() if _PARAM_NAME.fullmatch(k)}


# A DSML parameter name is written inside name="..." and ends at the first space.
_PARAM_NAME = re.compile(r'[^\s"\\<>]+')


class _Generator:
    def __init__(self, root: Any, rng: random.Random, lang: str) -> None:
        self._root = root
        self._rng = rng
        self._lang = lang

    # ------------------------------------------------------------------ dispatch

    def value(self, schema: Any, name: str, depth: int) -> Any:
        if schema is True or schema is None:
            return self._any(name, depth)
        if schema is False:
            raise SchemaError("the schema `false` accepts no value")
        if not isinstance(schema, dict):
            raise SchemaError(f"a schema must be an object, not {type(schema).__name__}")
        if depth > MAX_DEPTH:
            return self._minimal(schema)
        rng = self._rng
        if "$ref" in schema:
            target = self._resolve(schema["$ref"])
            return self._any(name, depth) if target is None else self.value(target, name, depth + 1)
        if "const" in schema:
            return copy.deepcopy(schema["const"])
        if "enum" in schema:
            options = schema["enum"]
            if not isinstance(options, list) or not options:
                raise SchemaError("enum must be a non-empty array")
            return copy.deepcopy(rng.choice(options))
        if "default" in schema:
            return copy.deepcopy(schema["default"])
        if "allOf" in schema:
            return self.value(self._merge_all_of(schema), name, depth + 1)
        for key in ("anyOf", "oneOf"):
            if key in schema:
                options = schema[key]
                if not isinstance(options, list) or not options:
                    raise SchemaError(f"{key} must be a non-empty array")
                return self.value(rng.choice(options), name, depth + 1)
        kind = self._type(schema)
        if kind == "object":
            return self._object(schema, depth)
        if kind == "array":
            return self._array(schema, name, depth)
        if kind == "string":
            return self._string(schema, name)
        if kind == "integer":
            return self._integer(schema)
        if kind == "number":
            return self._number(schema)
        if kind == "boolean":
            return rng.random() < 0.5
        if kind == "null":
            return None
        return self._any(name, depth)

    def _type(self, schema: dict[str, Any]) -> str:
        kind = schema.get("type")
        if isinstance(kind, list):
            if not kind:
                raise SchemaError("type must not be an empty array")
            non_null = [k for k in kind if k != "null"]
            if not non_null or (len(non_null) < len(kind) and self._rng.random() < NULL_PROBABILITY):
                return "null"
            return self._rng.choice(non_null)
        if isinstance(kind, str):
            if kind not in ("object", "array", "string", "integer", "number", "boolean", "null"):
                raise SchemaError(f"unknown type {kind!r}")
            return kind
        if kind is not None:
            raise SchemaError("type must be a string or an array of strings")
        if any(k in schema for k in ("properties", "required", "additionalProperties")):
            return "object"
        if any(k in schema for k in ("items", "prefixItems", "minItems", "maxItems")):
            return "array"
        if any(k in schema for k in ("pattern", "format", "minLength", "maxLength")):
            return "string"
        if any(k in schema for k in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf")):
            return "number"
        return "any"

    # ------------------------------------------------------------------ composites

    def _object(self, schema: dict[str, Any], depth: int) -> dict[str, Any]:
        properties = schema.get("properties") or {}
        required = schema.get("required") or []
        if not isinstance(properties, dict) or not isinstance(required, list):
            raise SchemaError("properties must be an object and required an array")
        out: dict[str, Any] = {}
        for key, sub in properties.items():
            if key in required or self._rng.random() < OPTIONAL_PROBABILITY:
                out[key] = self.value(sub, key, depth + 1)
        for key in required:
            if key not in out:
                out[key] = self._any(key, depth + 1)
        if not properties:
            extra = schema.get("additionalProperties", True)
            if isinstance(extra, dict) and extra:
                for term in self._rng.sample(TERMS["en"], 2):
                    out.setdefault(term.replace(" ", "_"), self.value(extra, term, depth + 1))
            elif extra is not False and depth == 0 and not required:
                out.update(self._generic_object())
        return out

    def _array(self, schema: dict[str, Any], name: str, depth: int) -> list[Any]:
        items = schema.get("items", {})
        prefix = schema.get("prefixItems") or []
        low = int(schema.get("minItems", 0))
        high = schema.get("maxItems")
        high = low + 3 if high is None else int(high)
        if high < low:
            raise SchemaError("maxItems is smaller than minItems")
        count_low = max(low, min(1, high))
        count = self._rng.randint(count_low, max(count_low, min(high, max(low, 3))))
        unique = bool(schema.get("uniqueItems"))
        if unique and not prefix and isinstance(items, dict) and isinstance(items.get("enum"), list):
            # Few possible values: draw them without replacement instead of retrying.
            options = items["enum"]
            if len(options) < low:
                raise SchemaError("uniqueItems needs more distinct enum values than minItems")
            return copy.deepcopy(self._rng.sample(options, min(count, len(options))))
        out: list[Any] = []
        seen: set[str] = set()
        for i in range(count):
            sub = prefix[i] if i < len(prefix) else items
            if sub is False:
                break
            for _ in range(20 if unique else 1):
                item = self.value(sub, name, depth + 1)
                key = json.dumps(item, sort_keys=True)
                if key not in seen:
                    break
            if unique and key in seen:
                continue
            seen.add(key)
            out.append(item)
        if len(out) < low:
            raise SchemaError("cannot generate enough distinct array items")
        return out

    def _merge_all_of(self, schema: dict[str, Any]) -> dict[str, Any]:
        merged = {k: v for k, v in schema.items() if k != "allOf"}
        parts = schema["allOf"]
        if not isinstance(parts, list):
            raise SchemaError("allOf must be an array")
        for part in parts:
            if isinstance(part, dict) and "$ref" in part:
                part = self._resolve(part["$ref"]) or {}
            if not isinstance(part, dict):
                continue
            for key, value in part.items():
                if key == "properties":
                    merged["properties"] = {**merged.get("properties", {}), **value}
                elif key == "required":
                    merged["required"] = list(dict.fromkeys([*merged.get("required", []), *value]))
                else:
                    merged.setdefault(key, value)
        return merged

    def _resolve(self, ref: Any) -> Any:
        """Follow a local JSON pointer; None when it does not resolve."""
        if not isinstance(ref, str) or not ref.startswith("#"):
            return None
        parts = [p.replace("~1", "/").replace("~0", "~") for p in ref[1:].split("/") if p]
        candidates = [parts]
        if parts and parts[0] in _DEF_KEYS:  # recipe keeps `$def` but leaves refs into `$defs` as written
            candidates += [[alt, *parts[1:]] for alt in _DEF_KEYS if alt != parts[0]]
        for path in candidates:
            node = self._root
            for part in path:
                if isinstance(node, dict) and part in node:
                    node = node[part]
                elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
                    node = node[int(part)]
                else:
                    break
            else:
                return node
        return None

    # ------------------------------------------------------------------ scalars

    def _string(self, schema: dict[str, Any], name: str) -> str:
        if "pattern" in schema:
            try:
                return regex_gen.generate(schema["pattern"], self._rng)
            except regex_gen.RegexError as exc:
                raise SchemaError(f"string pattern: {exc}") from None
        fmt = schema.get("format")
        if isinstance(fmt, str) and fmt in _FORMATS:
            return _FORMATS[fmt](self._rng)
        text = self._named_string(name)
        min_len = int(schema.get("minLength", 0))
        max_len = schema.get("maxLength")
        while len(text) < min_len:
            text += (" " if self._lang == "en" else "") + self._rng.choice(TERMS[self._lang])
        if max_len is not None and len(text) > int(max_len):
            text = text[: int(max_len)]
        return text

    def _integer(self, schema: dict[str, Any]) -> int:
        low, high = _bounds(schema, integer=True)
        multiple = schema.get("multipleOf")
        if multiple:
            k_low, k_high = math.ceil(low / multiple), math.floor(high / multiple)
            if k_low > k_high:
                raise SchemaError("no multiple of multipleOf lies within the bounds")
            return int(self._rng.randint(k_low, k_high) * multiple)
        return self._rng.randint(int(low), int(high))

    def _number(self, schema: dict[str, Any]) -> float:
        low, high = _bounds(schema, integer=False)
        multiple = schema.get("multipleOf")
        if multiple:
            k_low, k_high = math.ceil(low / multiple), math.floor(high / multiple)
            if k_low > k_high:
                raise SchemaError("no multiple of multipleOf lies within the bounds")
            return round(self._rng.randint(k_low, k_high) * multiple, 10)
        value = round(self._rng.uniform(low, high), 2)
        if not _within(schema, value):
            value = (low + high) / 2
        return float(value)

    # ------------------------------------------------------------------ content

    def _named_string(self, name: str) -> str:
        words = {w.lower() for w in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", name)}
        for keywords, make in _NAME_RULES:
            if words & keywords:
                return make(self._rng, self._lang)
        return self._rng.choice(TERMS[self._lang])

    def _generic_object(self) -> dict[str, Any]:
        rng, lang = self._rng, self._lang
        return {
            "answer": rng.choice(ANSWER[lang]),
            "confidence": round(rng.uniform(0.6, 0.99), 2),
            "keywords": rng.sample(TERMS[lang], 2),
        }

    def _any(self, name: str, depth: int) -> Any:
        return self._generic_object() if depth == 0 else self._named_string(name)

    def _minimal(self, schema: dict[str, Any]) -> Any:
        kind = schema.get("type")
        kind = kind[0] if isinstance(kind, list) and kind else kind
        return {"object": {}, "array": [], "string": "", "integer": 0, "number": 0.0, "boolean": False}.get(
            kind if isinstance(kind, str) else "", None
        )


def _bounds(schema: dict[str, Any], *, integer: bool) -> tuple[float, float]:
    """Inclusive bounds; exclusive ones are tightened by 1 (integers) or a tiny step (numbers)."""
    step = 1 if integer else 1e-6
    low, high = -math.inf, math.inf
    minimum, maximum = schema.get("minimum"), schema.get("maximum")
    ex_min, ex_max = schema.get("exclusiveMinimum"), schema.get("exclusiveMaximum")
    if minimum is not None:
        low = minimum + (step if ex_min is True else 0)  # draft 4: boolean exclusiveMinimum
    if maximum is not None:
        high = maximum - (step if ex_max is True else 0)
    if ex_min is not None and not isinstance(ex_min, bool):
        low = max(low, ex_min + step)
    if ex_max is not None and not isinstance(ex_max, bool):
        high = min(high, ex_max - step)
    if low == -math.inf and high == math.inf:
        low, high = 0, 100
    elif low == -math.inf:
        low = high - 100
    elif high == math.inf:
        high = low + 100
    if integer:
        low, high = math.ceil(low), math.floor(high)
    if low > high:
        raise SchemaError("the numeric bounds are empty")
    return low, high


def _within(schema: dict[str, Any], value: float) -> bool:
    try:
        low, high = _bounds(schema, integer=False)
    except SchemaError:
        return False
    return low <= value <= high


def _date(rng: random.Random) -> str:
    return f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"


def _time(rng: random.Random) -> str:
    return f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:00"


def _slug(rng: random.Random) -> str:
    return rng.choice(TERMS["en"]).lower().replace(" ", "-")


_FORMATS: dict[str, Callable[[random.Random], str]] = {
    "date": _date,
    "time": _time,
    "date-time": lambda rng: f"{_date(rng)}T{_time(rng)}Z",
    "email": lambda rng: f"{_slug(rng)}@example.com",
    "hostname": lambda rng: f"{_slug(rng)}.example.com",
    "ipv4": lambda rng: f"10.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}",
    "ipv6": lambda rng: f"fd00::{rng.getrandbits(16):x}:{rng.getrandbits(16):x}",
    "uuid": lambda rng: str(uuid.UUID(int=rng.getrandbits(128), version=4)),
    "uri": lambda rng: f"https://example.com/{_slug(rng)}",
    "url": lambda rng: f"https://example.com/{_slug(rng)}",
    "uri-reference": lambda rng: f"/docs/{_slug(rng)}",
    "duration": lambda rng: f"PT{rng.randint(1, 59)}M",
}

_NameRule = tuple[frozenset[str], Callable[[random.Random, str], str]]
_NAME_RULES: list[_NameRule] = [
    (frozenset({"email", "mail"}), lambda rng, lang: _FORMATS["email"](rng)),
    (frozenset({"url", "uri", "link", "website", "homepage", "endpoint"}), lambda rng, lang: _FORMATS["url"](rng)),
    (frozenset({"city", "location", "place", "region", "address", "destination"}), lambda rng, lang: rng.choice(CITIES[lang])),
    (frozenset({"date", "day", "deadline"}), lambda rng, lang: _date(rng)),
    (frozenset({"time", "timestamp", "datetime"}), lambda rng, lang: _FORMATS["date-time"](rng)),
    (frozenset({"lang", "language", "locale"}), lambda rng, lang: lang),
    (frozenset({"unit", "units"}), lambda rng, lang: rng.choice(("celsius", "fahrenheit"))),
    (frozenset({"phone", "tel"}), lambda rng, lang: f"+1-555-01{rng.randint(0, 99):02d}"),
    (frozenset({"id", "uuid", "key", "token"}), lambda rng, lang: f"{rng.getrandbits(48):012x}"),
    (
        frozenset({"query", "question", "prompt", "search", "q"}),
        lambda rng, lang: (
            f"How does {rng.choice(TERMS['en'])} work?" if lang == "en" else f"{rng.choice(TERMS['zh'])}是如何工作的？"
        ),
    ),
    (
        frozenset(
            {"description", "summary", "text", "content", "message", "reason", "answer", "comment", "note",
             "body", "explanation", "detail", "details"}
        ),
        lambda rng, lang: rng.choice(ANSWER[lang]),
    ),
]
