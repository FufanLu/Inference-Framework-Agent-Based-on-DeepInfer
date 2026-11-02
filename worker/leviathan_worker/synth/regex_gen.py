"""Generate a string that fully matches a regular expression.

Walks the parse tree of Python's own regex parser, so the accepted syntax is
exactly ``re``'s. Literals, classes (ranges, categories, negation), repeats
(greedy, lazy, possessive), alternation, groups, backreferences and
conditionals are generated; anchors and lookarounds emit nothing. Every result
is checked with ``re.fullmatch`` and regenerated on a miss.
"""

from __future__ import annotations

import random
import re
import string
from collections.abc import Callable
from typing import Any

try:  # Python 3.11+
    from re import _constants as _c
    from re import _parser as _sre_parse
except ImportError:  # pragma: no cover - Python 3.10
    import sre_constants as _c  # type: ignore[no-redef]
    import sre_parse as _sre_parse  # type: ignore[no-redef]

MAXREPEAT = _c.MAXREPEAT
#: Extra repetitions beyond a repeat's minimum, for unbounded or wide repeats.
EXTRA_REPEATS = 3
ATTEMPTS = 25

# Characters for ``.`` and negated classes: printable ASCII without the
# characters of markup (``<``, ``>``, ``|``) and backquotes.
_ALPHABET = "".join(c for c in string.ascii_letters + string.digits + " -_.,:;!?@#%&*+=/()[]{}'\"" if c not in "<>|`")
_WORD = string.ascii_letters + string.digits + "_"


class RegexError(ValueError):
    """The pattern is invalid or no matching string could be generated."""


def _category_chars() -> dict[Any, str]:
    table = {
        "CATEGORY_DIGIT": string.digits,
        "CATEGORY_NOT_DIGIT": string.ascii_letters,
        "CATEGORY_SPACE": " ",
        "CATEGORY_NOT_SPACE": string.ascii_letters + string.digits,
        "CATEGORY_WORD": _WORD,
        "CATEGORY_NOT_WORD": " -.,!?",
        "CATEGORY_LINEBREAK": "\n",
        "CATEGORY_NOT_LINEBREAK": string.ascii_letters,
    }
    result: dict[Any, str] = {}
    for name, chars in table.items():
        for variant in (name, name.replace("CATEGORY_", "CATEGORY_UNI_"), name.replace("CATEGORY_", "CATEGORY_LOC_")):
            if hasattr(_c, variant):
                result[getattr(_c, variant)] = chars
    return result


def _category_tests() -> dict[Any, Callable[[str], bool]]:
    def word(ch: str) -> bool:
        return ch.isalnum() or ch == "_"

    table: dict[str, Callable[[str], bool]] = {
        "CATEGORY_DIGIT": str.isdigit,
        "CATEGORY_NOT_DIGIT": lambda ch: not ch.isdigit(),
        "CATEGORY_SPACE": str.isspace,
        "CATEGORY_NOT_SPACE": lambda ch: not ch.isspace(),
        "CATEGORY_WORD": word,
        "CATEGORY_NOT_WORD": lambda ch: not word(ch),
        "CATEGORY_LINEBREAK": lambda ch: ch == "\n",
        "CATEGORY_NOT_LINEBREAK": lambda ch: ch != "\n",
    }
    result: dict[Any, Callable[[str], bool]] = {}
    for name, test in table.items():
        for variant in (name, name.replace("CATEGORY_", "CATEGORY_UNI_"), name.replace("CATEGORY_", "CATEGORY_LOC_")):
            if hasattr(_c, variant):
                result[getattr(_c, variant)] = test
    return result


_CATEGORY_CHARS = _category_chars()
_CATEGORY_TESTS = _category_tests()
_REPEATS = {getattr(_c, n) for n in ("MAX_REPEAT", "MIN_REPEAT", "POSSESSIVE_REPEAT") if hasattr(_c, n)}
_ATOMIC = getattr(_c, "ATOMIC_GROUP", None)


def generate(pattern: str, rng: random.Random) -> str:
    """A string ``s`` with ``re.fullmatch(pattern, s)``; raises RegexError."""
    try:
        compiled = re.compile(pattern)
        tree = _sre_parse.parse(pattern)
    except (re.error, RecursionError, OverflowError) as exc:
        raise RegexError(f"invalid regex: {exc}") from None
    for _ in range(ATTEMPTS):
        text = _Generator(rng).sequence(tree)
        if compiled.fullmatch(text):
            return text
    raise RegexError(f"could not generate text that fully matches {pattern!r}")


class _Generator:
    def __init__(self, rng: random.Random) -> None:
        self._rng = rng
        self._groups: dict[int, str] = {}

    def sequence(self, items: Any) -> str:
        return "".join(self._item(op, av) for op, av in items)

    def _item(self, op: Any, av: Any) -> str:
        rng = self._rng
        if op is _c.LITERAL:
            return chr(av)
        if op is _c.NOT_LITERAL:
            return rng.choice([c for c in _ALPHABET if ord(c) != av])
        if op is _c.ANY:
            return rng.choice(_ALPHABET)
        if op is _c.IN:
            return self._in(av)
        if op in _REPEATS:
            low, high, sub = av
            return "".join(self.sequence(sub) for _ in range(self._count(low, high)))
        if op is _c.SUBPATTERN:
            group, _add_flags, _del_flags, sub = av
            text = self.sequence(sub)
            if group:
                self._groups[group] = text
            return text
        if _ATOMIC is not None and op is _ATOMIC:
            return self.sequence(av)
        if op is _c.BRANCH:
            _, alternatives = av
            return self.sequence(rng.choice(alternatives))
        if op is _c.GROUPREF:
            return self._groups.get(av, "")
        if op is _c.GROUPREF_EXISTS:
            group, yes, no = av
            return self.sequence(yes if group in self._groups else (no or []))
        if op in (_c.AT, _c.ASSERT, _c.ASSERT_NOT):
            return ""  # anchors and lookarounds; fullmatch verification catches misses
        raise RegexError(f"unsupported regex construct {op}")

    def _count(self, low: int, high: int) -> int:
        top = low + EXTRA_REPEATS if high == MAXREPEAT else min(high, low + EXTRA_REPEATS)
        return self._rng.randint(low, top)

    def _in(self, items: list[tuple[Any, Any]]) -> str:
        rng = self._rng
        if items and items[0][0] is _c.NEGATE:
            rest = items[1:]
            allowed = [c for c in _ALPHABET if not _class_matches(rest, c)]
            if not allowed:
                raise RegexError("negated character class excludes every generatable character")
            return rng.choice(allowed)
        op, av = rng.choice(items)
        if op is _c.LITERAL:
            return chr(av)
        if op is _c.RANGE:
            low, high = av
            # Prefer a printable ASCII character when the range has one.
            ascii_low, ascii_high = max(low, 0x20), min(high, 0x7E)
            if ascii_low <= ascii_high:
                return chr(rng.randint(ascii_low, ascii_high))
            return chr(rng.randint(low, min(high, low + 0xFF)))
        if op is _c.CATEGORY:
            return rng.choice(_CATEGORY_CHARS[av])
        raise RegexError(f"unsupported character class item {op}")


def _class_matches(items: list[tuple[Any, Any]], ch: str) -> bool:
    for op, av in items:
        if op is _c.LITERAL and ord(ch) == av:
            return True
        if op is _c.RANGE and av[0] <= ord(ch) <= av[1]:
            return True
        if op is _c.CATEGORY and _CATEGORY_TESTS[av](ch):
            return True
    return False
