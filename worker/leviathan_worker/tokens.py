"""The DeepSeek V4.1 tokenizer and the special tokens the worker relies on."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from tokenizers import Tokenizer

BOS = "<｜begin▁of▁sentence｜>"
EOS = "<｜end▁of▁sentence｜>"
THINK_START = "<think>"
THINK_END = "</think>"
DSML = "｜DSML｜"

#: The repository copy of the tokenizer, used when no path is given.
REPO_TOKENIZER = Path(__file__).resolve().parents[2] / "assets" / "tokenizers" / "v41" / "tokenizer.json"


def default_tokenizer_path() -> Path | None:
    """``$LEVIATHAN_TOKENIZER``, else the repository asset when it exists."""
    env = os.environ.get("LEVIATHAN_TOKENIZER")
    if env:
        return Path(env)
    return REPO_TOKENIZER if REPO_TOKENIZER.is_file() else None


class V41Tokenizer:
    """A ``tokenizers.Tokenizer`` with the V4.1 special token IDs resolved.

    Special tokens are added-vocabulary entries, so encoding text that spells
    them (``</think>``, ``｜DSML｜``) yields their IDs, which is what the
    gateway's output parser requires.
    """

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tok = tokenizer
        self.bos_id = self._id(BOS)
        self.eos_id = self._id(EOS)
        self.think_start_id = self._id(THINK_START)
        self.think_end_id = self._id(THINK_END)
        self.dsml_id = self._id(DSML)
        self.vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
        #: IDs below this are byte-level BPE tokens (the added vocabulary sits above it).
        self.plain_vocab_size = tokenizer.get_vocab_size(with_added_tokens=False)

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> V41Tokenizer:
        return cls(Tokenizer.from_file(str(path)))

    def _id(self, token: str) -> int:
        token_id = self._tok.token_to_id(token)
        if token_id is None:
            raise ValueError(f"tokenizer has no {token!r} token; is it the DeepSeek V4.1 tokenizer?")
        return token_id

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        return [e.ids for e in self._tok.encode_batch(list(texts), add_special_tokens=False)]

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = False) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=skip_special_tokens)
