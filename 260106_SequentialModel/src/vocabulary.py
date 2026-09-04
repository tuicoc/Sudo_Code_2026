"""The fixed 20,000-word vocabulary, and the mapping both ways.

Why cap it: the softmax is the most expensive part of the model and its cost is linear in
vocabulary size. `coverage_table` prints how much text each candidate size covers, so the
cap comes from numbers rather than a guess.

Index 0 is `<pad>` and is never emitted, so a padded batch cannot be mistaken for real
text. 1 is `<unk>`, 2 is `<eob>` (appended per book, so the model learns that books end).
"""

from __future__ import annotations

import collections
import re

import numpy as np

from src.config import Config

UNKNOWN_ID = 1
END_OF_BOOK_ID = 2


class Vocabulary:
    """Maps tokens to ids and back, built from token counts over the whole corpus."""

    def __init__(self, itos: list[str]) -> None:
        self.itos = itos
        self.stoi = {token: index for index, token in enumerate(itos)}

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def build(cls, counts: collections.Counter, config: Config) -> "Vocabulary":
        """Take the most common tokens, after the reserved special tokens."""
        specials: list[str] = config.require("vocabulary.specials")
        size: int = config.require("vocabulary.size")
        most_common = [token for token, _count in counts.most_common(size - len(specials))]
        return cls(specials + most_common)

    @staticmethod
    def coverage_table(counts: collections.Counter, config: Config) -> list[dict]:
        """Per candidate size: what share of tokens it covers, and the `<unk>` rate it costs."""
        targets = set(config.require("vocabulary.coverage_targets"))
        total = sum(counts.values())
        rows, cumulative = [], 0
        for rank, (_token, count) in enumerate(counts.most_common(), start=1):
            cumulative += count
            if rank in targets:
                coverage = cumulative / total
                rows.append({
                    "vocab": rank,
                    "coverage": coverage,
                    "unk_rate": 1 - coverage,
                })
        return rows

    def encode(self, tokens: list[str]) -> list[int]:
        """Tokens -> ids, with anything out of vocabulary mapped to `<unk>`."""
        return [self.stoi.get(token, UNKNOWN_ID) for token in tokens]

    def decode(self, ids) -> str:
        """Ids -> text. Punctuation is re-attached to the word before it."""
        return re.sub(r" ([,.!?;:])", r"\1", " ".join(self.itos[int(i)] for i in ids))

    def encode_book(self, tokens: list[str]) -> np.ndarray:
        """One book as a `uint16` array, terminated by `<eob>`."""
        return np.array(self.encode(tokens) + [END_OF_BOOK_ID], dtype=np.uint16)

    def to_payload(self, extra: dict | None = None) -> dict:
        """What gets written to `vocab.json`."""
        return {"itos": self.itos, "vocab_size": len(self.itos), **(extra or {})}

    @classmethod
    def from_payload(cls, payload: dict) -> "Vocabulary":
        return cls(payload["itos"])
