"""One vocabulary per language.

Separate rather than shared: the two sides have almost no word types in common, and sharing
would spend English slots on Vietnamese syllables and the other way round.
"""

from __future__ import annotations

import collections

import numpy as np

from src.config import Config


class Vocabulary:
    """Word <-> id for one language."""

    def __init__(self, itos: list[str], specials: list[str]) -> None:
        self.itos = itos
        self.stoi = {word: i for i, word in enumerate(itos)}
        self.specials = specials
        self.pad, self.unk, self.bos, self.eos = (specials.index(s) for s in
                                                  ("<pad>", "<unk>", "<bos>", "<eos>"))

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def build(cls, counter: collections.Counter, size: int, config: Config) -> "Vocabulary":
        specials = list(config.require("preprocessing.specials"))
        common = [word for word, _ in counter.most_common(size - len(specials))]
        return cls(specials + common, specials)

    def encode(self, tokens: list[str], length: int, with_markers: bool) -> list[int]:
        """Source sequences are bare; target sequences are wrapped in <bos> ... <eos>."""
        ids = [self.stoi.get(word, self.unk) for word in tokens]
        ids = [self.bos] + ids[: length - 2] + [self.eos] if with_markers else ids[:length]
        return ids + [self.pad] * (length - len(ids))

    def encode_many(self, sentences, preprocessor, length: int, with_markers: bool) -> np.ndarray:
        return np.array([self.encode(preprocessor.tokens(s), length, with_markers)
                         for s in sentences], np.int32)

    def decode(self, ids) -> list[str]:
        """Ids back to words, stopping at the first <eos> or <pad>."""
        words = []
        for i in ids:
            if i in (self.eos, self.pad):
                break
            if i != self.bos:
                words.append(self.itos[i])
        return words

    def to_payload(self) -> dict:
        return {"itos": self.itos, "specials": self.specials}

    @classmethod
    def from_payload(cls, payload: dict) -> "Vocabulary":
        return cls(payload["itos"], payload["specials"])
