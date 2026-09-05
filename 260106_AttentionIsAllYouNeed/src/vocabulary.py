"""One shared vocabulary for articles and abstracts.

It is the same language on both sides, so sharing it lets a single embedding matrix serve the
encoder and the decoder, and a word copied out of the article keeps the same id.
"""

from __future__ import annotations

import collections

import numpy as np

from src.config import Config


class Vocabulary:
    """Word <-> id, plus the coverage table that justifies the size."""

    def __init__(self, itos: list[str], specials: list[str]) -> None:
        self.itos = itos
        self.stoi = {word: i for i, word in enumerate(itos)}
        self.specials = specials
        self.pad, self.unk, self.bos, self.eos = (specials.index(s) for s in
                                                  ("<pad>", "<unk>", "<bos>", "<eos>"))

    def __len__(self) -> int:
        return len(self.itos)

    # -- building -----------------------------------------------------------------------

    @staticmethod
    def count(frame, preprocessor) -> collections.Counter:
        """Count over the *truncated* article, so the counts match what the model will see."""
        counts = collections.Counter()
        for article, abstract in zip(frame.article, frame.abstract):
            counts.update(preprocessor.article_tokens(article))
            counts.update(abstract.split())
        return counts

    @classmethod
    def build(cls, counts: collections.Counter, config: Config) -> "Vocabulary":
        specials = list(config.require("vocabulary.specials"))
        size = config.require("vocabulary.size")
        common = [word for word, _ in counts.most_common(size - len(specials))]
        return cls(specials + common, specials)

    @staticmethod
    def coverage_table(counts: collections.Counter, config: Config) -> list[dict]:
        """What share of tokens each candidate vocabulary size covers, and the <unk> rate left."""
        total = sum(counts.values())
        ranked = counts.most_common()
        rows = []
        for size in sorted(set(config.require("vocabulary.coverage_targets"))):
            if size > len(ranked):
                continue
            covered = sum(n for _, n in ranked[:size]) / total
            rows.append({"vocab": size, "coverage": covered, "unk_rate": 1 - covered})
        return rows

    # -- encoding -----------------------------------------------------------------------

    def encode_article(self, tokens: list[str], length: int) -> list[int]:
        ids = [self.stoi.get(word, self.unk) for word in tokens]
        return ids + [self.pad] * (length - len(ids))

    def encode_summary(self, tokens: list[str], length: int) -> list[int]:
        ids = [self.bos] + [self.stoi.get(word, self.unk) for word in tokens] + [self.eos]
        return ids + [self.pad] * (length - len(ids))

    def encode_frame(self, frame, preprocessor) -> tuple[np.ndarray, np.ndarray]:
        articles = np.array([self.encode_article(preprocessor.article_tokens(a),
                                                 preprocessor.max_article)
                             for a in frame.article], np.int32)
        summaries = np.array([self.encode_summary(preprocessor.summary_tokens(s),
                                                  preprocessor.max_summary)
                              for s in frame.abstract], np.int32)
        return articles, summaries

    def decode(self, ids) -> list[str]:
        """Ids back to words, stopping at the first <eos> or <pad>."""
        words = []
        for i in ids:
            if i in (self.eos, self.pad):
                break
            if i != self.bos:
                words.append(self.itos[i])
        return words

    # -- persistence --------------------------------------------------------------------

    def to_payload(self, extra: dict | None = None) -> dict:
        return {"itos": self.itos, "specials": self.specials, **(extra or {})}

    @classmethod
    def from_payload(cls, payload: dict) -> "Vocabulary":
        return cls(payload["itos"], payload["specials"])
