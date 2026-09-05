"""Tokenising, length filtering, and the measurements that chose the vocabulary sizes.

EVBNews is already tokenised -- punctuation is space-separated -- so lowercasing and splitting on
whitespace is the whole tokenizer. The interesting part is what the measurements say about the two
languages, which is why they live here rather than in a notebook cell.
"""

from __future__ import annotations

import collections

import numpy as np

from src.config import Config


class Preprocessor:
    """Tokenisation, the length filter, and the corpus statistics behind the config."""

    def __init__(self, config: Config) -> None:
        self.max_len: int = config.require("preprocessing.max_len")
        self.coverage_targets = config.require("preprocessing.coverage_targets")

    @staticmethod
    def tokens(sentence: str) -> list[str]:
        return sentence.lower().split()

    def keep(self, english: str, vietnamese: str) -> bool:
        """Both sides non-empty and short enough to leave room for <bos>/<eos>."""
        return (1 <= len(english.split()) <= self.max_len - 2
                and 1 <= len(vietnamese.split()) <= self.max_len - 2)

    def filter_rows(self, rows):
        return [row for row in rows if self.keep(row[0], row[1])]

    # -- measurements -------------------------------------------------------------------

    def length_stats(self, rows) -> dict[str, dict]:
        stats = {}
        for name, index in (("EN", 0), ("VI", 1)):
            lengths = np.array([len(self.tokens(row[index])) for row in rows])
            stats[name] = {"mean": float(lengths.mean()), "median": float(np.median(lengths)),
                           "p95": float(np.percentile(lengths, 95)), "max": int(lengths.max())}
        return stats

    def counts(self, rows, index: int) -> collections.Counter:
        return collections.Counter(w for row in rows for w in self.tokens(row[index]))

    def coverage_table(self, counter: collections.Counter) -> list[dict]:
        """What share of tokens each candidate vocabulary size covers.

        This is what shows the asymmetry between the two languages: English needs 16k word types
        for 97.5% coverage, Vietnamese reaches 99.4% with 8k, because it writes syllables and its
        syllable inventory is close to a closed set.
        """
        total, ranked = sum(counter.values()), counter.most_common()
        rows = []
        for size in sorted(set(self.coverage_targets)):
            if size > len(ranked):
                continue
            covered = sum(n for _, n in ranked[:size]) / total
            rows.append({"vocab": size, "coverage": covered})
        return rows
