"""Turning an article and its abstract into token lists, and measuring what that costs.

VNDS arrives already word-segmented, so there is no cleaning pipeline here -- splitting on
whitespace is the whole tokenizer. What this class does contribute is the measurements that
justify `max_article`, which is the one preprocessing decision with a real price attached.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.config import Config


class Preprocessor:
    """Truncation, sentence splitting, and the statistics behind the truncation length."""

    def __init__(self, config: Config) -> None:
        self.max_article: int = config.require("preprocessing.max_article")
        self.max_summary: int = config.require("preprocessing.max_summary")
        self.sentence_pattern = re.compile(config.require("preprocessing.sentence_split"))

    def article_tokens(self, text: str) -> list[str]:
        return text.split()[: self.max_article]

    def summary_tokens(self, text: str) -> list[str]:
        """Two slots are reserved for <bos> and <eos>, which the vocabulary adds."""
        return text.split()[: self.max_summary - 2]

    def sentences(self, text: str) -> list[str]:
        return [s.strip() for s in self.sentence_pattern.split(text) if s.strip()]

    # -- measurements -------------------------------------------------------------------

    def length_stats(self, frame: pd.DataFrame) -> dict[str, dict]:
        """Length distribution of both fields, and what share each limit keeps whole."""
        stats = {}
        for name, column, limit in (("article", "article", self.max_article),
                                    ("abstract", "abstract", self.max_summary)):
            lengths = frame[column].str.split().str.len().to_numpy()
            stats[name] = {
                "mean": float(lengths.mean()),
                "median": float(np.median(lengths)),
                "p90": float(np.percentile(lengths, 90)),
                "p95": float(np.percentile(lengths, 95)),
                "max": int(lengths.max()),
                "within_limit": float((lengths <= limit).mean()),
            }
        return stats

    def lead_coverage(self, frame: pd.DataFrame, positions, sample: int = 2000,
                      seed: int = 42) -> dict[int, float]:
        """Share of the abstract's words that appear in the first N tokens of the article.

        This is what decides `max_article`. If the number barely moves past some N, the rest of
        the article is mostly not where the summary comes from, and truncating there is cheap.
        It is also the first evidence of the lead bias that makes the Lead-n baseline strong.
        """
        probe = frame.sample(min(sample, len(frame)), random_state=seed)
        coverage = {}
        for n in positions:
            shares = [
                sum(word in set(article.split()[:n]) for word in abstract.split())
                / max(len(abstract.split()), 1)
                for article, abstract in zip(probe.article, probe.abstract)
            ]
            coverage[n] = float(np.mean(shares))
        return coverage
