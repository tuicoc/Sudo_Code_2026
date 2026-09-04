"""The two vectorizers being compared: Bag-of-Words counts, and TF-IDF weights.

Both are built fresh per CV fold. Fitting once on the whole corpus would leak the
validation fold's vocabulary and document frequencies into training.
"""

from __future__ import annotations

from typing import Callable

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.config import Config


class FeatureExtractor:
    """Builds the vectorizer named by an experiment's `vectorizer` config key."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.ngram_range = tuple(config.require("vectorizer.ngram_range"))
        self.min_df: int = config.require("vectorizer.min_df")

    def factory(self, kind: str) -> Callable[[], CountVectorizer | TfidfVectorizer]:
        """A callable making a fresh, unfitted vectorizer -- CV needs a new one per fold."""
        vectorizers = {"count": CountVectorizer, "tfidf": TfidfVectorizer}
        if kind not in vectorizers:
            raise ValueError(f"Unknown vectorizer {kind!r}; expected one of {sorted(vectorizers)}")
        cls = vectorizers[kind]
        return lambda: cls(ngram_range=self.ngram_range, min_df=self.min_df)

    def describe(self, kind: str) -> str:
        """How the vectorizer is configured, recorded alongside the metrics."""
        name = {"count": "CountVectorizer", "tfidf": "TfidfVectorizer"}[kind]
        return f"{name}(ngram_range={self.ngram_range}, min_df={self.min_df})"
