"""Bag-of-Words counts, and the TF-IDF weights built on top of them.

Kept as two steps (`CountVectorizer` then `TfidfTransformer`) instead of one
`TfidfVectorizer`, because the raw count is what makes the weight readable.
`breakdown_table` shows count, IDF and TF-IDF side by side.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from src.config import Config


@dataclass
class Features:
    """One fitted vectorization: the two fitted objects and the two matrices."""

    name: str
    ngram_range: tuple[int, int]
    count_vectorizer: CountVectorizer
    counts: csr_matrix
    tfidf_transformer: TfidfTransformer
    tfidf: csr_matrix

    @property
    def n_documents(self) -> int:
        return self.counts.shape[0]

    @property
    def vocabulary_size(self) -> int:
        return self.counts.shape[1]

    @property
    def sparsity(self) -> float:
        """Fraction of the document-term matrix that is zero."""
        return 1 - self.counts.nnz / (self.n_documents * self.vocabulary_size)


class FeatureExtractor:
    """Turns a list of documents into TF-IDF features, and explains the result."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.min_df: int = config.require("vectorizer.min_df")
        self.ngram_ranges: dict[str, list[int]] = config.require("vectorizer.ngram_ranges")

    def fit(self, corpus: list[str], ngram_range: tuple[int, int], name: str = "") -> Features:
        """Fit Bag-of-Words, then TF-IDF on top of it, for one n-gram range."""
        count_vectorizer = CountVectorizer(min_df=self.min_df, ngram_range=ngram_range)
        counts = count_vectorizer.fit_transform(corpus)

        tfidf_transformer = TfidfTransformer()
        tfidf = tfidf_transformer.fit_transform(counts)

        return Features(
            name=name or f"ngram{ngram_range}",
            ngram_range=ngram_range,
            count_vectorizer=count_vectorizer,
            counts=counts,
            tfidf_transformer=tfidf_transformer,
            tfidf=tfidf,
        )

    def fit_all(self, corpus: list[str]) -> dict[str, Features]:
        """Fit every n-gram range listed in the config, keyed by its config name."""
        return {
            name: self.fit(corpus, ngram_range=tuple(bounds), name=name)
            for name, bounds in self.ngram_ranges.items()
        }

    def breakdown_table(self, features: Features, row: int | None = None,
                        top_n: int | None = None) -> pd.DataFrame:
        """One document, term by term: count, IDF, and the TF-IDF they produce."""
        row = self.config.require("report.example_row") if row is None else row
        top_n = self.config.require("report.top_n") if top_n is None else top_n

        table = pd.DataFrame({
            "Term": features.count_vectorizer.get_feature_names_out(),
            "BoW Count": features.counts[row].toarray().ravel(),
            "IDF": features.tfidf_transformer.idf_,
            "TF-IDF": features.tfidf[row].toarray().ravel(),
        })
        return table[table["BoW Count"] > 0].sort_values("TF-IDF", ascending=False).head(top_n)

    @staticmethod
    def summary(fitted: dict[str, Features]) -> pd.DataFrame:
        """One row per n-gram range: vocabulary size and sparsity."""
        return pd.DataFrame(
            [
                {
                    "ngram_range": f.ngram_range,
                    "documents": f.n_documents,
                    "vocabulary_size": f.vocabulary_size,
                    "sparsity": f.sparsity,
                }
                for f in fitted.values()
            ],
            index=list(fitted),
        )
