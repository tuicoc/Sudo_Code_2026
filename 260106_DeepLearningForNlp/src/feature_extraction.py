"""TF-IDF features over the segmented corpus.

The custom tokenizer is the point: sklearn's default `token_pattern` splits on the
underscore that holds "xuất_khẩu" together, undoing segmentation immediately.
`vn_tokenizer` splits on whitespace, keeps `_`, drops punctuation, and drops tokens with
digits (dates and prices are noise for a topic classifier).

`strip_accents` must stay None -- stripping accents merges Vietnamese words that differ
only by diacritic into one feature.
"""

from __future__ import annotations

import re
import string

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from src.config import Config

_HAS_DIGIT = re.compile(r"\d")


class FeatureExtractor:
    """Builds the TF-IDF matrices and the label encoding the model trains on."""

    def __init__(self, config: Config) -> None:
        self.config = config
        extra = config.require("features.extra_punctuation")
        # Everything to strip, minus "_": the underscore is what makes "xuất_khẩu" one token.
        self.punctuation = (set(string.punctuation) | set(extra)) - {"_"}
        self.vectorizer: TfidfVectorizer | None = None
        self.label_encoder: LabelEncoder | None = None

    def vn_tokenizer(self, document: str) -> list[str]:
        """Split on whitespace, strip punctuation, keep `_`, drop tokens containing digits."""
        tokens = []
        for token in document.split():
            token = "".join(c for c in token if c not in self.punctuation).strip("_")
            if token and not _HAS_DIGIT.search(token):
                tokens.append(token)
        return tokens

    def build_vectorizer(self) -> TfidfVectorizer:
        """A vectorizer configured for segmented Vietnamese."""
        return TfidfVectorizer(
            tokenizer=self.vn_tokenizer,
            # `token_pattern=None` silences the warning sklearn raises when a custom
            # tokenizer makes the default pattern dead configuration.
            token_pattern=None,
            # `.lower()` is diacritic-safe: XUẤT_KHẨU -> xuất_khẩu.
            lowercase=True,
            strip_accents=self.config.get("features.strip_accents"),
            max_features=self.config.require("features.max_features"),
            min_df=self.config.require("features.min_df"),
        )

    def fit_transform(self, train_texts: list[str], test_texts: list[str]
                      ) -> tuple[csr_matrix, csr_matrix]:
        """Fit on train only, then transform both splits -- fitting on test would leak."""
        self.vectorizer = self.build_vectorizer()
        X_train = self.vectorizer.fit_transform(train_texts).astype("float32")
        X_test = self.vectorizer.transform(test_texts).astype("float32")
        return X_train, X_test

    def encode_labels(self, train_labels: list[str], test_labels: list[str]
                      ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Class-name strings -> integer ids, plus the ordered class names."""
        self.label_encoder = LabelEncoder().fit(train_labels)
        return (
            self.label_encoder.transform(train_labels),
            self.label_encoder.transform(test_labels),
            list(self.label_encoder.classes_),
        )

    def vocabulary_payload(self, class_names: list[str]) -> dict:
        """What gets cached to `vocabulary.json`: class names, feature names and IDF."""
        vectorizer = self._require_vectorizer()
        return {
            "class_names": class_names,
            "feature_names": vectorizer.get_feature_names_out().tolist(),
            "idf": vectorizer.idf_.tolist(),
        }

    def describe_matrix(self, X: csr_matrix) -> str:
        """One line of numbers showing why sparse storage is not optional."""
        cells = X.shape[0] * X.shape[1]
        dense_mb = cells * 4 / 1e6
        sparse_mb = (X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) / 1e6
        return (
            f"{X.nnz:,} non-zero of {cells:,} cells ({100 * X.nnz / cells:.2f}% full)  "
            f"dense {dense_mb:,.0f} MB vs sparse {sparse_mb:,.0f} MB "
            f"({dense_mb / sparse_mb:.0f}x smaller)"
        )

    def compound_feature_share(self) -> tuple[int, int]:
        """How many features are segmented compounds -- i.e. what segmentation bought."""
        names = self._require_vectorizer().get_feature_names_out()
        return sum("_" in name for name in names), len(names)

    def _require_vectorizer(self) -> TfidfVectorizer:
        if self.vectorizer is None:
            raise RuntimeError("No vectorizer yet: call fit_transform() first.")
        return self.vectorizer
