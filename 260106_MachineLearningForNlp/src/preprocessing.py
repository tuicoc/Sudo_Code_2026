"""Cleans Vietnamese product reviews for sentiment classification.

Gentler than the news pipeline, because the task is different. Two things that look like
omissions and are not:

- Stopwords are NOT removed. The list contains "không", "chưa", "rất", "quá" -- the words
  that flip or scale sentiment. `stopwords_that_would_be_lost()` prints them.
- Run-together words ("sảnphẩm") are left alone. No rule separates the typo from a real
  brand name like "iPhone" without breaking the brand name too.
"""

from __future__ import annotations

import re
import string
import unicodedata

import pandas as pd
from underthesea import word_tokenize

from src.config import Config

# Negation and intensifier words this corpus cannot afford to lose.
SENTIMENT_CARRYING_STOPWORDS = ("không", "chưa", "rất", "quá")


class Preprocessor:
    """Raw review text -> the space-joined token string the vectorizers consume."""

    def __init__(self, config: Config, stopwords: set[str] | None = None) -> None:
        self.config = config
        self.stopwords = stopwords or set()
        self.teencode: dict[str, str] = config.require("preprocessing.teencode")
        self.remove_stopwords_enabled: bool = config.require("preprocessing.remove_stopwords")

        patterns = config.require("patterns")
        self._url = re.compile(patterns["url"])
        self._digits = re.compile(patterns["digits"])
        self._whitespace = re.compile(patterns["whitespace_run"])

        extra = config.require("preprocessing.extra_punctuation")
        self._punctuation = re.compile(f"[{re.escape(string.punctuation)}{extra}]")

    # -- the stages --------------------------------------------------------------------

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Canonicalize to NFC so precomposed and decomposed diacritics compare equal."""
        return unicodedata.normalize("NFC", str(text))

    def clean_text(self, text: str) -> str:
        """Strip URLs, digits and punctuation, replacing each with a space."""
        text = self._url.sub(" ", text)
        text = self._digits.sub(" ", text)
        text = self._punctuation.sub(" ", text)
        return self._whitespace.sub(" ", text).strip()

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Word-segment with underthesea. `format="text"` gives "sản_phẩm" as one token."""
        return word_tokenize(text, format="text").split()

    def expand_teencode(self, tokens: list[str]) -> list[str]:
        """Expand abbreviations, token by token.

        After tokenizing, not before: a string replace would hit "vs" inside longer words.
        """
        return [self.teencode.get(token, token) for token in tokens]

    def drop_stopwords(self, tokens: list[str]) -> list[str]:
        """Available, but off by default -- see this module's docstring."""
        return [token for token in tokens if token not in self.stopwords]

    # -- whole-document and whole-corpus entry points ----------------------------------

    def process_text(self, text: str) -> list[str]:
        """Run one review through the pipeline."""
        cleaned = self.clean_text(self.normalize_unicode(text)).lower()
        tokens = self.expand_teencode(self.tokenize(cleaned))
        return self.drop_stopwords(tokens) if self.remove_stopwords_enabled else tokens

    def process_dataframe(self, df: pd.DataFrame, column: str = "comment") -> pd.DataFrame:
        """Add `tokens` and `clean_comment` columns to the raw review DataFrame."""
        df = df.copy()
        df["tokens"] = df[column].apply(self.process_text)
        df["clean_comment"] = df["tokens"].apply(" ".join)
        return df

    def stopwords_that_would_be_lost(self) -> list[str]:
        """The sentiment words the stopword list would delete -- evidence for not using it."""
        return [word for word in SENTIMENT_CARRYING_STOPWORDS if word in self.stopwords]
