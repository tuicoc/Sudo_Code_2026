"""The 5 preprocessing steps from NLTK book chapter 3, applied to Vietnamese news.

    1. normalize_unicode   NFC
    2. clean_text          remove URLs, emails, phones, domains, codes, digits, punctuation
    3. fold_case           lowercase, then expand abbreviations
    4. tokenize            split into word tokens
    5. remove_stopwords    drop words with no topic signal

The order matters: cleaning before case folding keeps the uppercase-only reference-code
pattern working, and case folding before tokenizing keeps the teencode map lowercase-only.
"""

from __future__ import annotations

import re
import string
import unicodedata

import nltk
import pandas as pd
from nltk.tokenize import word_tokenize

from src.config import Config


class Preprocessor:
    """Raw article text -> a clean token list. All regexes come from config.yaml."""

    def __init__(self, config: Config, stopwords: set[str] | None = None) -> None:
        self.config = config
        self.stopwords = stopwords or set()
        self.teencode: dict[str, str] = config.require("preprocessing.teencode")

        patterns = config.require("patterns")
        self._email = re.compile(patterns["email"])
        self._url = re.compile(patterns["url"])
        self._url_loose = re.compile(patterns["url_loose"])
        self._phone = re.compile(patterns["phone"])
        self._cdata = re.compile(patterns["cdata"], re.DOTALL)
        self._domain = re.compile(patterns["domain"], re.IGNORECASE)
        self._reference_code = re.compile(patterns["reference_code"])
        self._html_tag = re.compile(patterns["html_tag"])
        self._digits = re.compile(patterns["digits"])
        self._whitespace = re.compile(patterns["whitespace_run"])

        extra = config.require("preprocessing.extra_punctuation")
        self._punctuation = re.compile(f"[{re.escape(string.punctuation)}{extra}]")

        nltk.download(config.require("preprocessing.nltk_tokenizer_model"), quiet=True)

    def scan_noise(self, text: str) -> dict[str, int]:
        """Count each kind of noise. Run this first -- only clean what the corpus has."""
        return {
            "email": len(self._email.findall(text)),
            "url": len(self._url.findall(text)),
            "phone": len(self._phone.findall(text)),
            "cdata": len(self._cdata.findall(text)),
            "domain": len(self._domain.findall(text)),
            "reference_code": len(self._reference_code.findall(text)),
        }

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Canonicalize to NFC so precomposed and decomposed diacritics compare equal."""
        return unicodedata.normalize("NFC", str(text))

    def clean_text(self, text: str) -> str:
        """Remove noise, digits and punctuation.

        Each match becomes a space, not "", so neighbouring words are not glued together.
        """
        text = self._cdata.sub(" ", text)
        text = self._url_loose.sub(" ", text)
        text = self._html_tag.sub(" ", text)
        # Emails before bare domains, or the domain half of an address is left behind.
        text = self._email.sub(" ", text)
        text = self._domain.sub(" ", text)
        text = self._reference_code.sub(" ", text)
        text = self._digits.sub(" ", text)
        text = self._punctuation.sub(" ", text)
        return self._whitespace.sub(" ", text).strip()

    def fold_case(self, text: str) -> str:
        """Lowercase, then expand the abbreviations listed in the config."""
        # Pad with spaces so a slang word at the very start or end of the string still
        # has neighbours to match against. `clean_text` already turned every punctuation
        # mark into a space, so every word boundary here is a space.
        padded = f" {text.lower()} "
        for abbreviation, full_form in self.teencode.items():
            padded = padded.replace(abbreviation, full_form)
        return padded.strip()

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Split into word tokens.

        NLTK is syllable-level on Vietnamese: "sản phẩm" stays 2 tokens. Accepted here;
        see 260106_Word2Vec for the project where it is not good enough.
        """
        return word_tokenize(text)

    def remove_stopwords(self, tokens: list[str]) -> list[str]:
        """Drop tokens that carry little topical information."""
        return [token for token in tokens if token not in self.stopwords]

    def process_text(self, text: str) -> list[str]:
        """Run one document through all five stages."""
        cleaned = self.fold_case(self.clean_text(self.normalize_unicode(text)))
        return self.remove_stopwords(self.tokenize(cleaned))

    def process_dataframe(self, df: pd.DataFrame, column: str = "content") -> pd.DataFrame:
        """Run the corpus through the pipeline, one column per stage so it stays inspectable."""
        df = df.copy()
        df[column] = df[column].fillna("").apply(self.normalize_unicode)
        df[column] = df[column].apply(self.clean_text)
        df[column] = df[column].apply(self.fold_case)
        df["content_tokens"] = df[column].apply(self.tokenize)
        df["content_tokenized"] = df["content_tokens"].apply(" ".join)
        df["content_tokens"] = df["content_tokens"].apply(self.remove_stopwords)
        df["content_no_stopwords"] = df["content_tokens"].apply(" ".join)
        return df

    @staticmethod
    def build_vocabulary(token_lists: "pd.Series[list[str]]") -> list[str]:
        """Collect the tokens into the corpus vocabulary."""
        return sorted({token for tokens in token_lists for token in tokens})
