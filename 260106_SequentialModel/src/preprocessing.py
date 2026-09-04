"""Cleans and tokenizes the book corpus.

Every choice here is the opposite of the classification projects, because this model has
to *produce* text, not label it:

* Punctuation is kept as its own token -- a model that cannot emit a comma emits a word list.
* Digits are kept, for the same reason.
* Stopwords stay -- "và", "là", "của" are most of what fluent Vietnamese is made of.

The rules come from surveying a 150-book sample: HTML entities dominate (~24,000), while
URLs (333), domains (25), emails (2) and phones (1) are rare, and there are no HTML tags
or control characters at all. So `html.unescape` does the heaviest lifting.
"""

from __future__ import annotations

import html
import re
import unicodedata

from src.config import Config


class Preprocessor:
    """Book text -> the token list the vocabulary is built from."""

    def __init__(self, config: Config) -> None:
        self.config = config
        patterns = config.require("patterns")
        self._url = re.compile(patterns["url"])
        self._email = re.compile(patterns["email"])
        self._phone = re.compile(patterns["phone"])
        self._domain = re.compile(patterns["domain"], re.IGNORECASE)
        self._html_tag = re.compile(patterns["html_tag"])
        self._html_entity = re.compile(patterns["html_entity"])
        self._control_char = re.compile(patterns["control_char"])
        self._whitespace = re.compile(patterns["whitespace_run"])
        self._token = re.compile(patterns["token"], re.UNICODE)

    def scan_noise(self, text: str) -> dict[str, int]:
        """Count each kind of noise, so the cleaning rules follow evidence."""
        return {
            "URL": len(self._url.findall(text)),
            "email": len(self._email.findall(text)),
            "HTML tag": len(self._html_tag.findall(text)),
            "HTML entity": len(self._html_entity.findall(text)),
            "phone (VN)": len(self._phone.findall(text)),
            "bare domain": len(self._domain.findall(text)),
            "control char": len(self._control_char.findall(text)),
        }

    def clean_text(self, text: str) -> str:
        """Unescape entities, strip the web noise, normalize to NFC, lowercase."""
        text = html.unescape(text)          # &amp; -> &   (the ~24,000-occurrence problem)
        text = self._html_tag.sub(" ", text)
        # Emails before domains, or half of each address survives as a bare domain.
        text = self._email.sub(" ", text)
        text = self._url.sub(" ", text)
        text = self._domain.sub(" ", text)
        text = self._phone.sub(" ", text)
        text = unicodedata.normalize("NFC", text)
        return self._whitespace.sub(" ", text.lower()).strip()

    def tokenize(self, text: str) -> list[str]:
        """Split into word / number / punctuation tokens, each punctuation mark its own."""
        return self._token.findall(text)

    def process_text(self, text: str) -> list[str]:
        """Clean then tokenize one book."""
        return self.tokenize(self.clean_text(text))
