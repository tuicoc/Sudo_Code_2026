"""Turns raw viwik18 text into the one-sentence-per-line file gensim trains on.

Two facts about this corpus drive the whole module:

1. It has no punctuation, so there is no period to split sentences on. Runs of 2+ spaces
   are where a paragraph break used to be -- the only sentence boundary available.
2. `underthesea`'s default output returns "tổ chức" with a space, which would re-split
   into 2 tokens once written to a space-joined line. `format="text"` gives "tổ_chức".
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Callable, Iterable

from underthesea import word_tokenize

from src.config import Config


class Preprocessor:
    """Segment text into sentences, then into whole Vietnamese words."""

    def __init__(self, config: Config, stopwords: set[str] | None = None) -> None:
        self.config = config
        self.stopwords = stopwords or set()
        self._segment_boundary = re.compile(config.require("preprocessing.segment_boundary"))

    def split_segments(self, text: str) -> list[str]:
        """Split a shard into sentence-like segments on runs of whitespace."""
        return [segment.strip() for segment in self._segment_boundary.split(text) if segment.strip()]

    def segment_to_tokens(self, segment: str) -> list[str]:
        """Word-segment one sentence and drop stopwords.

        `format="text"` is required -- see this module's docstring.
        """
        tokens = word_tokenize(segment, format="text").split()
        return [token for token in tokens if token not in self.stopwords]

    def build_sentences_file(
        self,
        shard_names: Iterable[str],
        read_shard: Callable[[str], str],
        out_path: Path | None = None,
        progress: bool = True,
    ) -> Path:
        """Write every shard's segments to one sentence-per-line file.

        Streamed shard by shard: the 10 shards are ~1 GB, and gensim reads one line at a time.
        """
        out_path = out_path or self.config.path("paths.sentences_file")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as out:
            for name in shard_names:
                started = time.time()
                written = 0
                for segment in self.split_segments(read_shard(name)):
                    tokens = self.segment_to_tokens(segment)
                    if tokens:
                        out.write(" ".join(tokens) + "\n")
                        written += 1
                if progress:
                    print(f"{name}: {written:,} sentences in {time.time() - started:.1f}s", flush=True)
        return out_path
