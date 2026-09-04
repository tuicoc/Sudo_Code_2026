"""Turns 10,415 books into two `uint16` token streams.

Two passes, and the reason is a bug worth not repeating. The obvious single-pass version
keeps each book's tokens so the files are read once -- but 334M tokens as Python strings is
~19.8 GB (a `str` costs ~59 bytes of header; the characters are almost incidental). As
`uint16` the same data is 0.67 GB. On a 12.7 GB machine the single-pass version dies around
book 5,000, and the crash looks like a batch-size problem.

So: pass 1 counts tokens to build the vocabulary, pass 2 re-reads and encodes to `uint16`.
Reading each book twice is far cheaper than holding the strings.

The split is by book, not by token offset -- a book split down the middle would put its own
style on both sides and flatter the validation loss.
"""

from __future__ import annotations

import collections
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from src.config import Config
from src.preprocessing import Preprocessor
from src.vocabulary import UNKNOWN_ID, Vocabulary

# Set once per worker process by `_init_worker`, so each book task does not have to carry
# a pickled copy of the preprocessor and vocabulary with it.
_PREPROCESSOR: Preprocessor | None = None
_VOCABULARY: Vocabulary | None = None


def _init_worker(config: Config, vocabulary: Vocabulary | None) -> None:
    global _PREPROCESSOR, _VOCABULARY
    _PREPROCESSOR = Preprocessor(config)
    _VOCABULARY = vocabulary


def count_book(path: Path) -> collections.Counter:
    """Pass one: token counts for a single book."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return collections.Counter(_PREPROCESSOR.process_text(text))


def encode_book(path: Path) -> np.ndarray:
    """Pass two: one book as `uint16` ids, terminated by `<eob>`."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return _VOCABULARY.encode_book(_PREPROCESSOR.process_text(text))


class CorpusBuilder:
    """Runs both passes over the books and produces the train/validation token streams."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.workers: int = config.require("preprocessing.workers")
        self.chunk_size: int = config.require("preprocessing.chunk_size")
        self.report_every: int = config.require("preprocessing.report_every")

    def _pool(self, vocabulary: Vocabulary | None = None) -> ProcessPoolExecutor:
        # `fork`, not macOS's default `spawn`: forked workers inherit the already-imported
        # modules instead of each paying the import cost again.
        return ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=mp.get_context("fork"),
            initializer=_init_worker,
            initargs=(self.config, vocabulary),
        )

    def count_tokens(self, books: list[Path], progress: bool = True) -> collections.Counter:
        """Pass one: count every token in the corpus, in parallel."""
        started = time.time()
        counts: collections.Counter = collections.Counter()
        with self._pool() as pool:
            for i, book_counts in enumerate(pool.map(count_book, books, chunksize=self.chunk_size), 1):
                counts.update(book_counts)
                if progress and i % self.report_every == 0:
                    print(f"  {i:,}/{len(books):,}  {time.time() - started:.0f}s")
        if progress:
            print(f"  {sum(counts.values()):,} tokens, {len(counts):,} distinct "
                  f"({time.time() - started:.0f}s)")
        return counts

    def encode_books(self, books: list[Path], vocabulary: Vocabulary,
                     progress: bool = True) -> list[np.ndarray]:
        """Pass two: encode every book to `uint16` ids, in parallel."""
        started = time.time()
        with self._pool(vocabulary) as pool:
            encoded = list(pool.map(encode_book, books, chunksize=self.chunk_size))
        if progress:
            print(f"  encoded {len(encoded):,} books in {time.time() - started:.0f}s")
        return encoded

    def split(self, encoded: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Concatenate the encoded books into a train and a validation stream.

        Whole books go to one side or the other, so no book's own vocabulary appears on
        both sides of the split.
        """
        rng = np.random.default_rng(self.config.require("corpus.split_seed"))
        order = rng.permutation(len(encoded))
        n_val = int(self.config.require("corpus.val_fraction") * len(order))
        val_ids = np.concatenate([encoded[i] for i in order[:n_val]])
        train_ids = np.concatenate([encoded[i] for i in order[n_val:]])
        return train_ids, val_ids

    @staticmethod
    def describe(train_ids: np.ndarray, val_ids: np.ndarray) -> str:
        """The sanity checks worth printing after a build."""
        total = train_ids.size + val_ids.size
        return (
            f"train {train_ids.size:,} tokens, val {val_ids.size:,} tokens "
            f"({100 * val_ids.size / total:.1f}% val)\n"
            f"  <unk> rate {100 * (train_ids == UNKNOWN_ID).mean():.2f}%\n"
            f"  index 0 never used: {(train_ids == 0).sum() == 0 and (val_ids == 0).sum() == 0}"
        )
