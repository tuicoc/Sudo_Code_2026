"""Converts the raw VNTC corpus into text the rest of the pipeline can read.

Three problems, and the first is why this module exists:

1. The files are UTF-16LE. Reading them as UTF-8 does NOT raise -- it returns the text with
   a null byte after every character, a vectorizer builds a vocabulary out of it, training
   runs, and the model learns nothing. The encoding is pinned in the config for that reason.
2. Line endings are CRLF. `open(..., encoding="utf-16")` handles the BOM and newlines in
   one call, which is why text mode is used instead of a manual `.decode()`.
3. Vietnamese needs word segmentation: "xuất khẩu" -> the single token "xuất_khẩu".

Segmentation is the slow step, so it runs once and caches to `data/processed/`. Resumable:
a file whose output already exists is skipped.
"""

from __future__ import annotations

import multiprocessing as mp
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from src.config import Config


def convert_article(job: tuple[Path, Path, str]) -> int:
    """Decode, NFC-normalize, word-segment, write as UTF-8. Returns 1 if it worked, 0 if skipped.

    Module level, not a method, because `ProcessPoolExecutor` has to pickle it.
    """
    source, destination, encoding = job
    if destination.exists():
        return 0

    text = source.read_text(encoding=encoding)          # BOM and CRLF handled here
    text = unicodedata.normalize("NFC", text)

    # Imported inside the worker: underthesea loads its models on import, and doing that
    # once per process is what keeps the pool from paying for it on every task.
    from underthesea import word_tokenize

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(word_tokenize(text, format="text"), encoding="utf-8")
    return 1


class Preprocessor:
    """Runs the raw -> segmented conversion over the whole corpus, in parallel."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.raw_dir = config.path("paths.raw_dir")
        self.processed_dir = config.path("paths.processed_dir")
        self.encoding: str = config.require("dataset.raw_encoding")
        self.workers: int = config.require("preprocessing.workers")
        self.report_every: int = config.require("preprocessing.report_every")

    def build_jobs(self, splits: list[str]) -> list[tuple[Path, Path, str]]:
        """One (source, destination, encoding) per article, across every split."""
        jobs = []
        for split in splits:
            split_dir = self.raw_dir / split
            if not split_dir.exists():
                raise FileNotFoundError(
                    f"{split_dir} is missing. See the README's Data section for how to unpack VNTC."
                )
            for class_dir in sorted(split_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
                for source in class_dir.glob("*.txt"):
                    jobs.append(
                        (source, self.processed_dir / split / class_dir.name / source.name,
                         self.encoding)
                    )
        return jobs

    def run(self, splits: list[str] | None = None, progress: bool = True) -> int:
        """Segment every article that has not been segmented yet. Returns the count done."""
        splits = splits or self.config.require("dataset.splits")
        jobs = self.build_jobs(splits)
        todo = [job for job in jobs if not job[1].exists()]
        if progress:
            print(f"{len(jobs):,} articles total, {len(todo):,} still to convert")
        if not todo:
            return 0

        started = time.time()
        done = 0
        # `fork`, not macOS's default `spawn`: forked workers inherit the already-imported
        # underthesea models instead of each re-importing them.
        context = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=self.workers, mp_context=context) as pool:
            for result in pool.map(convert_article, todo, chunksize=64):
                done += result
                if progress and done and done % self.report_every == 0:
                    rate = done / (time.time() - started)
                    remaining = (len(todo) - done) / rate / 60
                    print(f"  {done:,}/{len(todo):,}  {rate:.0f} files/s  ETA {remaining:.1f} min")

        if progress:
            print(f"converted {done:,} files in {(time.time() - started) / 60:.1f} min")
        return done
