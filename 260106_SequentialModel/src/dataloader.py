"""Locating the book corpus, and reading/writing every artifact the stages exchange."""

from __future__ import annotations

import collections
import glob
import json
from pathlib import Path

import kagglehub
import numpy as np

from src.config import Config


class DataLoader:
    """Finds the 10,415 book files, and handles all of the project's file I/O."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.processed_dir = config.path("paths.processed_dir")
        self.outputs_dir = config.path("paths.outputs_dir")

    def find_books(self, root: Path | None = None) -> list[Path]:
        """Every book `.txt`, sorted.

        kagglehub nests the download behind a version folder whose name has changed
        between releases, so the directory holding the most `.txt` files is searched for
        rather than hardcoded.
        """
        root = root or Path(kagglehub.dataset_download(self.config.require("dataset.kaggle_id")))
        pattern = self.config.require("dataset.books_glob")
        hits = [Path(p) for p in glob.glob(f"{root}/**/{pattern}", recursive=True)]
        if not hits:
            raise FileNotFoundError(f"No {pattern} files found under {root}")
        corpus_dir = collections.Counter(p.parent for p in hits).most_common(1)[0][0]
        return sorted(p for p in hits if p.parent == corpus_dir)

    @staticmethod
    def read_book(path: Path) -> str:
        """Read one book, replacing anything that will not decode rather than failing."""
        return path.read_text(encoding="utf-8", errors="replace")

    # -- artifacts ---------------------------------------------------------------------

    def save_tokens(self, train_ids: np.ndarray, val_ids: np.ndarray) -> None:
        """Cache the two token streams as `uint16` -- see `corpus.py` for why not strings."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.config.path("paths.train_tokens"), train_ids)
        np.save(self.config.path("paths.val_tokens"), val_ids)

    def load_tokens(self) -> tuple[np.ndarray, np.ndarray]:
        """Load the cached train and validation token streams."""
        train_path = self.config.path("paths.train_tokens")
        if not train_path.exists():
            raise FileNotFoundError(
                f"{train_path} is missing. Run: python main.py --stage prepare"
            )
        return np.load(train_path), np.load(self.config.path("paths.val_tokens"))

    def save_json(self, path_key: str, payload: dict) -> Path:
        path = self.config.path(path_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        return path

    def load_json(self, path_key: str) -> dict:
        path = self.config.path(path_key)
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing. Run the earlier stages first.")
        with open(path, encoding="utf-8") as f:
            return json.load(f)
