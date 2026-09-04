"""Reading the VNTC corpus, and every artifact the pipeline stages hand to each other.

The corpus is one `.txt` per article in a folder named after its class:

    data/raw/Train_Full/The thao/TT_ VNE_ (100).txt

which is also the layout `data/processed/` mirrors after segmentation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import sparse

from src.config import Config


class DataLoader:
    """Locates corpus files, and saves/loads the intermediate artifacts."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.raw_dir = config.path("paths.raw_dir")
        self.processed_dir = config.path("paths.processed_dir")
        self.outputs_dir = config.path("paths.outputs_dir")
        self.splits: list[str] = config.require("dataset.splits")

    # -- the corpus --------------------------------------------------------------------

    def split_dir(self, split: str, processed: bool = True) -> Path:
        """The folder holding one split, either raw or segmented."""
        return (self.processed_dir if processed else self.raw_dir) / split

    def article_paths(self, split: str, processed: bool = True) -> list[Path]:
        """Every article file in a split, sorted so runs are reproducible."""
        directory = self.split_dir(split, processed)
        if not directory.exists():
            what = "segmented" if processed else "raw"
            raise FileNotFoundError(
                f"{directory} is missing ({what} corpus). "
                + ("Run: python main.py --stage prepare" if processed
                   else "See the README's Data section for how to unpack VNTC.")
            )
        return sorted(directory.rglob("*.txt"))

    def load_split(self, split: str, processed: bool = True) -> tuple[list[str], list[str]]:
        """Read one split as (texts, class names), the class taken from the folder name."""
        encoding = "utf-8" if processed else self.config.require("dataset.raw_encoding")
        paths = self.article_paths(split, processed)
        texts = [p.read_text(encoding=encoding, errors="replace") for p in paths]
        labels = [p.parent.name for p in paths]
        return texts, labels

    def count_articles(self, split: str, processed: bool = True) -> int:
        """How many articles a split holds -- used to verify segmentation lost nothing."""
        directory = self.split_dir(split, processed)
        return sum(1 for _ in directory.rglob("*.txt")) if directory.exists() else 0

    # -- artifacts ---------------------------------------------------------------------

    def save_features(self, X_train, X_test, y_train, y_test, vocabulary: dict) -> None:
        """Cache the TF-IDF matrices, labels and vocabulary so training can start cold."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(self.config.path("paths.features_train"), X_train)
        sparse.save_npz(self.config.path("paths.features_test"), X_test)
        np.save(self.config.path("paths.labels_train"), y_train)
        np.save(self.config.path("paths.labels_test"), y_test)
        self.save_json("paths.vocabulary_file", vocabulary)

    def load_features(self, split: str = "train"):
        """Load one split's TF-IDF matrix and its labels."""
        features = self.config.path(f"paths.features_{split}")
        labels = self.config.path(f"paths.labels_{split}")
        if not features.exists():
            raise FileNotFoundError(
                f"{features} is missing. Run: python main.py --stage features"
            )
        return sparse.load_npz(features), np.load(labels)

    def load_class_names(self) -> list[str]:
        """The label order the encoder used, so a predicted index can be named."""
        return self.load_json("paths.vocabulary_file")["class_names"]

    def save_json(self, path_key: str, payload: dict) -> Path:
        """Write one JSON artifact, creating its folder if needed."""
        path = self.config.path(path_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        return path

    def load_json(self, path_key: str) -> dict:
        """Read one JSON artifact."""
        path = self.config.path(path_key)
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing. Run the earlier stages first.")
        with open(path, encoding="utf-8") as f:
            return json.load(f)
