"""Downloading the review corpus, and reading and writing the project's data files."""

from __future__ import annotations

import json
from pathlib import Path

import kagglehub
import pandas as pd

from src.config import Config


class DataLoader:
    """Fetches the labelled Vietnamese review corpus and handles all file I/O."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.raw_path = config.path("paths.raw_file")
        self.processed_path = config.path("paths.processed_file")
        self.outputs_dir = config.path("paths.outputs_dir")

    def load_raw(self) -> pd.DataFrame:
        """Load the raw reviews, downloading them from Kaggle on the first run.

        A local copy is kept in `data/raw/` so later runs do not depend on the network,
        and so the exact bytes the results came from stay pinned.
        """
        if self.raw_path.exists():
            df = pd.read_csv(self.raw_path)
        else:
            dataset_dir = Path(kagglehub.dataset_download(self.config.require("dataset.kaggle_id")))
            df = pd.read_csv(
                dataset_dir / self.config.require("dataset.file_name"),
                header=None,
                names=self.config.require("dataset.columns"),
            )
            self.raw_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.raw_path, index=False)
        df["comment"] = df["comment"].fillna("")
        return df

    def load_stopwords(self) -> set[str]:
        """Load the Vietnamese stopword list, plus underscore-joined forms of its phrases.

        The list has multi-word entries ("cho nên"); after word segmentation those appear
        as "cho_nên", so both spellings are needed for a lookup to ever match.
        """
        path = kagglehub.dataset_download(
            self.config.require("stopwords.kaggle_id"),
            self.config.require("stopwords.file_name"),
        )
        with open(path, encoding="utf-8") as f:
            words = {line.strip() for line in f if line.strip()}
        return words | {word.replace(" ", "_") for word in words if " " in word}

    def save_processed(self, df: pd.DataFrame) -> Path:
        """Write the cleaned corpus that the model stages read."""
        self.processed_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.processed_path, index=False)
        return self.processed_path

    def load_processed(self) -> pd.DataFrame:
        """Read the cleaned corpus, with a pointed error if it has not been built yet."""
        if not self.processed_path.exists():
            raise FileNotFoundError(
                f"{self.processed_path} is missing. Build it first: python main.py --stage prepare"
            )
        return pd.read_parquet(self.processed_path)

    def save_metrics(self, key: str, results: dict) -> Path:
        """Save one experiment's metrics as JSON, so the comparison stage can re-read it."""
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        path = self.outputs_dir / f"{key}_metrics.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        return path

    def load_metrics(self, key: str) -> dict:
        """Read back one experiment's metrics."""
        path = self.outputs_dir / f"{key}_metrics.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. Run that experiment first: python main.py --stage train"
            )
        with open(path, encoding="utf-8") as f:
            return json.load(f)
