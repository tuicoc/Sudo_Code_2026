"""Everything that moves data in or out of the project: downloads, reads, writes."""

from __future__ import annotations

from pathlib import Path

import kagglehub
import pandas as pd

from src.config import Config


class DataLoader:
    """Downloads the news corpus and the stopword list from Kaggle, and saves the output.

    Nothing large is committed: a fresh clone downloads what it needs on the first run.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.raw_dir = config.path("paths.raw_dir")

    def download_articles(self) -> Path:
        """Download the news dataset into `data/raw/` and return the file path."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        return Path(
            kagglehub.dataset_download(
                self.config.require("dataset.kaggle_id"),
                self.config.require("dataset.file_name"),
                output_dir=str(self.raw_dir),
            )
        )

    def load_articles(self) -> pd.DataFrame:
        """Load the news dataset as a DataFrame, downloading it if it is not there yet."""
        local_copy = self.raw_dir / self.config.require("dataset.file_name")
        path = local_copy if local_copy.exists() else self.download_articles()
        df = pd.read_json(path)
        df["content"] = df["content"].fillna("")
        df["title"] = df["title"].fillna("")
        return df

    def load_stopwords(self) -> set[str]:
        """Load the Vietnamese stopword list (one word or phrase per line)."""
        path = kagglehub.dataset_download(
            self.config.require("stopwords.kaggle_id"),
            self.config.require("stopwords.file_name"),
        )
        with open(path, encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}

    def save_processed(self, df: pd.DataFrame) -> Path:
        """Write the parquet that 260106_Scikit-learnTextFeatureExtraction reads.

        A file handover, never an import -- see the repo README, "Cross-project data flow".
        """
        export_path = self.config.path("paths.export_file")
        export_path.parent.mkdir(parents=True, exist_ok=True)
        df[self.config.require("export.columns")].to_parquet(export_path, index=False)
        return export_path
