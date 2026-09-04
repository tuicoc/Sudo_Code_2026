"""Reading the corpus this project vectorizes."""

from __future__ import annotations

import pandas as pd

from src.config import Config


class DataLoader:
    """Loads `data/raw/processed_news.parquet`, written by 260106_TextPreprocessingwithNLP.

    The two projects are joined by a file on disk, never by an import.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.input_path = config.path("paths.input_file")
        self.text_column = config.require("data.text_column")

    def load(self) -> pd.DataFrame:
        """Read the corpus, with a pointed error if the upstream project has not run."""
        if not self.input_path.exists():
            upstream = self.config.require("upstream.project")
            command = self.config.require("upstream.command")
            raise FileNotFoundError(
                f"{self.input_path} is missing.\n"
                f"It is produced by {upstream}: run `{command}` there first."
            )
        return pd.read_parquet(self.input_path)

    def load_corpus(self) -> list[str]:
        """Just the text column, as the list of documents a vectorizer expects."""
        return self.load()[self.text_column].tolist()
