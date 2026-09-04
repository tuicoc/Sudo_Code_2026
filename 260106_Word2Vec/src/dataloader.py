"""Downloading the viwik18 shards and the Vietnamese stopword list."""

from __future__ import annotations

from pathlib import Path

import kagglehub
import requests

from src.config import Config


class DataLoader:
    """Fetches the corpus, one shard at a time, into `data/raw/`.

    Downloads are skipped when the shard is already on disk, so a re-run costs nothing and
    an interrupted download can be resumed by simply running again.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.raw_dir = config.path("paths.raw_dir")
        self.base_url: str = config.require("corpus.base_url")

    @property
    def shard_names(self) -> list[str]:
        """The corpus file names, e.g. `viwik18_aa` ... `viwik18_aj`."""
        prefix = self.config.require("corpus.shard_prefix")
        return [prefix + suffix for suffix in self.config.require("corpus.shard_suffixes")]

    def download_shard(self, name: str) -> Path:
        """Download one shard if it is not already there, and return its path."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        path = self.raw_dir / name
        if path.exists():
            return path
        response = requests.get(
            self.base_url + name,
            timeout=self.config.require("corpus.download_timeout_seconds"),
        )
        response.raise_for_status()
        path.write_bytes(response.content)
        return path

    def fetch_sample(self, name: str | None = None, n_bytes: int | None = None) -> str:
        """Fetch just the first n bytes of a shard, with an HTTP Range request.

        Enough to look at the corpus and check assumptions about it without paying for the
        full ~94 MB download.
        """
        name = name or self.shard_names[0]
        n_bytes = n_bytes or self.config.require("corpus.sample_bytes")
        response = requests.get(
            self.base_url + name,
            headers={"Range": f"bytes=0-{n_bytes}"},
            timeout=self.config.require("corpus.sample_timeout_seconds"),
        )
        response.raise_for_status()
        return response.content.decode("utf-8", errors="ignore")

    def read_shard(self, name: str) -> str:
        """Download the shard if needed, then read it as text."""
        path = self.download_shard(name)
        return path.read_text(encoding="utf-8", errors="ignore")

    def load_stopwords(self) -> set[str]:
        """Load the Vietnamese stopword list (one word or phrase per line)."""
        path = kagglehub.dataset_download(
            self.config.require("stopwords.kaggle_id"),
            self.config.require("stopwords.file_name"),
        )
        with open(path, encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}
