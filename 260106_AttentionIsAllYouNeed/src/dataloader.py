"""Getting the corpus in, and every artifact out.

The dataset is downloaded rather than committed: `data/` ships as an empty skeleton and this
class fills it. Nothing else in `src/` touches the filesystem.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config


class DataLoader:
    """Downloads the VNDS parquet files and reads/writes everything under `data/`."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.raw_dir = config.path("paths.raw_dir")
        self.processed_dir = config.path("paths.processed_dir")
        self.outputs_dir = config.path("paths.outputs_dir")
        for directory in (self.raw_dir, self.processed_dir, self.outputs_dir):
            directory.mkdir(parents=True, exist_ok=True)

    # -- the corpus ---------------------------------------------------------------------

    def fetch(self, split: str) -> Path:
        """Download one split into `data/raw/`, once. Returns the local path."""
        import requests

        destination = self.raw_dir / f"vietnews_{split}.parquet"
        if destination.exists():
            return destination

        url = f"{self.config.require('dataset.base_url')}/{self.config.require(f'dataset.files.{split}')}"
        print(f"  downloading {split} ...", end=" ", flush=True)
        try:
            with requests.get(url, stream=True, timeout=180) as response:
                response.raise_for_status()
                with open(destination, "wb") as f:
                    for chunk in response.iter_content(1 << 20):
                        f.write(chunk)
        except Exception as error:
            # A half-written parquet would be read as a corrupt file on the next run, and the
            # error that produces says nothing about the download.
            destination.unlink(missing_ok=True)
            raise RuntimeError(
                f"could not download {split} ({error}).\n"
                "On Kaggle this is almost always the internet switch: Settings -> Internet -> On."
            ) from error
        print(f"{destination.stat().st_size / 1e6:.0f} MB")
        return destination

    def load_split(self, split: str) -> pd.DataFrame:
        """One split, subsampled per the config.

        The subset is a random sample rather than the first n rows: `guid` is sequential, so the
        head of the file is one stretch of the crawl, not a sample of the corpus.
        """
        frame = pd.read_parquet(self.fetch(split))
        n = self.config.get(f"dataset.subset.{split}")
        if not n or n >= len(frame):
            return frame.reset_index(drop=True)
        seed = self.config.require("training.seed")
        return frame.sample(n, random_state=seed).reset_index(drop=True)

    # -- artifacts ----------------------------------------------------------------------

    def save_json(self, path_key: str, payload: dict) -> Path:
        path = self.config.path(path_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        return path

    def load_json(self, path_key: str) -> dict:
        path = self.config.path(path_key)
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing. Run: python main.py --stage prepare")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def save_encoded(self, arrays: dict[str, np.ndarray]) -> Path:
        path = self.config.path("paths.encoded_file")
        np.savez_compressed(path, **arrays)
        return path

    def load_encoded(self) -> dict[str, np.ndarray]:
        path = self.config.path("paths.encoded_file")
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing. Run: python main.py --stage prepare")
        with np.load(path) as data:
            return {key: data[key] for key in data.files}

    def weights_path(self, variant: str) -> Path:
        return self.outputs_dir / f"{variant}_attention.weights.h5"
