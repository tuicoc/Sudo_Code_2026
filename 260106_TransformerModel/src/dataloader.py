"""Getting EVBNews in, and every artifact out.

The corpus is downloaded rather than committed. It arrives as a `.rar`, which has no pure-Python
reader -- `rarfile` is a wrapper and still needs a backend -- so `extract` tries the tools that
tend to exist and installs one if none does.
"""

from __future__ import annotations

import json
import random
import re
import subprocess
from pathlib import Path

import numpy as np

from src.config import Config

SPAIR = re.compile(r"<spair id='\d+'>(.*?)</spair>", re.S)
EN_SENTENCE = re.compile(r"<s id='en\d+'>(.*?)</s>", re.S)
VI_SENTENCE = re.compile(r"<s id='vn\d+'>(.*?)</s>", re.S)
ALIGNMENT = re.compile(r"<a id='ev\d+'>(.*?)</a>", re.S)


class DataLoader:
    """Downloads and unpacks EVBNews, parses its SGML, and reads/writes everything in `data/`."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.raw_dir = config.path("paths.raw_dir")
        self.processed_dir = config.path("paths.processed_dir")
        self.outputs_dir = config.path("paths.outputs_dir")
        self.corpus_dir = config.path("dataset.corpus_dir")
        for directory in (self.raw_dir, self.processed_dir, self.outputs_dir):
            directory.mkdir(parents=True, exist_ok=True)

    # -- the corpus ---------------------------------------------------------------------

    def download(self) -> Path:
        import requests

        destination = self.raw_dir / self.config.require("dataset.archive_name")
        if destination.exists():
            return destination
        url = self.config.require("dataset.archive_url")
        print(f"  downloading {destination.name} ...", end=" ", flush=True)
        try:
            with requests.get(url, stream=True, timeout=180) as response:
                response.raise_for_status()
                with open(destination, "wb") as f:
                    for chunk in response.iter_content(1 << 20):
                        f.write(chunk)
        except Exception as error:
            destination.unlink(missing_ok=True)
            raise RuntimeError(f"could not download {url} ({error})") from error
        print(f"{destination.stat().st_size / 1e6:.1f} MB")
        return destination

    def extract(self, archive: Path) -> str:
        """Unpack the archive with whatever rar tool this machine has.

        macOS `tar` is bsdtar and reads RAR through libarchive; GNU `tar` on Linux does not.
        `unar` is the one packaged everywhere, so it is what gets installed as a fallback.
        """
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        if list(self.corpus_dir.glob("*.sgml")):
            return "nothing to do, already unpacked"

        commands = [["tar", "-xf", str(archive), "-C", str(self.corpus_dir)],
                    ["bsdtar", "-xf", str(archive), "-C", str(self.corpus_dir)],
                    ["unar", "-q", "-f", "-o", str(self.corpus_dir), str(archive)],
                    ["7z", "x", f"-o{self.corpus_dir}", "-y", str(archive)],
                    ["unrar", "x", "-y", str(archive), f"{self.corpus_dir}/"]]
        for attempt in range(2):
            for command in commands:
                try:
                    subprocess.run(command, check=True, capture_output=True)
                except Exception:
                    continue
                if list(self.corpus_dir.glob("*.sgml")):
                    return f"unpacked with {command[0]}"
            if attempt == 0:
                print("  no rar tool found, installing one ...")
                for install in (["apt-get", "-qq", "install", "-y", "unar"],
                                ["apt-get", "-qq", "install", "-y", "libarchive-tools"]):
                    try:
                        subprocess.run(install, check=True, capture_output=True, timeout=300)
                    except Exception:
                        continue
        raise RuntimeError("could not unpack the .rar. Install one of unar / libarchive-tools / "
                           "p7zip and try again:\n    apt-get install -y unar")

    @staticmethod
    def parse_document(path: Path) -> list[tuple[str, str, str]]:
        """Every (english, vietnamese, alignment) triple in one SGML file."""
        text = path.read_text(encoding="utf-8", errors="replace")
        rows = []
        for block in SPAIR.findall(text):
            english, vietnamese = EN_SENTENCE.search(block), VI_SENTENCE.search(block)
            alignment = ALIGNMENT.search(block)
            if english and vietnamese:
                rows.append((english.group(1).strip(), vietnamese.group(1).strip(),
                             alignment.group(1).strip() if alignment else ""))
        return rows

    def load_splits(self) -> dict[str, list[tuple[str, str, str]]]:
        """Whole documents held out, shuffled by a fixed seed."""
        documents = sorted(self.corpus_dir.glob("*.sgml"))
        if not documents:
            raise FileNotFoundError(f"no .sgml under {self.corpus_dir}. "
                                    "Run: python main.py --stage prepare")
        random.Random(self.config.require("dataset.split_seed")).shuffle(documents)
        n_test = self.config.require("dataset.test_docs")
        n_val = self.config.require("dataset.val_docs")
        train = documents[n_test + n_val:]
        cap = self.config.get("dataset.max_train_docs")
        groups = {"test": documents[:n_test],
                  "validation": documents[n_test:n_test + n_val],
                  "train": train[:cap] if cap else train}
        return {name: [row for path in paths for row in self.parse_document(path)]
                for name, paths in groups.items()}

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
