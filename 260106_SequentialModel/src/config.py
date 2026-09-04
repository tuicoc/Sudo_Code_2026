"""Reads `config/config.yaml`.

Paths in the config are written relative to the project root and resolved here, so `src/`
works the same from `main.py`, a notebook, or a shell.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class Config:
    """A read-only view of `config/config.yaml` with dotted-key lookup."""

    def __init__(self, values: dict[str, Any], project_root: Path = PROJECT_ROOT) -> None:
        self._values = values
        self.project_root = project_root

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Look up a nested value, e.g. `config.get("dataset.kaggle_id")`."""
        node: Any = self._values
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted_key: str) -> Any:
        """Like `get`, but fail loudly instead of returning None on a typo."""
        value = self.get(dotted_key, default=_MISSING)
        if value is _MISSING:
            raise KeyError(f"{dotted_key!r} is not set in {CONFIG_PATH}")
        return value

    def path(self, dotted_key: str) -> Path:
        """A path from the config, resolved against the project root."""
        return (self.project_root / str(self.require(dotted_key))).resolve()

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __repr__(self) -> str:
        return f"Config({', '.join(sorted(self._values))})"


_MISSING = object()


def load_config(path: Path = CONFIG_PATH) -> Config:
    """Read `config/config.yaml`. Call this once and pass the result down."""
    with open(path, encoding="utf-8") as f:
        return Config(yaml.safe_load(f))
