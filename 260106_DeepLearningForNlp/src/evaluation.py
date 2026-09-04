"""Scores the model, and the plots that make the score readable.

"0.9266 accuracy" means nothing until you know that always guessing the largest class
scores 0.1502, so `baseline_scores` computes both floors first. The published RIVF'07
numbers in the config are the ceiling at the other end.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.config import Config


class Evaluator:
    """Test-set metrics, the confusion matrix, and the learning-curve plot."""

    def __init__(self, config: Config, class_names: list[str]) -> None:
        self.config = config
        self.class_names = class_names
        self.dpi: int = config.require("evaluation.figure.dpi")

    # -- scores ------------------------------------------------------------------------

    def baseline_scores(self, y_true: np.ndarray) -> dict[str, dict[str, float]]:
        """What "learned nothing" looks like, so the real score has something to beat."""
        largest_class = np.bincount(y_true).argmax()
        rng = np.random.default_rng(self.config.require("training.seed"))
        candidates = {
            "always largest class": np.full_like(y_true, largest_class),
            "uniform random": rng.integers(0, len(self.class_names), len(y_true)),
        }
        return {
            name: {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            }
            for name, y_pred in candidates.items()
        }

    def score(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Accuracy, macro/weighted F1, per-class F1, confusion matrix.

        Macro F1 is the headline: it weights every class equally, so a model failing on the
        smallest topic cannot hide behind the largest.
        """
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
            "per_class_f1": {
                name: float(value)
                for name, value in zip(self.class_names, f1_score(y_true, y_pred, average=None))
            },
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }

    def report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """sklearn's per-class precision/recall/F1 table."""
        return classification_report(y_true, y_pred, target_names=self.class_names, digits=4)

    def largest_confusions(self, y_true: np.ndarray, y_pred: np.ndarray) -> list[tuple[float, str, str]]:
        """The worst off-diagonal cells: which topics the model mixes up, and how badly."""
        matrix = confusion_matrix(y_true, y_pred)
        normalized = matrix / matrix.sum(axis=1, keepdims=True)
        pairs = [
            (normalized[i, j], self.class_names[i], self.class_names[j])
            for i in range(len(self.class_names))
            for j in range(len(self.class_names))
            if i != j
        ]
        return sorted(pairs, reverse=True)[: self.config.require("evaluation.top_confusions")]

    # -- plots -------------------------------------------------------------------------

    def plot_learning_curves(self, history: dict, out_path: Path) -> Path:
        """Train vs. validation loss and accuracy, epoch by epoch."""
        epochs = range(1, len(history["loss"]) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, (key, title) in zip(axes, [("loss", "Loss"), ("accuracy", "Accuracy")]):
            ax.plot(epochs, history[key], "o-", label="train")
            ax.plot(epochs, history[f"val_{key}"], "s-", label="val")
            ax.set_title(title)
            ax.set_xlabel("epoch")
            ax.legend()
            ax.grid(alpha=0.3)
        fig.tight_layout()
        return self._save(fig, out_path)

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> Path:
        """Row-normalized: each row reads as "of the true X, what fraction went where"."""
        matrix = confusion_matrix(y_true, y_pred)
        normalized = matrix / matrix.sum(axis=1, keepdims=True)
        n = len(self.class_names)

        fig, ax = plt.subplots(figsize=(8.5, 7))
        image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(n), self.class_names, rotation=45, ha="right")
        ax.set_yticks(range(n), self.class_names)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title("Neural network on VNTC test (row-normalised)")
        for i in range(n):
            for j in range(n):
                if normalized[i, j] > 0.01:
                    ax.text(
                        j, i, f"{normalized[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if normalized[i, j] > 0.5 else "black",
                    )
        fig.colorbar(image, shrink=0.8)
        fig.tight_layout()
        return self._save(fig, out_path)

    def _save(self, fig, out_path: Path) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=self.dpi)
        plt.close(fig)
        return out_path
