"""The comparison the project exists to make, on two axes:

* model -- the network against LinearSVC and MultinomialNB, on identical features
* input -- word-segmented text against the raw, unsegmented corpus

Without the second axis, a good score could be credit owed to `underthesea` rather than to
the network. Both axes means the two can be read separately.
"""

from __future__ import annotations

import time
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from src.config import Config
from src.dataloader import DataLoader
from src.feature_extraction import FeatureExtractor
from src.model import TopicClassifier

CLASSIC_MODELS = {"LinearSVC": LinearSVC, "MultinomialNB": MultinomialNB}


class ModelComparison:
    """Runs every model on both input variants and collects one score table."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.loader = DataLoader(config)
        self.seed: int = config.require("training.seed")
        self.results: dict[str, dict] = {}

    def record(self, name: str, y_true: np.ndarray, y_pred: np.ndarray, seconds: float) -> dict:
        """Score one run, keep it, and print the line."""
        result = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "train_secs": round(seconds, 1),
        }
        self.results[name] = result
        print(f"  {name:<32} acc={result['accuracy']:.4f}  "
              f"macro-F1={result['macro_f1']:.4f}  ({seconds:.0f}s)")
        return result

    # -- the two input variants --------------------------------------------------------

    def run_segmented(self) -> None:
        """Score every model on the cached, word-segmented TF-IDF features."""
        X_train, y_train = self.loader.load_features("train")
        X_test, y_test = self.loader.load_features("test")

        classifier = TopicClassifier(self.config)
        classifier.load(self.config.path("paths.model_file"))
        trained_seconds = self.loader.load_json("paths.train_meta_file")["train_secs"]
        self.record("Neural network (segmented)", y_test, classifier.predict(X_test), trained_seconds)

        self._run_classic_models(X_train, y_train, X_test, y_test, suffix="segmented")

    def run_unsegmented(self) -> None:
        """Re-run everything on the raw corpus, with no word segmentation at all."""
        train_texts, train_labels = self._load_raw("Train_Full")
        test_texts, test_labels = self._load_raw("Test_Full")

        encoder = LabelEncoder().fit(train_labels)
        y_train, y_test = encoder.transform(train_labels), encoder.transform(test_labels)

        extractor = FeatureExtractor(self.config)
        X_train, X_test = extractor.fit_transform(train_texts, test_texts)
        compounds, _total = extractor.compound_feature_share()
        print(f"  compound features in the unsegmented vocabulary: {compounds}")

        started = time.time()
        network = TopicClassifier(self.config)
        network.build(n_features=X_train.shape[1])
        network.train(X_train, y_train, verbose=0)
        self.record("Neural network (unsegmented)", y_test, network.predict(X_test),
                    time.time() - started)

        self._run_classic_models(X_train, y_train, X_test, y_test, suffix="unsegmented")

    def run_floors(self) -> None:
        """The two "learned nothing" baselines, on the same test set."""
        _X_test, y_test = self.loader.load_features("test")
        rng = np.random.default_rng(self.seed)
        n_classes = self.config.require("model.n_classes")
        self.record("Floor: always largest class", y_test,
                    np.full_like(y_test, np.bincount(y_test).argmax()), 0.0)
        self.record("Floor: uniform random", y_test,
                    rng.integers(0, n_classes, len(y_test)), 0.0)

    # -- output ------------------------------------------------------------------------

    def ranking(self) -> list[tuple[str, dict]]:
        """Every run, best macro-F1 first."""
        return sorted(self.results.items(), key=lambda item: -item[1]["macro_f1"])

    def format_table(self) -> str:
        """The score table as printed text."""
        lines = [f"{'model':<34} {'accuracy':>9} {'macro-F1':>9} {'train':>8}", "-" * 63]
        for name, result in self.ranking():
            lines.append(f"{name:<34} {result['accuracy']:9.4f} "
                         f"{result['macro_f1']:9.4f} {result['train_secs']:7.0f}s")
        lines.append("")
        for name, accuracy in self.config.require("evaluation.baselines").items():
            lines.append(f"published {name:<24} {accuracy:9.4f}")
        return "\n".join(lines)

    def plot(self, out_path: Path | None = None) -> Path:
        """Horizontal bars per model, with the published baseline and the floor marked."""
        ranked = [(name, r) for name, r in self.ranking() if not name.startswith("Floor")]
        names = [name for name, _ in ranked]
        colors = ["#2b6cb0" if "Neural" in name else "#a0aec0" for name in names]
        floor = self.results.get("Floor: always largest class", {}).get("accuracy")
        baselines = self.config.require("evaluation.baselines")

        fig, ax = plt.subplots(figsize=(9.5, 4.6))
        positions = np.arange(len(names))
        ax.barh(positions - 0.2, [r["accuracy"] for _, r in ranked], 0.38,
                label="accuracy", color=colors, alpha=0.55)
        ax.barh(positions + 0.2, [r["macro_f1"] for _, r in ranked], 0.38,
                label="macro-F1", color=colors)
        ax.set_yticks(positions, names)
        ax.invert_yaxis()

        for name, accuracy in baselines.items():
            ax.axvline(accuracy, ls="--", c="crimson", lw=1.2, label=f"{name} ({accuracy:.1%})")
        if floor is not None:
            ax.axvline(floor, ls=":", c="gray", lw=1.2, label=f"majority-class floor ({floor:.1%})")

        ax.set_xlim(0, 1.0)
        ax.set_xlabel("score on VNTC official test set")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()

        out_path = out_path or self.config.path("paths.comparison_figure")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=self.config.require("evaluation.figure.dpi"))
        plt.close(fig)
        return out_path

    # -- internals ---------------------------------------------------------------------

    def _run_classic_models(self, X_train, y_train, X_test, y_test, suffix: str) -> None:
        for name, factory in CLASSIC_MODELS.items():
            model = factory(random_state=self.seed) if name == "LinearSVC" else factory()
            started = time.time()
            model.fit(X_train, y_train)
            seconds = time.time() - started
            self.record(f"{name} ({suffix})", y_test, model.predict(X_test), seconds)

    def _load_raw(self, split: str) -> tuple[list[str], list[str]]:
        """Read a raw split, NFC-normalized but *not* word-segmented."""
        texts, labels = self.loader.load_split(split, processed=False)
        return [unicodedata.normalize("NFC", text) for text in texts], labels
