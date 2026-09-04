"""Stratified k-fold cross-validation, shared by every experiment.

One 80/20 split on 3,040 reviews would make the comparison mostly luck. Five folds give a
mean AND a spread, and a difference only counts if it beats the spread. Stratified, so
every fold keeps the corpus's class proportions.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.config import Config

METRIC_KEYS = ("accuracy", "precision_macro", "recall_macro", "f1_macro")


class CrossValidator:
    """Runs one (vectorizer, model) combination through k folds and summarizes it."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.n_splits: int = config.require("evaluation.n_splits")
        self.random_state: int = config.require("evaluation.random_state")
        self.average: str = config.require("evaluation.average")

    def model_factory(self, kind: str) -> Callable[[], MultinomialNB | LinearSVC]:
        """Return a zero-argument callable making a fresh, untrained classifier."""
        models: dict[str, Callable[[], MultinomialNB | LinearSVC]] = {
            "multinomial_nb": MultinomialNB,
            "linear_svc": lambda: LinearSVC(random_state=self.random_state),
        }
        if kind not in models:
            raise ValueError(f"Unknown model {kind!r}; expected one of {sorted(models)}")
        return models[kind]

    def run(self, vectorizer_factory: Callable, model_factory: Callable,
            X: np.ndarray, y: np.ndarray) -> list[dict]:
        """Fit and score one fold at a time.

        The vectorizer is fitted inside the loop, on the training fold only -- no leakage.
        """
        splitter = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )

        fold_metrics = []
        for fold, (train_index, val_index) in enumerate(splitter.split(X, y), start=1):
            vectorizer = vectorizer_factory()
            X_train = vectorizer.fit_transform(X[train_index])
            X_val = vectorizer.transform(X[val_index])
            y_train, y_val = y[train_index], y[val_index]

            model = model_factory()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            precision, recall, f1, _support = precision_recall_fscore_support(
                y_val, y_pred, average=self.average, zero_division=0
            )
            fold_metrics.append({
                "fold": fold,
                "vocab_size": len(vectorizer.vocabulary_),
                "accuracy": accuracy_score(y_val, y_pred),
                "precision_macro": precision,
                "recall_macro": recall,
                "f1_macro": f1,
            })
        return fold_metrics

    def summarize(self, fold_metrics: list[dict], experiment: dict,
                  vectorizer_description: str) -> dict:
        """Fold metrics -> the JSON saved to data/outputs/. Keeps mean AND std."""
        return {
            "method": experiment["key"],
            "label": experiment["label"],
            "vectorizer": vectorizer_description,
            "model": experiment["model"],
            "n_splits": self.n_splits,
            "folds": fold_metrics,
            "mean": {k: float(np.mean([f[k] for f in fold_metrics])) for k in METRIC_KEYS},
            "std": {k: float(np.std([f[k] for f in fold_metrics])) for k in METRIC_KEYS},
        }
