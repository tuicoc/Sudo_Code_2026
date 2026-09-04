"""The neural network: one hidden layer over TF-IDF features.

Deliberately small -- the question is whether a simple network beats a linear model, which
is only answerable if the result is attributable to the architecture, not to tuning.

`sparse=True` on the input layer keeps the 10,000-feature matrix sparse; densifying it
would cost ~1.3 GB for the training split alone.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

from src.config import Config


class TopicClassifier:
    """Builds, trains, saves and loads the Keras topic classifier."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.seed: int = config.require("training.seed")
        self.model: tf.keras.Model | None = None
        self.train_seconds: float = 0.0

    def build(self, n_features: int) -> tf.keras.Model:
        """One hidden ReLU layer with dropout, then a softmax over the classes."""
        tf.keras.utils.set_random_seed(self.seed)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(n_features,), sparse=True),
            tf.keras.layers.Dense(self.config.require("model.hidden_units"), activation="relu"),
            tf.keras.layers.Dropout(self.config.require("model.dropout")),
            tf.keras.layers.Dense(self.config.require("model.n_classes"), activation="softmax"),
        ])
        self.model.compile(
            optimizer=self.config.require("model.optimizer"),
            loss=self.config.require("model.loss"),
            metrics=["accuracy"],
        )
        return self.model

    def split_validation(self, X: csr_matrix, y: np.ndarray):
        """Stratified validation split.

        NOT Keras's `validation_split`, which takes the last N rows -- and the corpus is
        ordered by class, so that tail would be a handful of classes only.
        """
        return train_test_split(
            X, y,
            test_size=self.config.require("training.validation_split"),
            stratify=y,
            random_state=self.seed,
        )

    def train(self, X: csr_matrix, y: np.ndarray, verbose: int = 2) -> dict:
        """Fit with early stopping, and return the history.

        Early stopping restores the best weights, so the saved model is the epoch with the
        lowest validation loss, not the last one.
        """
        X_train, X_val, y_train, y_val = self.split_validation(X, y)
        early_stopping = self.config.require("training.early_stopping")

        started = time.time()
        history = self._require_model().fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.require("training.epochs"),
            batch_size=self.config.require("training.batch_size"),
            callbacks=[tf.keras.callbacks.EarlyStopping(**early_stopping)],
            verbose=verbose,
        )
        self.train_seconds = time.time() - started
        return {key: [float(v) for v in values] for key, values in history.history.items()}

    def predict(self, X: csr_matrix) -> np.ndarray:
        """Predicted class index per row."""
        return self._require_model().predict(X, verbose=0).argmax(axis=1)

    def metadata(self, history: dict) -> dict:
        """What the run cost, saved next to the model."""
        return {
            "train_secs": round(self.train_seconds, 1),
            "epochs_run": len(history["loss"]),
            "params": int(self._require_model().count_params()),
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._require_model().save(path)
        return path

    def load(self, path: Path) -> tf.keras.Model:
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing. Run: python main.py --stage train")
        self.model = tf.keras.models.load_model(path)
        return self.model

    def _require_model(self) -> tf.keras.Model:
        if self.model is None:
            raise RuntimeError("No model yet: call build() or load() first.")
        return self.model
