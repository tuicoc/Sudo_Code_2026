"""The language model: embedding -> LSTM -> one score per word in the vocabulary.

The corpus predicts itself. A window of `seq_len + 1` tokens is cut from the stream; the
first `seq_len` are the input, the last `seq_len` the target -- the same window shifted one
step left. Every position is a training example.

Two settings that look arbitrary and are not:

* `recurrent_dropout=0.0` is required for the cuDNN fast path. Any other value silently
  falls back to a much slower implementation.
* Dense emits raw logits with `from_logits=True` on the loss -- doing the softmax inside
  the loss is numerically stabler.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.config import Config


class LanguageModel:
    """Builds, trains, saves and loads the LSTM language model."""

    def __init__(self, config: Config, vocab_size: int) -> None:
        self.config = config
        self.vocab_size = vocab_size
        self.seq_len: int = config.require("model.seq_len")
        self.model: tf.keras.Model | None = None
        self.train_seconds: float = 0.0

    # -- data ---------------------------------------------------------------------------

    def make_dataset(self, tokens: np.ndarray, shuffle: bool) -> tf.data.Dataset:
        """Cut the token stream into (input, target) windows shifted by one.

        The reshape is a view, not a copy -- the stream can be hundreds of megabytes.
        """
        window = self.seq_len + 1
        n_windows = (tokens.size - 1) // window
        windows = tokens[: n_windows * window].reshape(n_windows, window)

        dataset = tf.data.Dataset.from_tensor_slices(windows)
        if shuffle:
            dataset = dataset.shuffle(
                self.config.require("training.shuffle_buffer"),
                seed=self.config.require("training.seed"),
                reshuffle_each_iteration=True,
            )
        dataset = dataset.map(
            lambda w: (tf.cast(w[:-1], tf.int32), tf.cast(w[1:], tf.int32)),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        return (
            dataset.batch(self.config.require("training.batch_size"), drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )

    # -- the model ----------------------------------------------------------------------

    def build(self, seq_len: int | None = None) -> tf.keras.Model:
        """Embedding -> LSTM -> Dense(vocab_size) logits.

        `seq_len=None` gives an unconstrained input length, which generation needs.
        """
        tf.keras.utils.set_random_seed(self.config.require("training.seed"))
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(seq_len,), dtype="int32"),
            # mask_zero=False on purpose: index 0 is never emitted, so there is nothing to mask.
            tf.keras.layers.Embedding(self.vocab_size, self.config.require("model.embed_dim")),
            tf.keras.layers.LSTM(
                self.config.require("model.lstm_units"),
                return_sequences=True,
                recurrent_dropout=self.config.require("model.recurrent_dropout"),
            ),
            tf.keras.layers.Dense(self.vocab_size),
        ])
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(self.config.require("model.learning_rate")),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        )
        return self.model

    def train(self, train_tokens: np.ndarray, val_tokens: np.ndarray, verbose: int = 2) -> dict:
        """Fit on the training stream, validating on the held-out books."""
        model = self._require_model()
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                self.config.path("paths.model_file"), save_best_only=True, monitor="val_loss"
            ),
            tf.keras.callbacks.CSVLogger(self.config.path("paths.training_log"), append=True),
            tf.keras.callbacks.EarlyStopping(**self.config.require("training.early_stopping")),
        ]
        self.config.path("paths.outputs_dir").mkdir(parents=True, exist_ok=True)

        started = time.time()
        history = model.fit(
            self.make_dataset(train_tokens, shuffle=True),
            validation_data=self.make_dataset(val_tokens, shuffle=False),
            epochs=self.config.require("training.epochs"),
            callbacks=callbacks,
            verbose=verbose,
        )
        self.train_seconds = time.time() - started
        return {key: [float(v) for v in values] for key, values in history.history.items()}

    def evaluate(self, val_tokens: np.ndarray) -> dict:
        """Validation loss, and the perplexity it implies.

        `exp(loss)` is roughly how many words the model is choosing between at each step.
        Compare it against the vocabulary size, which is what "learned nothing" scores.
        """
        # Keras returns a bare float when the model has only a loss, and a list once any
        # metric is compiled in. Accept both, so adding a metric later cannot break this.
        result = self._require_model().evaluate(self.make_dataset(val_tokens, shuffle=False), verbose=0)
        loss = float(result[0] if isinstance(result, (list, tuple)) else result)
        return {
            "val_loss": loss,
            "val_perplexity": float(np.exp(loss)),
            "uniform_perplexity": self.vocab_size,
            "train_secs": round(self.train_seconds, 1),
        }

    def load(self, path: Path | None = None) -> tf.keras.Model:
        path = path or self.config.path("paths.model_file")
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing. Run: python main.py --stage train")
        self.model = tf.keras.models.load_model(path)
        return self.model

    def _require_model(self) -> tf.keras.Model:
        if self.model is None:
            raise RuntimeError("No model yet: call build() or load() first.")
        return self.model
