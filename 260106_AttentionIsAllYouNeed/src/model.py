"""The encoder-decoder, in the two shapes the experiment compares.

An encoder GRU reads the article and produces one hidden state per position; a decoder GRU
writes the summary one token at a time, starting from the encoder's final state.

* **Without attention** that final state is *all* the decoder ever gets: 256 tokens of article
  compressed into 256 numbers, still needed sixty output steps later. This is the bottleneck the
  attention papers were written about.
* **With attention** the decoder keeps the same starting state but also computes a fresh weighted
  read of every encoder position at every output step.

Everything else is identical between the two, including the width of the output layer -- with
attention, `[decoder state; context]` is folded back to `units` first, so the comparison stays
about attention rather than about parameter count.

Attention is applied to the decoder's output rather than fed back into the GRU as Bahdanau's
original does. That version needs a Python loop over output steps; this one is a single batched
call, which is roughly ten times faster to train. The cost is that the GRU never learns what it
attended to at the previous step.
"""

from __future__ import annotations

import time

import numpy as np
import tensorflow as tf

from src.attention import build_attention
from src.config import Config

scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")


def masked_loss(y_true, y_pred):
    """Cross-entropy over the real tokens only.

    Every summary is padded to `max_summary` but the average abstract is 28 tokens. Score the
    padding and more than half the loss is the model being graded on predicting <pad> after
    <pad> -- which it learns perfectly, hiding whether it learned anything else.
    """
    loss = scce(y_true, y_pred)
    mask = tf.cast(y_true != 0, loss.dtype)
    return tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)


class MaskedAccuracy(tf.keras.metrics.Metric):
    """How often the model's top guess is the right token, padding excluded.

    A plain function metric cannot be used here. `Embedding(mask_zero=True)` makes Keras attach a
    (batch, time) mask to the model's output and hand it to a function metric as `sample_weight`,
    which it then tries to broadcast onto the single number the function returned -- a
    `rank 2 into rank 0` error at the first training step, pointing at Keras internals rather
    than at the mask. A Metric subclass does its own masking instead, and is also more correct:
    totals accumulate across batches rather than averaging per-batch means.
    """

    def __init__(self, name: str = "masked_accuracy", **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        mask = tf.cast(y_true != 0, tf.float32)
        hit = tf.cast(tf.argmax(y_pred, -1, output_type=y_true.dtype) == y_true, tf.float32)
        self.correct.assign_add(tf.reduce_sum(hit * mask))
        self.total.assign_add(tf.reduce_sum(mask))

    def result(self):
        return self.correct / (self.total + 1e-9)

    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)


class Seq2Seq(tf.keras.Model):
    """GRU encoder-decoder for summarization, with attention or without it."""

    def __init__(self, config: Config, vocab_size: int, attention_kind: str = "additive") -> None:
        super().__init__()
        self.config = config
        self.attention_kind = attention_kind
        units = config.require("model.units")

        self.embedding = tf.keras.layers.Embedding(
            vocab_size, config.require("model.embed_dim"), mask_zero=True)
        self.encoder = tf.keras.layers.GRU(units, return_sequences=True, return_state=True)
        self.decoder = tf.keras.layers.GRU(units, return_sequences=True, return_state=True)
        self.attention = build_attention(attention_kind, config)
        self.combine = (tf.keras.layers.Dense(units, activation="tanh")
                        if self.attention is not None else None)
        # float32 out regardless of the global policy: softmax and cross-entropy are where fp16
        # loses precision most.
        self.out = tf.keras.layers.Dense(vocab_size, dtype="float32")
        self.train_seconds = 0.0

    # -- the two halves -----------------------------------------------------------------

    def encode(self, article):
        enc_out, state = self.encoder(self.embedding(article))
        return enc_out, state

    def decode(self, dec_in, enc_out, state, enc_mask, last_only: bool = False):
        """Returns (logits, attention weights or None).

        `last_only` scores just the final position, which is all generation needs; it keeps the
        vocabulary-sized output layer off the positions that are not being predicted.
        """
        dec_out, _ = self.decoder(self.embedding(dec_in), initial_state=state)
        weights = None
        if self.attention is not None:
            context, weights = self.attention(dec_out, enc_out, enc_mask)
            dec_out = self.combine(tf.concat([dec_out, context], axis=-1))
        if last_only:
            dec_out = dec_out[:, -1:, :]
        return self.out(dec_out), weights

    def call(self, inputs):
        article, dec_in = inputs
        enc_out, state = self.encode(article)
        return self.decode(dec_in, enc_out, state, article != 0)[0]

    # -- training -----------------------------------------------------------------------

    @staticmethod
    def make_dataset(config: Config, articles: np.ndarray, summaries: np.ndarray,
                     shuffle: bool) -> tf.data.Dataset:
        """((article, decoder input), target) -- the target is the input shifted one left."""
        dataset = tf.data.Dataset.from_tensor_slices(
            ((articles, summaries[:, :-1]), summaries[:, 1:]))
        if shuffle:
            dataset = dataset.shuffle(config.require("training.shuffle_buffer"),
                                      seed=config.require("training.seed"),
                                      reshuffle_each_iteration=True)
        return (dataset.batch(config.require("training.batch_size"), drop_remainder=True)
                .prefetch(tf.data.AUTOTUNE))

    @classmethod
    def create(cls, config: Config, vocab_size: int, attention_kind: str,
               sample_batch=None, compile_model: bool = True) -> "Seq2Seq":
        """Build, and compile unless this is only going to decode.

        A subclassed model has no shapes until it is called once, hence `sample_batch`.
        `compile_model=False` skips the optimizer: decoding does not need one, and building it
        makes `load_weights` warn that it is skipping optimizer state it never had.
        """
        tf.keras.utils.set_random_seed(config.require("training.seed"))
        model = cls(config, vocab_size, attention_kind)
        if sample_batch is not None:
            model(sample_batch)
        if not compile_model:
            return model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(config.require("training.learning_rate"),
                                               clipnorm=config.require("training.clipnorm")),
            loss=masked_loss,
            metrics=[MaskedAccuracy()],
        )
        return model

    def fit_on(self, train_ds, val_ds, verbose: int = 2) -> dict:
        callbacks = [tf.keras.callbacks.EarlyStopping(
            **self.config.require("training.early_stopping"))]
        started = time.time()
        history = self.fit(train_ds, validation_data=val_ds,
                           epochs=self.config.require("training.epochs"),
                           callbacks=callbacks, verbose=verbose, shuffle=False)
        self.train_seconds = time.time() - started
        return {key: [float(v) for v in values] for key, values in history.history.items()}
