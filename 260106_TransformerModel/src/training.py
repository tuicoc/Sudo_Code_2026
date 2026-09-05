"""The learning-rate schedule, the loss, and the distributed training loop.

Two things here are not boilerplate.

**The loss is label-smoothed, and that changes how its value reads.** Smoothing the target to
`(1-eps)*onehot + eps/V` makes the loss

    (1 - eps) * (-log p_true)  +  eps * mean_over_vocab(-log p_v)

The second term *grows* as the model sharpens, because a confident model drives most of the
vocabulary towards zero and -log 0 is large. So the loss never approaches zero, and validation
loss can rise while the model is still improving. Early stopping therefore watches accuracy.

**The model must be built inside `strategy.scope()`, and so must its first call.** A subclassed
Keras model creates its variables on first call, not in `__init__`; calling it outside the scope
leaves them unmirrored and training fails with `colocate_vars_with must only be passed a variable
created in this Strategy.scope()`, which names the scope but not the line that left it.
"""

from __future__ import annotations

import time

import numpy as np
import tensorflow as tf

from src.config import Config
from src.transformer import PAD, Transformer


class WarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear ramp for warmup_steps, then 1/sqrt(step) decay -- the schedule from the paper.

    Adam's second-moment estimate is unreliable in the first few hundred steps, and a Transformer
    given a full learning rate then tends to diverge.
    """

    def __init__(self, d_model: float, warmup_steps: float) -> None:
        super().__init__()
        self.d_model = float(d_model)
        self.warmup_steps = float(warmup_steps)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(
            tf.math.rsqrt(tf.maximum(step, 1.0)), step * self.warmup_steps ** -1.5)

    def get_config(self) -> dict:
        return {"d_model": self.d_model, "warmup_steps": self.warmup_steps}


class MaskedAccuracy(tf.keras.metrics.Metric):
    """Token accuracy over the positions that are not padding.

    A `Metric` subclass rather than a function: Keras hands a propagated mask to a function metric
    as `sample_weight`, which it cannot apply to the scalar the function returns. Subclassing also
    accumulates totals across batches instead of averaging per-batch means.
    """

    def __init__(self, name: str = "masked_accuracy", **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        mask = tf.cast(y_true != PAD, tf.float32)
        hit = tf.cast(tf.argmax(y_pred, -1, output_type=tf.int32) == y_true, tf.float32)
        self.correct.assign_add(tf.reduce_sum(hit * mask))
        self.total.assign_add(tf.reduce_sum(mask))

    def result(self):
        return self.correct / (self.total + 1e-9)

    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)


class Trainer:
    """Builds the model under a distribution strategy and fits it."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.label_smoothing: float = config.require("training.label_smoothing")
        self.strategy = self.make_strategy(config)
        self.replicas = self.strategy.num_replicas_in_sync
        self.global_batch = config.require("training.per_replica_batch") * self.replicas
        self.train_seconds = 0.0

    @staticmethod
    def make_strategy(config: Config):
        """MirroredStrategy when there is more than one GPU and the config allows it."""
        gpus = tf.config.list_physical_devices("GPU")
        if config.require("training.use_multi_gpu") and len(gpus) > 1:
            return tf.distribute.MirroredStrategy()
        return tf.distribute.get_strategy()

    def loss(self, y_true, y_pred):
        """Label-smoothed cross-entropy over real tokens only.

        Written out rather than via `categorical_crossentropy(label_smoothing=...)`, which needs a
        one-hot target: at 128 x 63 positions x 8000 words that is 258 MB for a tensor whose every
        row is a single 1.
        """
        y_true = tf.cast(y_true, tf.int32)          # Keras hands targets over as float32
        log_probs = tf.nn.log_softmax(y_pred, axis=-1)
        nll = -tf.gather(log_probs, y_true, batch_dims=2)
        uniform = -tf.reduce_mean(log_probs, axis=-1)
        per_token = (1 - self.label_smoothing) * nll + self.label_smoothing * uniform

        mask = tf.cast(y_true != PAD, per_token.dtype)
        return tf.reduce_sum(per_token * mask) / tf.reduce_sum(mask)

    def make_dataset(self, source: np.ndarray, target: np.ndarray, shuffle: bool):
        """((source, target_in), target_out) -- target_out is target_in shifted one left."""
        dataset = tf.data.Dataset.from_tensor_slices(((source, target[:, :-1]), target[:, 1:]))
        if shuffle:
            dataset = dataset.shuffle(self.config.require("training.shuffle_buffer"),
                                      seed=self.config.require("training.seed"),
                                      reshuffle_each_iteration=True)
        return (dataset.batch(self.global_batch, drop_remainder=True)
                .prefetch(tf.data.AUTOTUNE))

    def build(self, source_vocab: int, target_vocab: int, sample=None, strategy=None,
              compile_model: bool = True) -> Transformer:
        """Build, and compile unless this is only going to decode.

        `compile_model=False` skips the optimizer: decoding does not need one, and building it
        makes `load_weights` warn that it is skipping optimizer state it never had.
        """
        tf.keras.utils.set_random_seed(self.config.require("training.seed"))
        with (strategy or self.strategy).scope():
            model = Transformer(self.config, source_vocab, target_vocab)
            if sample is not None:
                model(sample)          # variables are created here, and must be in scope
            if not compile_model:
                return model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    WarmupSchedule(self.config.require("model.d_model"),
                                   self.config.require("training.warmup_steps")),
                    beta_1=0.9, beta_2=0.98, epsilon=1e-9),
                loss=self.loss, metrics=[MaskedAccuracy()])
        return model

    def fit(self, model: Transformer, train_ds, val_ds, verbose: int = 2) -> dict:
        callbacks = [tf.keras.callbacks.EarlyStopping(
            monitor=self.config.require("training.monitor"),
            mode=self.config.require("training.monitor_mode"),
            patience=self.config.require("training.patience"),
            restore_best_weights=True)]
        started = time.time()
        history = model.fit(train_ds, validation_data=val_ds,
                            epochs=self.config.require("training.epochs"),
                            callbacks=callbacks, verbose=verbose, shuffle=False)
        self.train_seconds = time.time() - started
        return {key: [float(v) for v in values] for key, values in history.history.items()}

    def benchmark(self, source_vocab: int, target_vocab: int, source: np.ndarray,
                  target: np.ndarray, sample, steps: int = 30) -> dict | None:
        """Seconds per step on one device against all of them. None when there is only one.

        Measured rather than assumed: the reference run got 1.56x on Kaggle's T4 x2 against a
        perfect 2.00x, and the gap is the per-step gradient all-reduce.
        """
        if self.replicas < 2:
            return None

        per_replica = self.config.require("training.per_replica_batch")

        def seconds_per_step(strategy, batch):
            model = self.build(source_vocab, target_vocab, sample, strategy=strategy)
            dataset = (tf.data.Dataset
                       .from_tensor_slices(((source, target[:, :-1]), target[:, 1:]))
                       .batch(batch, drop_remainder=True).repeat().prefetch(tf.data.AUTOTUNE))
            model.fit(dataset, steps_per_epoch=5, epochs=1, verbose=0, shuffle=False)  # trace
            started = time.time()
            model.fit(dataset, steps_per_epoch=steps, epochs=1, verbose=0, shuffle=False)
            return (time.time() - started) / steps

        device = "/gpu:0" if tf.config.list_physical_devices("GPU") else "/cpu:0"
        one = seconds_per_step(tf.distribute.OneDeviceStrategy(device), per_replica)
        many = seconds_per_step(self.strategy, self.global_batch)
        return {"one_replica_ms": round(one * 1000, 1),
                "all_replicas_ms": round(many * 1000, 1),
                "replicas": self.replicas,
                "speedup": round((self.global_batch / many) / (per_replica / one), 3)}
