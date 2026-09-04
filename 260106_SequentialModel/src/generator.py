"""Samples text from the trained model.

The trained model has a fixed input length of `seq_len`, which generation cannot use -- a
prompt is however long it is and grows by one token per step. So the weights are copied
into the same architecture built with an unconstrained input length. No retraining.

Three sampling strategies:

* greedy -- always take the highest-scoring token. Deterministic, and it loops.
* temperature -- divide the logits before the softmax. Below 1 is safer and duller, above 1
  is more surprising and less coherent.
* top-k -- keep only the k best candidates, then sample. Keeps the variety of sampling
  without the long tail of near-impossible tokens that derail a sentence.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from src.config import Config
from src.preprocessing import Preprocessor
from src.vocabulary import Vocabulary


class TextGenerator:
    """Generates text from a trained language model, one token at a time."""

    def __init__(self, config: Config, model: tf.keras.Model, vocabulary: Vocabulary) -> None:
        self.config = config
        self.vocabulary = vocabulary
        self.preprocessor = Preprocessor(config)
        self.model = self._rebuild_for_variable_length(model)

    def _rebuild_for_variable_length(self, trained: tf.keras.Model) -> tf.keras.Model:
        """Copy the trained weights into the same architecture, minus the fixed input length."""
        embedding, lstm, _dense = trained.layers
        generation_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(None,), dtype="int32"),
            tf.keras.layers.Embedding(len(self.vocabulary), embedding.output_dim),
            tf.keras.layers.LSTM(lstm.units, return_sequences=True),
            tf.keras.layers.Dense(len(self.vocabulary)),
        ])
        generation_model.set_weights(trained.get_weights())
        return generation_model

    def encode(self, text: str) -> list[int]:
        """Run the prompt through the same cleaning and tokenizing as the corpus."""
        return self.vocabulary.encode(self.preprocessor.process_text(text))

    def generate(self, prompt: str, n_tokens: int | None = None, temperature: float | None = None,
                 top_k: int | None = None, greedy: bool = False, seed: int | None = None) -> str:
        """Extend the prompt by `n_tokens`, sampling as configured."""
        n_tokens = n_tokens if n_tokens is not None else self.config.require("generation.n_tokens")
        temperature = temperature if temperature is not None else self.config.require("generation.temperature")
        seed = seed if seed is not None else self.config.require("generation.seed")

        rng = np.random.default_rng(seed)
        ids = self.encode(prompt)
        for _ in range(n_tokens):
            logits = self.model.predict(np.array([ids], dtype=np.int32), verbose=0)[0, -1]
            if greedy:
                ids.append(int(logits.argmax()))
                continue
            ids.append(int(rng.choice(len(self.vocabulary), p=self._probabilities(logits, temperature, top_k))))
        return self.vocabulary.decode(ids)

    @staticmethod
    def _probabilities(logits: np.ndarray, temperature: float, top_k: int | None) -> np.ndarray:
        """Logits -> a sampling distribution, after temperature and the optional top-k cut."""
        logits = logits / temperature
        if top_k:
            keep = np.argpartition(logits, -top_k)[-top_k:]
            masked = np.full_like(logits, -np.inf)
            masked[keep] = logits[keep]
            logits = masked
        # Subtract the max before exponentiating: standard guard against overflow.
        probabilities = np.exp(logits - logits.max())
        return probabilities / probabilities.sum()
