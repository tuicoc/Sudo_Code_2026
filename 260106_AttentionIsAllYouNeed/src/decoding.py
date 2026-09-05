"""Turning a trained model into summaries.

Greedy decoding: take the most likely token, append it, ask again. Nothing is sampled -- a
summary should be the model's best answer, not a creative one.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from src.config import Config
from src.vocabulary import Vocabulary


class Decoder:
    """Greedy decoding with n-gram blocking, and the attention weights behind one summary."""

    def __init__(self, config: Config, model, vocabulary: Vocabulary) -> None:
        self.config = config
        self.model = model
        self.vocabulary = vocabulary
        self.max_summary: int = config.require("preprocessing.max_summary")
        self.block: int = config.require("decoding.block_ngram")
        self.batch_size: int = config.require("decoding.batch_size")

    def generate(self, articles: np.ndarray) -> list[list[str]]:
        """Decode a batch of encoded articles into token lists.

        The prefix is re-run through the decoder at every step rather than carrying the GRU state
        forward. That is O(T^2) work for T <= 63 output tokens, which costs seconds, and it
        removes a whole class of bugs where the decoder behaves differently at training and at
        generation time. The encoder still runs once per batch.
        """
        vocab = self.vocabulary
        results = []

        for start in range(0, len(articles), self.batch_size):
            article = tf.constant(articles[start:start + self.batch_size])
            enc_mask = article != vocab.pad
            enc_out, state = self.model.encode(article)

            n = int(article.shape[0])
            sequence = np.full((n, 1), vocab.bos, np.int32)
            finished = np.zeros(n, bool)

            for step in range(self.max_summary - 1):
                logits, _ = self.model.decode(tf.constant(sequence), enc_out, state, enc_mask,
                                              last_only=True)
                scores = logits[:, -1, :].numpy().astype(np.float32)
                scores[:, [vocab.pad, vocab.unk, vocab.bos]] = -1e9   # <eos> is the only one allowed
                if self.block and step >= self.block:
                    self._block_repeats(scores, sequence)

                nxt = scores.argmax(-1).astype(np.int32)
                nxt[finished] = vocab.pad
                finished |= nxt == vocab.eos
                sequence = np.concatenate([sequence, nxt[:, None]], axis=1)
                if finished.all():
                    break

            results.extend(vocab.decode(row[1:]) for row in sequence)
        return results

    def _block_repeats(self, scores: np.ndarray, sequence: np.ndarray) -> None:
        """Refuse any token that would complete an n-gram already generated.

        Greedy decoding loops: it reaches a state whose likeliest continuation returns it to that
        state, and repeats a phrase until the length limit. This is the blunt standard fix.
        """
        for i, row in enumerate(sequence):
            history = row.tolist()
            prefix = tuple(history[-(self.block - 1):])
            banned = [history[j + self.block - 1]
                      for j in range(len(history) - self.block + 1)
                      if tuple(history[j:j + self.block - 1]) == prefix]
            if banned:
                scores[i, banned] = -1e9

    def attention_for(self, article_ids: np.ndarray):
        """Attention weights for one article, decoded greedily. (summary_len, article_len)."""
        if self.model.attention is None:
            raise ValueError("this model has no attention layer to inspect")
        article = tf.constant(article_ids[None, :])
        enc_out, state = self.model.encode(article)
        tokens = self.generate(article_ids[None, :])[0]
        if not tokens:
            return None, []
        ids = np.array([[self.vocabulary.bos] +
                        [self.vocabulary.stoi.get(t, self.vocabulary.unk) for t in tokens]],
                       np.int32)
        _, weights = self.model.decode(tf.constant(ids), enc_out, state,
                                       article != self.vocabulary.pad)
        return np.array(weights)[0], tokens
