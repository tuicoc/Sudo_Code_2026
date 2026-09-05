"""Turning a trained Transformer into translations.

Both decoders re-run the whole prefix at each step rather than caching keys and values. That is
O(T^2) work for T <= 64, and it keeps generation on exactly the same code path as training.

The `tf.function` is most of the speed, not decoration. A test set is ~1,400 decode calls and
beam search ~19,000; eager mode pays full Python dispatch on every one, which for a 6M-parameter
model costs more than the arithmetic. On the reference run tracing took greedy decoding of 2,793
sentences from 175 s to 31 s. `reduce_retracing=True` matters because the prefix grows by one
token per step -- without it TensorFlow traces a new graph for every length.

Decoding runs on **one** device. `MirroredStrategy` distributes training steps; a hand-written
generation loop is ordinary eager code. That is the right trade -- the loop is sequential by
nature, so there is nothing to split -- but it means decode timings are not comparable with the
training throughput.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from src.config import Config
from src.vocabulary import Vocabulary


class Translator:
    """Greedy and beam-search decoding for one trained model."""

    def __init__(self, config: Config, model, target_vocabulary: Vocabulary) -> None:
        self.config = config
        self.model = model
        self.vocabulary = target_vocabulary
        self.max_len: int = config.require("preprocessing.max_len")
        self.batch_size: int = config.require("decoding.batch_size")
        self.beam_width: int = config.require("decoding.beam_width")

        # An explicit signature pins this to ONE traced graph. Without it TensorFlow retraces
        # for every prefix length and every batch shape, warns about it after five, and the
        # tracing itself costs more than the decoding.
        compute_dtype = tf.keras.mixed_precision.global_policy().compute_dtype
        d_model = config.require("model.d_model")
        self._decode_step = tf.function(
            lambda sequence, context, source_mask: model.decode(sequence, context, source_mask),
            input_signature=[tf.TensorSpec([None, None], tf.int32),
                             tf.TensorSpec([None, None, d_model], compute_dtype),
                             tf.TensorSpec([None, None], tf.bool)])

    def greedy(self, sources: np.ndarray) -> list[list[str]]:
        """Batched greedy decoding. Returns lists of target tokens."""
        vocab = self.vocabulary
        outputs = []
        for start in range(0, len(sources), self.batch_size):
            source = tf.constant(sources[start:start + self.batch_size])
            context, source_mask = self.model.encode(source)

            n = int(source.shape[0])
            sequence = np.full((n, 1), vocab.bos, np.int32)
            done = np.zeros(n, bool)
            for _ in range(self.max_len - 1):
                logits = self._decode_step(tf.constant(sequence), context,
                                           source_mask)[:, -1, :].numpy()
                logits[:, [vocab.pad, vocab.bos]] = -1e9
                nxt = logits.argmax(-1).astype(np.int32)
                nxt[done] = vocab.pad
                done |= nxt == vocab.eos
                sequence = np.concatenate([sequence, nxt[:, None]], axis=1)
                if done.all():
                    break
            outputs.extend(vocab.decode(row[1:]) for row in sequence)
        return outputs

    def beam(self, source_ids: np.ndarray, beam: int | None = None,
             alpha: float = 1.0) -> list[str]:
        """Beam search for one source sentence.

        Scores are divided by length^alpha before a finished beam is kept. Without that division
        beam search systematically prefers short translations, because every extra word multiplies
        in another probability below 1.
        """
        vocab = self.vocabulary
        beam = beam or self.beam_width

        context, source_mask = self.model.encode(tf.constant(source_ids[None, :]))
        context = tf.repeat(context, beam, axis=0)
        source_mask = tf.repeat(source_mask, beam, axis=0)

        sequences = np.full((beam, 1), vocab.bos, np.int32)
        scores = np.full(beam, -1e9, np.float32)
        scores[0] = 0.0          # one live beam at the start, or all beams would tie
        finished = []

        for _ in range(self.max_len - 1):
            logits = self._decode_step(tf.constant(sequences), context,
                                       source_mask)[:, -1, :].numpy()
            logits[:, [vocab.pad, vocab.bos]] = -1e9
            shifted = logits - logits.max(-1, keepdims=True)
            log_probs = shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))

            candidates = scores[:, None] + log_probs
            best = candidates.ravel().argsort()[-beam:][::-1]
            source_beam, token = best // log_probs.shape[1], best % log_probs.shape[1]
            sequences = np.concatenate(
                [sequences[source_beam], token[:, None].astype(np.int32)], axis=1)
            scores = candidates.ravel()[best]

            for i, t in enumerate(token):
                if t == vocab.eos:
                    finished.append((scores[i] / (sequences.shape[1] - 1) ** alpha,
                                     sequences[i].copy()))
                    scores[i] = -1e9
            if len(finished) >= beam:
                break

        if not finished:
            finished = [(scores[0] / max(sequences.shape[1] - 1, 1) ** alpha, sequences[0])]
        return vocab.decode(max(finished, key=lambda pair: pair[0])[1][1:])
