"""The attention mechanism, written out rather than imported -- this is the exercise.

Attention is a weighted lookup. The decoder asks a question (its state at step t), every encoder
position offers an answer, a score function says how well each one matches, softmax turns the
scores into weights that sum to 1, and the answer is the weighted average:

    alpha_t = softmax(score(s_t, h_1..h_S)),    c_t = sum_i alpha_t,i * h_i

Only `score` differs between the two classes here, and the difference is not academic:

* **Additive** (Bahdanau, 2015) scores with a small feed-forward network. Vectorised over all
  output steps at once it builds a `(batch, summary_len, article_len, attn_units)` tensor --
  at 32 x 63 x 256 x 64 that is 33 M floats, 132 MB, before the backward pass wants its copies.
* **Scaled dot-product** is one matmul and never leaves `(batch, summary_len, article_len)`,
  132x smaller here. That is exactly the argument *Attention Is All You Need* gives for choosing
  it (section 3.2.1), and the `sqrt(d)` is there because without it large dimensions push the
  dot products into the flat tails of the softmax, where the gradient is nearly zero.
"""

from __future__ import annotations

import math

import tensorflow as tf


def attend(scores, enc_out, enc_mask):
    """Mask padding, softmax, and take the weighted average of the encoder states.

    The softmax runs in float32 deliberately. Under `mixed_float16` the -1e9 that kills padded
    positions is past fp16's minimum of -65504, so it becomes -inf and the softmax returns NaN --
    a loss that turns to NaN with nothing visibly wrong in the model.
    """
    scores = tf.cast(scores, tf.float32)
    scores = tf.where(enc_mask[:, None, :], scores, -1e9)
    weights = tf.nn.softmax(scores, axis=-1)                        # (B, T, S)
    context = tf.matmul(tf.cast(weights, enc_out.dtype), enc_out)   # (B, T, H)
    return context, weights


class AdditiveAttention(tf.keras.layers.Layer):
    """Bahdanau (2015) scoring: v^T tanh(W1 h_i + W2 s_t)."""

    # The encoder mask arrives as an explicit argument, not through Keras's mask plumbing.
    # Without this, Keras warns that the layer is "destroying" a mask it was never using.
    supports_masking = True

    def __init__(self, units: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.W1 = tf.keras.layers.Dense(units, use_bias=False)   # over encoder states
        self.W2 = tf.keras.layers.Dense(units, use_bias=False)   # over decoder states
        self.v = tf.keras.layers.Dense(1, use_bias=False)

    def call(self, dec_out, enc_out, enc_mask):
        # dec_out (B, T, H) | enc_out (B, S, H) | enc_mask (B, S) bool, True where real
        keys = self.W1(enc_out)[:, None, :, :]                   # (B, 1, S, A)
        queries = self.W2(dec_out)[:, :, None, :]                # (B, T, 1, A)
        scores = tf.squeeze(self.v(tf.tanh(keys + queries)), -1)  # (B, T, S)
        return attend(scores, enc_out, enc_mask)


class DotProductAttention(tf.keras.layers.Layer):
    """Scaled dot-product scoring: (s_t . h_i) / sqrt(d). No parameters at all."""

    supports_masking = True

    def call(self, dec_out, enc_out, enc_mask):
        scores = tf.matmul(dec_out, enc_out, transpose_b=True)   # (B, T, S)
        return attend(scores / math.sqrt(enc_out.shape[-1]), enc_out, enc_mask)


def build_attention(kind: str, config):
    """`none` gives the control model: the same network with no attention at all."""
    if kind == "none":
        return None
    if kind == "additive":
        return AdditiveAttention(config.require("model.attn_units"))
    if kind == "dot":
        return DotProductAttention()
    raise ValueError(f"unknown attention kind {kind!r}; expected none, additive or dot")
