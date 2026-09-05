"""The encoder-decoder Transformer of *Attention Is All You Need*.

Every attention layer here is the same operation with different things plugged into it:

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

and the three uses differ only in where Q, K, V come from and what gets masked:

| layer                        | Q from | K, V from | masked           |
|------------------------------|--------|-----------|------------------|
| encoder self-attention       | source | source    | padding          |
| decoder causal self-attention| target | target    | padding + causal |
| cross-attention              | target | source    | source padding   |

**Padding masks are explicit arguments, not Keras's `mask_zero` propagation.** That propagation
stops at the first layer that does not declare `supports_masking`, and every layer below is one,
so the mask would be dropped silently before reaching any attention layer -- the model would
attend over padding while everything still ran. `verify_masking()` at the bottom is the check
that caught exactly that.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from src.config import Config

PAD = 0


def positional_encoding(length: int, depth: int) -> tf.Tensor:
    """Sine and cosine waves of geometrically spaced frequencies.

    Attention is a weighted sum and a sum has no order, so without this "dog bites man" and
    "man bites dog" are the same input.
    """
    half = depth // 2
    positions = np.arange(length)[:, None]
    depths = np.arange(half)[None, :] / half
    angles = positions / (10_000 ** depths)
    return tf.cast(np.concatenate([np.sin(angles), np.cos(angles)], axis=-1), tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
    """Token embedding, scaled by sqrt(d_model), plus the positional signal.

    The scaling keeps the token signal from being drowned out by the positional one, whose values
    are in [-1, 1] while a fresh embedding's are much smaller.
    """

    def __init__(self, vocab_size: int, d_model: int, max_len: int = 2048, **kwargs) -> None:
        super().__init__(**kwargs)
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)   # mask_zero=False
        self.pos_encoding = positional_encoding(max_len, d_model)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x) * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        return x + self.pos_encoding[tf.newaxis, :length, :]


class BaseAttention(tf.keras.layers.Layer):
    """Multi-head attention + residual + layer norm, the shape all three uses share."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class GlobalSelfAttention(BaseAttention):
    """Encoder: the source looking at itself, in both directions.

    `mask` is (batch, source_len); MultiHeadAttention wants (batch, query_len, key_len), and
    (batch, 1, key_len) broadcasts -- every query is barred from the same padded keys.
    """

    def call(self, x, mask):
        return self.layernorm(self.add([x, self.mha(query=x, key=x, value=x,
                                                    attention_mask=mask[:, None, :])]))


class CausalSelfAttention(BaseAttention):
    """Decoder: the target looking at itself, but only backwards.

    `use_causal_mask=True` is the upper-triangular -inf. Without it the model reads the answer off
    position t during teacher forcing: near-zero training loss, and nothing at inference.
    """

    def call(self, x, mask):
        return self.layernorm(self.add([x, self.mha(query=x, key=x, value=x,
                                                    attention_mask=mask[:, None, :],
                                                    use_causal_mask=True)]))


class CrossAttention(BaseAttention):
    """Decoder: the target looking at the source. This is where translation happens.

    The mask is the *source* mask -- the target is asking, so what must be hidden is padding on
    the side being read. The scores are kept because they are the model's own alignment, and
    `Evaluator.alignment_agreement` checks them against the corpus's human one.
    """

    def call(self, x, context, source_mask):
        attended, scores = self.mha(query=x, key=context, value=context,
                                    attention_mask=source_mask[:, None, :],
                                    return_attention_scores=True)
        self.last_attention_scores = scores      # (batch, heads, target_len, source_len)
        return self.layernorm(self.add([x, attended]))


class FeedForward(tf.keras.layers.Layer):
    """Two dense layers per position, widening to dff and back."""

    def __init__(self, d_model: int, dff: int, dropout: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation="relu"),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout),
        ])
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

    def call(self, x):
        return self.layernorm(self.add([x, self.seq(x)]))


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model: int, num_heads: int, dff: int, dropout: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.self_attention = GlobalSelfAttention(num_heads=num_heads,
                                                  key_dim=d_model // num_heads, dropout=dropout)
        self.ffn = FeedForward(d_model, dff, dropout)

    def call(self, x, mask):
        return self.ffn(self.self_attention(x, mask))


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model: int, num_heads: int, dff: int, dropout: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout)
        self.cross_attention = CrossAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout)
        self.ffn = FeedForward(d_model, dff, dropout)

    def call(self, x, context, target_mask, source_mask):
        x = self.causal_self_attention(x, target_mask)
        x = self.cross_attention(x, context, source_mask)
        self.last_attention_scores = self.cross_attention.last_attention_scores
        return self.ffn(x)


class Transformer(tf.keras.Model):
    """Source ids + target-so-far ids -> logits over the target vocabulary."""

    def __init__(self, config: Config, source_vocab: int, target_vocab: int) -> None:
        super().__init__()
        d_model = config.require("model.d_model")
        num_heads = config.require("model.num_heads")
        dff = config.require("model.dff")
        dropout = config.require("model.dropout")
        num_layers = config.require("model.num_layers")

        self.source_embedding = PositionalEmbedding(source_vocab, d_model)
        self.target_embedding = PositionalEmbedding(target_vocab, d_model)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.encoder_layers = [EncoderLayer(d_model, num_heads, dff, dropout)
                               for _ in range(num_layers)]
        self.decoder_layers = [DecoderLayer(d_model, num_heads, dff, dropout)
                               for _ in range(num_layers)]
        # float32 out regardless of the global policy: softmax and cross-entropy lose the most
        # precision in fp16.
        self.final = tf.keras.layers.Dense(target_vocab, dtype="float32")

    def encode(self, source):
        """Returns (encoder output, source padding mask) -- the mask travels with the context."""
        mask = source != PAD
        x = self.dropout(self.source_embedding(source))
        for layer in self.encoder_layers:
            x = layer(x, mask)
        return x, mask

    def decode(self, target_in, context, source_mask):
        target_mask = target_in != PAD
        x = self.dropout(self.target_embedding(target_in))
        for layer in self.decoder_layers:
            x = layer(x, context, target_mask, source_mask)
        return self.final(x)

    def call(self, inputs):
        source, target_in = inputs
        context, source_mask = self.encode(source)
        return self.decode(target_in, context, source_mask)

    def attention_scores(self):
        """Cross-attention from the last call, one entry per decoder layer.

        Each is (batch, heads, target_len, source_len).
        """
        return [layer.last_attention_scores for layer in self.decoder_layers]


def verify_masking(model: Transformer, source: np.ndarray, target_in: np.ndarray,
                   extra_columns: int = 16) -> float:
    """Largest logit change when extra padding is appended to the source. Should be ~0.

    Choosing this test took a wrong turn worth recording. Filling the padded slots with garbage
    does *not* test masking: the mask is derived from the ids, so garbage in a padded slot simply
    becomes a real token and the output is supposed to change. What masking promises is that
    padding is inert, so the test is to add more of it.
    """
    padded = np.concatenate([source, np.zeros((source.shape[0], extra_columns), np.int32)], axis=1)
    return float(np.abs(model((source, target_in)).numpy()
                        - model((padded, target_in)).numpy()).max())
