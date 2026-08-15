"""
Transformer encoder layer, composed from the same sublayers used by the
decoder.

Only structural difference from SelfAttentionDecoderLayer: no causal
mask. Encoding is not autoregressive, so every position is allowed to
attend to every other position (bidirectional context)
"""

from .sublayers import SelfAttentionLayer, FeedForwardLayer


class SelfAttentionEncoderLayer:
    """
    Pre-LN Encoder Layer with self-attention mechanism.
    Used in the encoder part of encoder-decoder Transformer architectures (HW4P2).
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        self.self_attn = SelfAttentionLayer(d_model, num_heads, dropout)
        self.ffn = FeedForwardLayer(d_model, d_ff, dropout)
        self.layers = self.self_attn.layers + self.ffn.layers

    def forward(self, x, key_padding_mask=None, eval=False):
        """
        :param x: (N, T, d_model)
        :param key_padding_mask: (N, T) or None -- no attn_mask, unlike the decoder:
                                  the encoder attends bidirectionally, so there is
                                  nothing to hide except padding.
        :param eval: True disables dropout (inference mode)
        :return: (x, self_attn_weights)
            x: (N, T, d_model)
            self_attn_weights: (N, H, T, T)
        """
        x, self_attn_weights = self.self_attn.forward(
            x, key_padding_mask=key_padding_mask, attn_mask=None, eval=eval
        )
        x = self.ffn.forward(x, eval=eval)

        return x, self_attn_weights

    def backward(self, dLdOut):
        """
        :param dLdOut: gradient of loss wrt this layer's output, (N, T, d_model)
        :return: gradient of loss wrt this layer's input x, (N, T, d_model)
        """
        dLdX = self.ffn.backward(dLdOut)
        dLdX = self.self_attn.backward(dLdX)
        return dLdX