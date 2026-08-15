"""
Transformer decoder sublayers, built entirely from mytorch components
on our own from-scratch

Each layer follows a Pre-LN architecture:
    y = x + Dropout(SubLayer(LayerNorm(x)))
i.e. normalize first, run the core operation, dropout the result, then
add it back to the *original, un-normalized* input via a residual connection.
"""

from mytorch.nn.linear import Linear
from .multi_head_attention import MultiHeadAttention
from mytorch.nn.activation import GELU
from mytorch.nn.dropout import Dropout
from mytorch.nn.layernorm import LayerNorm


class SelfAttentionLayer:
    """
    Pre-LN Decoder Sub-Layer 1: causally-masked self-attention.
    """

    def __init__(self, d_model, num_heads, dropout=0.0):
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.norm = LayerNorm(d_model)
        self.dropout = Dropout(dropout)
        self.layers = self.mha.layers + [self.norm]

    def forward(self, x, key_padding_mask=None, attn_mask=None, eval=False):
        """
        :param x: (N, L, d_model)
        :param key_padding_mask: (N, L) or None
        :param attn_mask: (L, L) or None
        :param eval: True disables dropout (inference mode)
        :return: (out, attn_weights)
            out: (N, L, d_model)
            attn_weights: (N, H, L, L) attention scores from the mha call
        """
        residual = x

        normed = self.norm.forward(x)
        # self-attention: query, key, and value are all the same normalized tensor
        attn_out = self.mha.forward(normed, normed, normed,
                                     key_padding_mask=key_padding_mask,
                                     attn_mask=attn_mask)
        attn_weights = self.mha.attention.attention_scores

        dropped = self.dropout.forward(attn_out, eval=eval)
        out = residual + dropped

        return out, attn_weights

    def backward(self, dLdOut):
        """
        :param dLdOut: gradient of loss wrt this layer's output, (N, L, d_model)
        :return: gradient of loss wrt this layer's input x, (N, L, d_model)
        """
        # Residual connection: gradient flows unchanged to x on one branch,
        # and through the sublayer on the other. Both contributions get summed.
        dLdResidual = dLdOut
        dLdDropped = dLdOut

        dLdAttnOut = self.dropout.backward(dLdDropped)

        # normed was fed into mha as q, k, AND v -- three separate uses of the
        # same tensor, so their gradients must be summed (multivariable chain rule).
        dLdNormed_q, dLdNormed_k, dLdNormed_v = self.mha.backward(dLdAttnOut)
        dLdNormed = dLdNormed_q + dLdNormed_k + dLdNormed_v

        dLdX_from_norm = self.norm.backward(dLdNormed)

        dLdX = dLdResidual + dLdX_from_norm
        return dLdX


class CrossAttentionLayer:
    """
    Pre-LN Decoder Sub-Layer 2: cross-attention between decoder (query) and
    encoder output (key/value).
    """

    def __init__(self, d_model, num_heads, dropout=0.0):
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.norm = LayerNorm(d_model)
        self.dropout = Dropout(dropout)
        self.layers = self.mha.layers + [self.norm]

    def forward(self, x, y, key_padding_mask=None, attn_mask=None, eval=False):
        """
        :param x: decoder input (query stream), (N, L, d_model)
        :param y: encoder output (key/value stream), (N, S, d_model)
        :param key_padding_mask: (N, S) or None -- masks padded encoder positions
        :param attn_mask: (L, S) or None
        :param eval: True disables dropout (inference mode)
        :return: (out, attn_weights)
            out: (N, L, d_model)
            attn_weights: (N, H, L, S)
        """
        residual = x

        # Only the decoder (query) stream is pre-normalized here; the encoder
        # output y is used as-is, since it already carries its own final norm
        # from the top of the encoder stack.
        normed_x = self.norm.forward(x)
        attn_out = self.mha.forward(normed_x, y, y,
                                     key_padding_mask=key_padding_mask,
                                     attn_mask=attn_mask)
        attn_weights = self.mha.attention.attention_scores

        dropped = self.dropout.forward(attn_out, eval=eval)
        out = residual + dropped

        return out, attn_weights

    def backward(self, dLdOut):
        """
        :param dLdOut: gradient of loss wrt this layer's output, (N, L, d_model)
        :return: (dLdX, dLdY)
            dLdX: gradient wrt decoder input x, (N, L, d_model)
            dLdY: gradient wrt encoder output y, (N, S, d_model)
        """
        dLdResidual = dLdOut
        dLdDropped = dLdOut

        dLdAttnOut = self.dropout.backward(dLdDropped)

        # q came from normed_x; k and v both came from the same y tensor,
        # so their gradients must be summed.
        dLdNormedX, dLdY_k, dLdY_v = self.mha.backward(dLdAttnOut)
        dLdY = dLdY_k + dLdY_v

        dLdX_from_norm = self.norm.backward(dLdNormedX)
        dLdX = dLdResidual + dLdX_from_norm

        return dLdX, dLdY


class FeedForwardLayer:
    """
    Pre-LN Decoder Sub-Layer 3: position-wise feed-forward network.
    ffn: Linear(d_model -> d_ff) -> GELU -> Dropout -> Linear(d_ff -> d_model)
    """

    def __init__(self, d_model, d_ff, dropout=0.0):
        self.linear1 = Linear(d_model, d_ff)
        self.activation = GELU()
        self.ffn_dropout = Dropout(dropout)
        self.linear2 = Linear(d_ff, d_model)

        self.norm = LayerNorm(d_model)
        self.dropout = Dropout(dropout)

        self.layers = [self.linear1, self.linear2, self.norm]

    def forward(self, x, eval=False):
        """
        :param x: (N, L, d_model)
        :param eval: True disables dropout (inference mode)
        :return: (N, L, d_model)
        """
        residual = x

        normed = self.norm.forward(x)
        h = self.linear1.forward(normed)
        h = self.activation.forward(h)
        h = self.ffn_dropout.forward(h, eval=eval)
        h = self.linear2.forward(h)

        dropped = self.dropout.forward(h, eval=eval)
        out = residual + dropped

        return out

    def backward(self, dLdOut):
        """
        :param dLdOut: gradient of loss wrt this layer's output, (N, L, d_model)
        :return: gradient of loss wrt this layer's input x, (N, L, d_model)
        """
        dLdResidual = dLdOut
        dLdDropped = dLdOut

        dLdH = self.dropout.backward(dLdDropped)
        dLdH = self.linear2.backward(dLdH)
        dLdH = self.ffn_dropout.backward(dLdH)
        dLdH = self.activation.backward(dLdH)
        dLdNormed = self.linear1.backward(dLdH)

        dLdX_from_norm = self.norm.backward(dLdNormed)
        dLdX = dLdResidual + dLdX_from_norm

        return dLdX