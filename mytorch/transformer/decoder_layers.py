from mytorch.transformer.sublayers import SelfAttentionLayer, CrossAttentionLayer, FeedForwardLayer


class SelfAttentionDecoderLayer:
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        self.self_attn = SelfAttentionLayer(d_model, num_heads, dropout)
        self.ffn = FeedForwardLayer(d_model, d_ff, dropout)
        self.layers = self.self_attn.layers + self.ffn.layers

    def forward(self, x, key_padding_mask=None, attn_mask=None, eval=False):
        x, self_attn_weights = self.self_attn.forward(
            x, key_padding_mask=key_padding_mask, attn_mask=attn_mask, eval=eval
        )
        x = self.ffn.forward(x, eval=eval)
        return x, self_attn_weights

    def backward(self, dLdOut):
        dLdX = self.ffn.backward(dLdOut)
        dLdX = self.self_attn.backward(dLdX)
        return dLdX


class CrossAttentionDecoderLayer:
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        self.self_attn = SelfAttentionLayer(d_model, num_heads, dropout)
        self.cross_attn = CrossAttentionLayer(d_model, num_heads, dropout)
        self.ffn = FeedForwardLayer(d_model, d_ff, dropout)
        self.layers = self.self_attn.layers + self.cross_attn.layers + self.ffn.layers

    def forward(self, x, enc_output, dec_key_padding_mask=None,
                enc_key_padding_mask=None, attn_mask=None, eval=False):
        x, self_attn_weights = self.self_attn.forward(
            x, key_padding_mask=dec_key_padding_mask, attn_mask=attn_mask, eval=eval
        )
        x, cross_attn_weights = self.cross_attn.forward(
            x, enc_output, key_padding_mask=enc_key_padding_mask, attn_mask=None, eval=eval
        )
        x = self.ffn.forward(x, eval=eval)
        return x, self_attn_weights, cross_attn_weights

    def backward(self, dLdOut):
        dLdX = self.ffn.backward(dLdOut)
        dLdX, dLdEncOutput = self.cross_attn.backward(dLdX)
        dLdX = self.self_attn.backward(dLdX)
        return dLdX, dLdEncOutput