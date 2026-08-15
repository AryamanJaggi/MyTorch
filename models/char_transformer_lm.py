"""
A small decoder-only (GPT-style) Transformer language model, built entirely
out of mytorch. This is meant as a worked example of composing the library end-to-end: 
token embedding, positional encoding, a stack of causally-masked self-attention decoder
layers, a final norm, and an output projection to vocabulary logits.

Follows the same shape as the other models/ files (CNN, etc.): the model
owns its own layers explicitly, exposes .layers for the optimizer, and
implements forward/backward by hand since there's no autograd.
"""

import numpy as np
from mytorch.nn.embedding import Embedding
from mytorch.nn.linear import Linear
from mytorch.nn.dropout import Dropout
from mytorch.nn.layernorm import LayerNorm
from mytorch.nn.loss import CrossEntropyLoss
from mytorch.transformer.positional_encoding import PositionalEncoding
from mytorch.transformer.decoder_layers import SelfAttentionDecoderLayer
from mytorch.transformer.masks import CausalMask, PadMask


class CharTransformerLM:
    """
    Decoder-only Transformer for character-level (or any token-level)
    autoregressive language modeling.
    """

    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers,
                 max_len, dropout=0.1, pad_id=0):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_len = max_len
        self.pad_id = pad_id
        self.train_mode = True

        self.embedding = Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = Dropout(dropout)

        self.dec_layers = [
            SelfAttentionDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ]

        self.norm = LayerNorm(d_model)
        self.final_linear = Linear(d_model, vocab_size)

        self.criterion = CrossEntropyLoss()

        # Flat list of every leaf parametrized sub-layer, for the optimizer.
        self.layers = [self.embedding]
        for dec_layer in self.dec_layers:
            self.layers += dec_layer.layers
        self.layers += [self.norm, self.final_linear]

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

    def forward(self, token_ids, lengths=None):
        """
        :param token_ids: integer array of shape (N, T) -- right-padded token indices
        :param lengths: optional array of shape (N,) -- true (unpadded) lengths
        :return: (logits, attn_weights)
            logits: (N, T, vocab_size)
            attn_weights: dict of per-layer self-attention weight arrays
        """
        eval_mode = not self.train_mode

        pad_mask = None
        if lengths is not None:
            pad_mask = PadMask(token_ids, lengths)
        causal_mask = CausalMask(token_ids)

        x = self.embedding.forward(token_ids)
        x = self.positional_encoding.forward(x)
        x = self.dropout.forward(x, eval=eval_mode)

        attn_weights = {}
        for i, dec_layer in enumerate(self.dec_layers):
            x, attn = dec_layer.forward(
                x, key_padding_mask=pad_mask, attn_mask=causal_mask, eval=eval_mode
            )
            attn_weights[f"layer{i+1}_self_attn"] = attn

        x = self.norm.forward(x)
        logits = self.final_linear.forward(x)

        self.logits = logits
        return logits, attn_weights

    def backward(self, targets, lengths=None):
        """
        Compute cross-entropy loss against golden targets, ignoring padded
        positions entirely (equivalent to PyTorch's ignore_index), and
        backpropagate through the whole model.

        :param targets: integer array of shape (N, T) -- golden next-token ids
        :param lengths: optional array of shape (N,) -- true (unpadded) lengths
        :return: scalar mean per-token cross-entropy loss (over valid tokens only)
        """
        N, T, V = self.logits.shape
        flat_logits = self.logits.reshape(N * T, V)
        flat_targets = targets.reshape(N * T)

        if lengths is not None:
            positions = np.arange(T)[np.newaxis, :]
            valid_mask = (positions < np.asarray(lengths)[:, np.newaxis]).reshape(N * T)
        else:
            valid_mask = np.ones(N * T, dtype=bool)

        valid_logits = flat_logits[valid_mask]
        valid_targets_idx = flat_targets[valid_mask]

        # one-hot encode the valid targets for the existing CrossEntropyLoss interface
        valid_Y = np.zeros((valid_logits.shape[0], V))
        valid_Y[np.arange(valid_logits.shape[0]), valid_targets_idx] = 1.0

        loss = self.criterion.forward(valid_logits, valid_Y)
        dLd_valid_logits = self.criterion.backward()

        # scatter the gradient back into a full (N*T, V) array, zero at padded rows
        dLd_flat_logits = np.zeros_like(flat_logits)
        dLd_flat_logits[valid_mask] = dLd_valid_logits
        dLdLogits = dLd_flat_logits.reshape(N, T, V)

        dLdX = self.final_linear.backward(dLdLogits)
        dLdX = self.norm.backward(dLdX)

        for dec_layer in reversed(self.dec_layers):
            dLdX = dec_layer.backward(dLdX)

        dLdX = self.dropout.backward(dLdX)
        # positional_encoding is a pure addition, gradient passes through unchanged
        self.embedding.backward(dLdX)

        return loss

    def generate_greedy(self, prompt_ids, max_new_tokens, eos_id=None):
        """
        Simple greedy autoregressive generation: repeatedly predict the next
        token and append it.

        :param prompt_ids: integer array of shape (1, T0)
        :param max_new_tokens: how many tokens to generate beyond the prompt
        :param eos_id: optional token id to stop generation early on
        :return: integer array of shape (1, T0 + generated_len)
        """
        was_training = self.train_mode
        self.eval()

        x = prompt_ids.copy()
        for _ in range(max_new_tokens):
            logits, _ = self.forward(x, lengths=None)
            next_token_logits = logits[:, -1, :]
            next_token = np.argmax(next_token_logits, axis=-1, keepdims=True)
            x = np.concatenate([x, next_token], axis=1)
            if eos_id is not None and next_token.item() == eos_id:
                break

        if was_training:
            self.train()

        return x