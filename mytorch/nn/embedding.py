import numpy as np


class Embedding:
    """
    Token embedding lookup table: maps integer token indices to learned
    d_model-dimensional vectors.

    Forward is just indexing into the weight table: no matmul needed,
    since a one-hot-vector @ W is equivalent to (and much cheaper than)
    fancy-indexing W by the token id directly.

    Backward has one real subtlety: if the same token index appears more
    than once in a batch, the gradients from every occurrence must be
    SUMMED into that row of dLdW, not overwritten. Plain fancy-index
    assignment (dLdW[ids] = grad) silently drops all but one occurrence
    when ids repeat -- np.add.at is required for correct accumulation.
    """

    def __init__(self, vocab_size, d_model):
        self.vocab_size = vocab_size
        self.d_model = d_model
        # Small random init, no bias -- an embedding table has no bias term.
        self.W = np.random.randn(vocab_size, d_model) * 0.01
        self.dLdW = np.zeros_like(self.W)

    def forward(self, ids):
        """
        :param ids: integer array of token indices, shape (*)
        :return: embedded vectors, shape (*, d_model)
        """
        self.ids = ids
        return self.W[ids]

    def backward(self, dLdOut):
        """
        :param dLdOut: gradient of loss wrt the embedded output, shape (*, d_model)
        :return: None (token indices are not differentiable; this is always
                 the first layer, so there is nothing further to propagate to)
        """
        self.dLdW = np.zeros_like(self.W)
        # np.add.at accumulates correctly even when self.ids has repeats,
        # unlike self.dLdW[self.ids] += dLdOut, which silently overwrites
        # instead of accumulating for repeated indices.
        np.add.at(self.dLdW, self.ids, dLdOut)
        return None