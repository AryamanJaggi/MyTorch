import numpy as np


class LayerNorm:
    """
    Layer normalization: normalizes each sample independently across its
    feature dimension (the last axis), unlike BatchNorm which normalizes
    each feature independently across the batch dimension (axis 0).

    This is the norm variant used inside the pre-norm transformer sublayers,
    since it works identically regardless of batch size or sequence length,
    and normalizes over d_model for every (batch, time) position.
    """

    def __init__(self, num_features, eps=1e-5):
        self.num_features = num_features
        self.eps = eps

        self.BW = np.ones((num_features,))
        self.Bb = np.zeros((num_features,))
        self.dLdBW = np.zeros((num_features,))
        self.dLdBb = np.zeros((num_features,))

    def forward(self, Z):
        """
        Forward pass for layer normalization.
        :param Z: input data Z of shape (*, num_features). Normalization is
                   computed independently for each "row" (every index along
                   the leading dims *), over the last axis only.
        :return: layer-normalized, scaled and shifted data. Same shape as Z.
        """
        self.Z = Z
        self.D = Z.shape[-1]  # size of the normalized axis (num_features)

        self.M = np.mean(Z, axis=-1, keepdims=True)  # per-row mean, shape (*, 1)
        dif = Z - self.M
        self.V = np.mean(dif * dif, axis=-1, keepdims=True)  # per-row variance, shape (*, 1)

        self.NZ = (Z - self.M) / np.sqrt(self.V + self.eps)  # normalized input
        self.BZ = self.NZ * self.BW + self.Bb  # scale and shift

        return self.BZ

    def backward(self, dLdBZ):
        """
        Backward pass for layer normalization.
        :param dLdBZ: gradient of loss wrt output of LayerNorm, shape (*, num_features).
        :return: gradient of loss wrt input Z, same shape as Z.
        """
        # Sum over every leading (*) dim to collapse down to per-feature
        # gradients of shape (num_features,) for the learnable parameters.
        sum_axes = tuple(range(dLdBZ.ndim - 1))

        self.dLdBb = np.sum(dLdBZ, axis=sum_axes)
        self.dLdBW = np.sum(dLdBZ * self.NZ, axis=sum_axes)

        dLdNZ = dLdBZ * self.BW

        # Same structure as BatchNorm1d's backward, but the "reduction axis"
        # is the last (feature) axis instead of axis 0, and D (num_features)
        # plays the role that N (batch size) played there.
        dLdV = np.sum(
            dLdNZ * (-0.5) * np.power(self.V + self.eps, -1.5) * (self.Z - self.M),
            axis=-1, keepdims=True
        )
        dLdM = np.sum(dLdNZ * (-1 / np.sqrt(self.V + self.eps)), axis=-1, keepdims=True) \
            + dLdV * (-2 / self.D) * np.sum(self.Z - self.M, axis=-1, keepdims=True)

        dLdZ = dLdNZ / np.sqrt(self.V + self.eps) \
            + dLdV * 2 * (self.Z - self.M) / self.D \
            + dLdM / self.D

        return dLdZ